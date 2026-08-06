package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.UUID
import kotlin.math.sqrt

/*
 * OSNet으로 사람 crop의 착장 기반 특징 벡터를 생성하고,
 * 이전에 저장된 사람의 특징 벡터와 비교한다.
 *
 * 이 클래스의 gallery에는 실제 익명화 crop 저장에 성공한
 * 사람만 등록한다.
 */
class OsNetReIdentifier(
    context: Context,
    /*
     * 임시 초기값이다.
     *
     * 실제 동일인/타인 crop을 수집한 후 유사도 분포를 확인해
     * 반드시 조정해야 한다.
     */
    private val duplicateSimilarityThreshold: Float = 0.75f,
) : Closeable {

    companion object {
        private const val TAG =
            "CameraAccess:OsNet"

        private const val MODEL_ASSET_PATH =
            "osnet_x1_0_msmt17.onnx"

        /*
         * Torchreid OSNet의 일반적인 person re-identification 입력 크기.
         *
         * 세로 256, 가로 128이다.
         */
        private const val INPUT_HEIGHT =
            256

        private const val INPUT_WIDTH =
            128

        /*
         * ImageNet 정규화 값.
         *
         * 각 RGB 값을 0~1로 변환한 뒤:
         *
         * normalized = (value - mean) / std
         */
        private val CHANNEL_MEANS =
            floatArrayOf(
                0.485f,
                0.456f,
                0.406f,
            )

        private val CHANNEL_STDS =
            floatArrayOf(
                0.229f,
                0.224f,
                0.225f,
            )
    }

    /*
     * 중복으로 판단된 사람 정보.
     */
    data class DuplicateMatch(
        val personId: String,
        val similarity: Float,
        val savedFileName: String?,
        val registeredAtMs: Long,
    )

    /*
     * 저장 완료된 사람 한 명의 OSNet 정보.
     *
     * 앱 메모리에서만 유지하며 앱을 완전히 종료하면 초기화된다.
     */
    private data class SavedPerson(
        val personId: String,
        val embedding: FloatArray,
        val savedFileName: String?,
        val registeredAtMs: Long,
    )

    private val applicationContext =
        context.applicationContext

    private val ortEnvironment: OrtEnvironment =
        OrtEnvironment.getEnvironment()

    private val modelBytes: ByteArray =
        applicationContext.assets
            .open(MODEL_ASSET_PATH)
            .use { inputStream ->
                inputStream.readBytes()
            }

    private val ortSession: OrtSession =
        OrtSession.SessionOptions().use { sessionOptions ->
            ortEnvironment.createSession(
                modelBytes,
                sessionOptions,
            )
        }

    /*
     * export할 때 "images"로 지정했더라도,
     * 실제 ONNX 내부 이름을 자동으로 읽어 사용한다.
     */
    private val inputName: String =
        ortSession.inputNames.firstOrNull()
            ?: throw IllegalStateException(
                "OSNet 입력 노드를 찾지 못했습니다.",
            )

    private val outputName: String =
        ortSession.outputNames.firstOrNull()
            ?: throw IllegalStateException(
                "OSNet 출력 노드를 찾지 못했습니다.",
            )

    /*
     * 실제 익명화 crop 저장이 끝난 사람들의 특징 벡터.
     */
    private val savedPeople =
        mutableListOf<SavedPerson>()

    init {
        require(
            duplicateSimilarityThreshold in -1f..1f,
        ) {
            "OSNet cosine similarity threshold는 -1~1이어야 합니다."
        }

        Log.d(
            TAG,
            "OSNet model loaded: $MODEL_ASSET_PATH",
        )

        Log.d(
            TAG,
            "OSNet duplicate threshold: " +
                    duplicateSimilarityThreshold,
        )

        logModelInformation()
    }

    /**
     * 사람 crop에서 OSNet 특징 벡터를 생성한다.
     *
     * 반환되는 벡터는 L2 normalization이 적용되어 있다.
     */
    @Synchronized
    fun extractEmbedding(
        personBitmap: Bitmap,
    ): FloatArray {
        require(
            !personBitmap.isRecycled,
        ) {
            "재활용된 Bitmap에는 OSNet을 실행할 수 없습니다."
        }

        require(
            personBitmap.width > 0 &&
                    personBitmap.height > 0,
        ) {
            "OSNet 입력 Bitmap의 크기가 올바르지 않습니다."
        }

        val inputBuffer =
            preprocessBitmap(
                bitmap = personBitmap,
            )

        inputBuffer.rewind()

        val inputShape =
            longArrayOf(
                1,
                3,
                INPUT_HEIGHT.toLong(),
                INPUT_WIDTH.toLong(),
            )

        val inferenceStartedAtMs =
            SystemClock.elapsedRealtime()

        val rawEmbedding =
            OnnxTensor.createTensor(
                ortEnvironment,
                inputBuffer,
                inputShape,
            ).use { inputTensor ->

                val inputs =
                    mapOf(
                        inputName to inputTensor,
                    )

                ortSession.run(inputs).use { result ->
                    val outputTensor =
                        result[0] as? OnnxTensor
                            ?: throw IllegalStateException(
                                "OSNet 출력이 OnnxTensor가 아닙니다.",
                            )

                    val outputBuffer =
                        outputTensor.floatBuffer
                            ?: throw IllegalStateException(
                                "OSNet Float 출력을 읽을 수 없습니다.",
                            )

                    outputBuffer.rewind()

                    val embeddingSize =
                        outputBuffer.remaining()

                    if (embeddingSize <= 0) {
                        throw IllegalStateException(
                            "OSNet 출력 특징 벡터가 비어 있습니다.",
                        )
                    }

                    val embedding =
                        FloatArray(
                            embeddingSize,
                        )

                    outputBuffer.get(
                        embedding,
                    )

                    embedding
                }
            }

        val normalizedEmbedding =
            normalizeEmbedding(
                rawEmbedding,
            )

        val elapsedMs =
            SystemClock.elapsedRealtime() -
                    inferenceStartedAtMs

        Log.d(
            TAG,
            "OSNet inference completed: " +
                    "time=${elapsedMs}ms, " +
                    "embeddingSize=${normalizedEmbedding.size}",
        )

        return normalizedEmbedding
    }

    /**
     * 전달받은 embedding과 gallery의 모든 사람을 비교한다.
     *
     * 기준값 이상인 사람 중 가장 유사한 사람을 반환한다.
     * 중복이 없으면 null을 반환한다.
     */
    @Synchronized
    fun findDuplicate(
        embedding: FloatArray,
    ): DuplicateMatch? {
        require(
            embedding.isNotEmpty(),
        ) {
            "비어 있는 OSNet embedding은 비교할 수 없습니다."
        }

        var bestPerson: SavedPerson? =
            null

        var bestSimilarity =
            -1f

        savedPeople.forEach { savedPerson ->
            if (
                savedPerson.embedding.size !=
                embedding.size
            ) {
                Log.w(
                    TAG,
                    "Embedding size mismatch: " +
                            "saved=${savedPerson.embedding.size}, " +
                            "new=${embedding.size}",
                )

                return@forEach
            }

            val similarity =
                cosineSimilarity(
                    first = embedding,
                    second = savedPerson.embedding,
                )

            if (similarity > bestSimilarity) {
                bestSimilarity = similarity
                bestPerson = savedPerson
            }
        }

        val matchedPerson =
            bestPerson
                ?: return null

        if (
            bestSimilarity <
            duplicateSimilarityThreshold
        ) {
            Log.d(
                TAG,
                "No duplicate: " +
                        "bestSimilarity=$bestSimilarity, " +
                        "threshold=$duplicateSimilarityThreshold",
            )

            return null
        }

        Log.d(
            TAG,
            "Duplicate person found: " +
                    "personId=${matchedPerson.personId}, " +
                    "similarity=$bestSimilarity",
        )

        return DuplicateMatch(
            personId =
                matchedPerson.personId,

            similarity =
                bestSimilarity,

            savedFileName =
                matchedPerson.savedFileName,

            registeredAtMs =
                matchedPerson.registeredAtMs,
        )
    }

    /**
     * 중복인지 여부만 필요할 때 사용한다.
     */
    @Synchronized
    fun isDuplicate(
        embedding: FloatArray,
    ): Boolean {
        return findDuplicate(
            embedding = embedding,
        ) != null
    }

    /**
     * 익명화 crop 저장이 성공한 뒤에만 호출한다.
     *
     * 저장 전에 등록하면 촬영 또는 저장 실패 시
     * 실제 파일이 없는 사람도 중복 처리될 수 있다.
     */
    @Synchronized
    fun registerSavedPerson(
        embedding: FloatArray,
        savedFileName: String?,
    ): String {
        require(
            embedding.isNotEmpty(),
        ) {
            "비어 있는 OSNet embedding은 등록할 수 없습니다."
        }

        val normalizedEmbedding =
            normalizeEmbedding(
                embedding,
            )

        val personId =
            UUID.randomUUID()
                .toString()

        savedPeople.add(
            SavedPerson(
                personId = personId,
                embedding =
                    normalizedEmbedding.copyOf(),
                savedFileName = savedFileName,
                registeredAtMs =
                    System.currentTimeMillis(),
            ),
        )

        Log.d(
            TAG,
            "Saved person registered: " +
                    "personId=$personId, " +
                    "gallerySize=${savedPeople.size}, " +
                    "file=$savedFileName",
        )

        return personId
    }

    /**
     * 현재 앱 실행 중 등록된 사람 수를 반환한다.
     */
    @Synchronized
    fun getRegisteredPersonCount(): Int {
        return savedPeople.size
    }

    /**
     * 현재 메모리의 중복 판정 목록을 초기화한다.
     *
     * 실제 JPEG 파일은 삭제하지 않는다.
     */
    @Synchronized
    fun clearGallery() {
        savedPeople.clear()

        Log.d(
            TAG,
            "OSNet gallery cleared",
        )
    }

    /**
     * Bitmap을 OSNet 입력 FloatBuffer로 변환한다.
     *
     * 입력:
     * [1, 3, 256, 128]
     *
     * 배열 순서:
     * R 전체 → G 전체 → B 전체
     */
    private fun preprocessBitmap(
        bitmap: Bitmap,
    ): FloatBuffer {
        val resizedBitmap =
            Bitmap.createScaledBitmap(
                bitmap,
                INPUT_WIDTH,
                INPUT_HEIGHT,
                true,
            )

        try {
            val pixelCount =
                INPUT_WIDTH *
                        INPUT_HEIGHT

            val pixels =
                IntArray(
                    pixelCount,
                )

            resizedBitmap.getPixels(
                pixels,
                0,
                INPUT_WIDTH,
                0,
                0,
                INPUT_WIDTH,
                INPUT_HEIGHT,
            )

            val inputBuffer =
                ByteBuffer
                    .allocateDirect(
                        3 *
                                pixelCount *
                                Float.SIZE_BYTES,
                    )
                    .order(
                        ByteOrder.nativeOrder(),
                    )
                    .asFloatBuffer()

            /*
             * R 채널.
             */
            pixels.forEach { pixel ->
                val value =
                    Color.red(pixel) /
                            255.0f

                inputBuffer.put(
                    (
                            value -
                                    CHANNEL_MEANS[0]
                            ) /
                            CHANNEL_STDS[0],
                )
            }

            /*
             * G 채널.
             */
            pixels.forEach { pixel ->
                val value =
                    Color.green(pixel) /
                            255.0f

                inputBuffer.put(
                    (
                            value -
                                    CHANNEL_MEANS[1]
                            ) /
                            CHANNEL_STDS[1],
                )
            }

            /*
             * B 채널.
             */
            pixels.forEach { pixel ->
                val value =
                    Color.blue(pixel) /
                            255.0f

                inputBuffer.put(
                    (
                            value -
                                    CHANNEL_MEANS[2]
                            ) /
                            CHANNEL_STDS[2],
                )
            }

            inputBuffer.rewind()

            return inputBuffer
        } finally {
            if (
                resizedBitmap !== bitmap &&
                !resizedBitmap.isRecycled
            ) {
                resizedBitmap.recycle()
            }
        }
    }

    /**
     * 특징 벡터의 길이가 1이 되도록 정규화한다.
     */
    private fun normalizeEmbedding(
        embedding: FloatArray,
    ): FloatArray {
        var squaredSum =
            0.0

        embedding.forEach { value ->
            squaredSum +=
                value.toDouble() *
                        value.toDouble()
        }

        val norm =
            sqrt(
                squaredSum,
            )

        if (
            norm <= 1e-12
        ) {
            throw IllegalStateException(
                "OSNet embedding의 norm이 0입니다.",
            )
        }

        return FloatArray(
            embedding.size,
        ) { index ->
            (
                    embedding[index] /
                            norm
                    ).toFloat()
        }
    }

    /**
     * 두 L2-normalized embedding의 cosine similarity를 계산한다.
     */
    private fun cosineSimilarity(
        first: FloatArray,
        second: FloatArray,
    ): Float {
        require(
            first.size ==
                    second.size,
        ) {
            "서로 다른 크기의 embedding은 비교할 수 없습니다."
        }

        var dotProduct =
            0.0

        for (
        index in first.indices
        ) {
            dotProduct +=
                first[index].toDouble() *
                        second[index].toDouble()
        }

        return dotProduct
            .toFloat()
            .coerceIn(
                -1f,
                1f,
            )
    }

    /**
     * 실제 ONNX 모델의 입력·출력 이름과 shape를 Logcat에 표시한다.
     */
    private fun logModelInformation() {
        ortSession.inputInfo.forEach {
                (name, nodeInfo) ->

            val tensorInfo =
                nodeInfo.info as? TensorInfo

            Log.d(
                TAG,
                "Input: " +
                        "name=$name, " +
                        "shape=${tensorInfo?.shape?.contentToString()}, " +
                        "type=${tensorInfo?.type}",
            )
        }

        ortSession.outputInfo.forEach {
                (name, nodeInfo) ->

            val tensorInfo =
                nodeInfo.info as? TensorInfo

            Log.d(
                TAG,
                "Output: " +
                        "name=$name, " +
                        "shape=${tensorInfo?.shape?.contentToString()}, " +
                        "type=${tensorInfo?.type}",
            )
        }

        Log.d(
            TAG,
            "Selected nodes: " +
                    "input=$inputName, " +
                    "output=$outputName",
        )
    }

    override fun close() {
        clearGallery()

        ortSession.close()

        Log.d(
            TAG,
            "OSNet model closed",
        )
    }
}