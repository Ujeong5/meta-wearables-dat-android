package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
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
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class PoseKeypoint(
    val x: Float,
    val y: Float,
    val confidence: Float,
)

data class PoseDetection(
    val box: RectF,
    val confidence: Float,
    val keypoints: List<PoseKeypoint>,
)

class YoloPoseModel(
    context: Context,
) : Closeable {

    companion object {
        private const val TAG = "YoloPoseModel"

        private const val MODEL_FILE_NAME = "yolo11n-pose.onnx"

        // 모델 입력 크기
        private const val INPUT_SIZE = 640

        // 출력 형태: [1, 56, 8400]
        private const val OUTPUT_CHANNELS = 56
        private const val NUMBER_OF_CANDIDATES = 8400

        // 사람 검출 최소 신뢰도
        private const val CONFIDENCE_THRESHOLD = 0.25f

        // 겹치는 박스를 제거하는 기준
        private const val NMS_THRESHOLD = 0.45f

        // 한 프레임에서 최대로 남길 사람 수
        private const val MAX_DETECTIONS = 20
    }

    private val ortEnvironment: OrtEnvironment =
        OrtEnvironment.getEnvironment()

    private val modelBytes: ByteArray =
        context.assets.open(MODEL_FILE_NAME).use { inputStream ->
            inputStream.readBytes()
        }

    private val ortSession: OrtSession =
        OrtSession.SessionOptions().use { sessionOptions ->
            ortEnvironment.createSession(
                modelBytes,
                sessionOptions,
            )
        }

    private val inputName: String =
        ortSession.inputNames.first()

    init {
        Log.d(TAG, "YOLO model loaded successfully")

        logModelInformation()
    }

    /**
     * Bitmap 한 장에서 사람을 검출한다.
     */
    @Synchronized
    fun detect(bitmap: Bitmap): List<PoseDetection> {
        val preprocessResult =
            preprocessBitmap(bitmap)

        val inputBuffer =
            preprocessResult.inputBuffer

        inputBuffer.rewind()

        val inputShape =
            longArrayOf(
                1,
                3,
                INPUT_SIZE.toLong(),
                INPUT_SIZE.toLong(),
            )

        val startTime =
            SystemClock.elapsedRealtime()

        val detections =
            OnnxTensor.createTensor(
                ortEnvironment,
                inputBuffer,
                inputShape,
            ).use { inputTensor ->

                val inputMap =
                    mapOf(
                        inputName to inputTensor,
                    )

                ortSession.run(inputMap).use { result ->
                    val outputTensor =
                        result[0] as? OnnxTensor
                            ?: throw IllegalStateException(
                                "YOLO 출력이 OnnxTensor가 아닙니다.",
                            )

                    val outputBuffer =
                        outputTensor.floatBuffer
                            ?: throw IllegalStateException(
                                "YOLO Float 출력을 읽을 수 없습니다.",
                            )

                    outputBuffer.rewind()

                    val expectedOutputSize =
                        OUTPUT_CHANNELS * NUMBER_OF_CANDIDATES

                    if (outputBuffer.remaining() < expectedOutputSize) {
                        throw IllegalStateException(
                            "출력 데이터 크기가 예상보다 작습니다. " +
                                    "expected=$expectedOutputSize, " +
                                    "actual=${outputBuffer.remaining()}",
                        )
                    }

                    val outputArray =
                        FloatArray(expectedOutputSize)

                    outputBuffer.get(outputArray)

                    parseOutput(
                        output = outputArray,
                        preprocessResult = preprocessResult,
                    )
                }
            }

        val elapsedTime =
            SystemClock.elapsedRealtime() - startTime

        Log.d(
            TAG,
            "YOLO inference completed: " +
                    "${elapsedTime}ms, people=${detections.size}",
        )

        return detections
    }

    /**
     * 원본 Bitmap을 YOLO 입력인 640×640 RGB Float 형식으로 변환한다.
     */
    private fun preprocessBitmap(
        originalBitmap: Bitmap,
    ): PreprocessResult {

        val originalWidth =
            originalBitmap.width

        val originalHeight =
            originalBitmap.height

        // 원본 비율을 유지하면서 640×640 안에 들어가는 비율을 계산한다.
        val scale =
            min(
                INPUT_SIZE.toFloat() / originalWidth.toFloat(),
                INPUT_SIZE.toFloat() / originalHeight.toFloat(),
            )

        val resizedWidth =
            max(
                1,
                (originalWidth * scale).roundToInt(),
            )

        val resizedHeight =
            max(
                1,
                (originalHeight * scale).roundToInt(),
            )

        // 남는 공간을 좌우 또는 상하에 절반씩 배치한다.
        val paddingX =
            (INPUT_SIZE - resizedWidth) / 2f

        val paddingY =
            (INPUT_SIZE - resizedHeight) / 2f

        // YOLO에서 사용하는 회색 여백을 가진 640×640 Bitmap을 만든다.
        val letterboxBitmap =
            Bitmap.createBitmap(
                INPUT_SIZE,
                INPUT_SIZE,
                Bitmap.Config.ARGB_8888,
            )

        val canvas =
            Canvas(letterboxBitmap)

        canvas.drawColor(
            Color.rgb(
                114,
                114,
                114,
            ),
        )

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                        Paint.FILTER_BITMAP_FLAG,
            )

        val destinationRect =
            RectF(
                paddingX,
                paddingY,
                paddingX + resizedWidth,
                paddingY + resizedHeight,
            )

        canvas.drawBitmap(
            originalBitmap,
            null,
            destinationRect,
            paint,
        )

        val pixelCount =
            INPUT_SIZE * INPUT_SIZE

        val pixels =
            IntArray(pixelCount)

        letterboxBitmap.getPixels(
            pixels,
            0,
            INPUT_SIZE,
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE,
        )

        /*
         * ONNX 입력 메모리를 만든다.
         *
         * 입력 순서:
         * [1, 3, 640, 640]
         *
         * 채널 순서:
         * R 전체 → G 전체 → B 전체
         */
        val inputBuffer =
            ByteBuffer
                .allocateDirect(
                    3 * pixelCount * Float.SIZE_BYTES,
                )
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()

        // R 채널
        for (pixel in pixels) {
            val red =
                Color.red(pixel) / 255.0f

            inputBuffer.put(red)
        }

        // G 채널
        for (pixel in pixels) {
            val green =
                Color.green(pixel) / 255.0f

            inputBuffer.put(green)
        }

        // B 채널
        for (pixel in pixels) {
            val blue =
                Color.blue(pixel) / 255.0f

            inputBuffer.put(blue)
        }

        inputBuffer.rewind()

        letterboxBitmap.recycle()

        return PreprocessResult(
            inputBuffer = inputBuffer,
            originalWidth = originalWidth,
            originalHeight = originalHeight,
            scale = scale,
            paddingX = paddingX,
            paddingY = paddingY,
        )
    }

    /**
     * [1, 56, 8400] 출력을 사람 박스와 키포인트 목록으로 변환한다.
     */
    private fun parseOutput(
        output: FloatArray,
        preprocessResult: PreprocessResult,
    ): List<PoseDetection> {

        val candidates =
            mutableListOf<PoseDetection>()

        for (candidateIndex in 0 until NUMBER_OF_CANDIDATES) {
            val confidence =
                outputValue(
                    output = output,
                    channel = 4,
                    candidateIndex = candidateIndex,
                )

            if (confidence < CONFIDENCE_THRESHOLD) {
                continue
            }

            val centerX =
                outputValue(
                    output,
                    0,
                    candidateIndex,
                )

            val centerY =
                outputValue(
                    output,
                    1,
                    candidateIndex,
                )

            val width =
                outputValue(
                    output,
                    2,
                    candidateIndex,
                )

            val height =
                outputValue(
                    output,
                    3,
                    candidateIndex,
                )

            if (width <= 0f || height <= 0f) {
                continue
            }

            // 640×640 모델 좌표에서 박스 모서리를 계산한다.
            val modelLeft =
                centerX - width / 2f

            val modelTop =
                centerY - height / 2f

            val modelRight =
                centerX + width / 2f

            val modelBottom =
                centerY + height / 2f

            // Letterbox 여백을 제거하고 원본 Bitmap 좌표로 되돌린다.
            val originalLeft =
                mapXToOriginal(
                    modelLeft,
                    preprocessResult,
                )

            val originalTop =
                mapYToOriginal(
                    modelTop,
                    preprocessResult,
                )

            val originalRight =
                mapXToOriginal(
                    modelRight,
                    preprocessResult,
                )

            val originalBottom =
                mapYToOriginal(
                    modelBottom,
                    preprocessResult,
                )

            if (
                originalRight <= originalLeft ||
                originalBottom <= originalTop
            ) {
                continue
            }

            val keypoints =
                mutableListOf<PoseKeypoint>()

            /*
             * 채널 5부터 키포인트가 시작된다.
             *
             * 각 키포인트:
             * x, y, confidence
             *
             * 총 17개
             */
            for (keypointIndex in 0 until 17) {
                val startChannel =
                    5 + keypointIndex * 3

                val modelX =
                    outputValue(
                        output,
                        startChannel,
                        candidateIndex,
                    )

                val modelY =
                    outputValue(
                        output,
                        startChannel + 1,
                        candidateIndex,
                    )

                val keypointConfidence =
                    outputValue(
                        output,
                        startChannel + 2,
                        candidateIndex,
                    )

                keypoints.add(
                    PoseKeypoint(
                        x =
                            mapXToOriginal(
                                modelX,
                                preprocessResult,
                            ),
                        y =
                            mapYToOriginal(
                                modelY,
                                preprocessResult,
                            ),
                        confidence = keypointConfidence,
                    ),
                )
            }

            candidates.add(
                PoseDetection(
                    box =
                        RectF(
                            originalLeft,
                            originalTop,
                            originalRight,
                            originalBottom,
                        ),
                    confidence = confidence,
                    keypoints = keypoints,
                ),
            )
        }

        return applyNms(candidates)
    }

    /**
     * [1, 56, 8400] 배열에서 특정 값을 가져온다.
     */
    private fun outputValue(
        output: FloatArray,
        channel: Int,
        candidateIndex: Int,
    ): Float {
        val index =
            channel * NUMBER_OF_CANDIDATES +
                    candidateIndex

        return output[index]
    }

    private fun mapXToOriginal(
        modelX: Float,
        preprocessResult: PreprocessResult,
    ): Float {
        return (
                (modelX - preprocessResult.paddingX) /
                        preprocessResult.scale
                ).coerceIn(
                0f,
                preprocessResult.originalWidth.toFloat(),
            )
    }

    private fun mapYToOriginal(
        modelY: Float,
        preprocessResult: PreprocessResult,
    ): Float {
        return (
                (modelY - preprocessResult.paddingY) /
                        preprocessResult.scale
                ).coerceIn(
                0f,
                preprocessResult.originalHeight.toFloat(),
            )
    }

    /**
     * 같은 사람을 가리키는 중복 박스를 제거한다.
     */
    private fun applyNms(
        detections: List<PoseDetection>,
    ): List<PoseDetection> {

        val sortedDetections =
            detections.sortedByDescending {
                it.confidence
            }

        val selectedDetections =
            mutableListOf<PoseDetection>()

        for (candidate in sortedDetections) {
            val overlapsExistingDetection =
                selectedDetections.any { selected ->
                    calculateIoU(
                        candidate.box,
                        selected.box,
                    ) > NMS_THRESHOLD
                }

            if (!overlapsExistingDetection) {
                selectedDetections.add(candidate)
            }

            if (selectedDetections.size >= MAX_DETECTIONS) {
                break
            }
        }

        return selectedDetections
    }

    private fun calculateIoU(
        first: RectF,
        second: RectF,
    ): Float {

        val intersectionLeft =
            max(
                first.left,
                second.left,
            )

        val intersectionTop =
            max(
                first.top,
                second.top,
            )

        val intersectionRight =
            min(
                first.right,
                second.right,
            )

        val intersectionBottom =
            min(
                first.bottom,
                second.bottom,
            )

        val intersectionWidth =
            max(
                0f,
                intersectionRight - intersectionLeft,
            )

        val intersectionHeight =
            max(
                0f,
                intersectionBottom - intersectionTop,
            )

        val intersectionArea =
            intersectionWidth *
                    intersectionHeight

        val firstArea =
            max(
                0f,
                first.width(),
            ) *
                    max(
                        0f,
                        first.height(),
                    )

        val secondArea =
            max(
                0f,
                second.width(),
            ) *
                    max(
                        0f,
                        second.height(),
                    )

        val unionArea =
            firstArea +
                    secondArea -
                    intersectionArea

        if (unionArea <= 0f) {
            return 0f
        }

        return intersectionArea / unionArea
    }

    private fun logModelInformation() {
        ortSession.inputInfo.forEach { (name, nodeInfo) ->
            val tensorInfo =
                nodeInfo.info as? TensorInfo

            Log.d(
                TAG,
                "Input name=$name, " +
                        "shape=${tensorInfo?.shape?.contentToString()}, " +
                        "type=${tensorInfo?.type}",
            )
        }

        ortSession.outputInfo.forEach { (name, nodeInfo) ->
            val tensorInfo =
                nodeInfo.info as? TensorInfo

            Log.d(
                TAG,
                "Output name=$name, " +
                        "shape=${tensorInfo?.shape?.contentToString()}, " +
                        "type=${tensorInfo?.type}",
            )
        }
    }

    override fun close() {
        ortSession.close()

        Log.d(
            TAG,
            "YOLO model closed",
        )
    }

    private data class PreprocessResult(
        val inputBuffer: FloatBuffer,
        val originalWidth: Int,
        val originalHeight: Int,
        val scale: Float,
        val paddingX: Float,
        val paddingY: Float,
    )
}