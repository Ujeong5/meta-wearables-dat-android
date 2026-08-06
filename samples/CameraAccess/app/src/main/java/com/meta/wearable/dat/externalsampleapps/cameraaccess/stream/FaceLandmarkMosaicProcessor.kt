package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.io.Closeable
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/*
 * 사람 crop에서 모자·머리카락·헤어 액세서리는 보존하고,
 * 눈·코·입이 포함된 얼굴 중앙부만 픽셀 모자이크한다.
 *
 * 처리 순서:
 * 1. YOLO pose 얼굴/어깨 키포인트로 머리 ROI를 계산한다.
 * 2. 머리 ROI를 확대해 Face Landmarker를 먼저 실행한다.
 * 3. 실패하면 사람 crop 전체에서 한 번 더 실행한다.
 * 4. 그래도 실패했지만 YOLO에서 코와 눈이 보이면 pose fallback 모자이크를 적용한다.
 * 5. 얼굴 키포인트도 보이지 않으면 뒷모습으로 보고 아무 처리도 하지 않는다.
 */
class FaceLandmarkMosaicProcessor(
    context: Context,
) : Closeable {

    companion object {
        private const val TAG =
            "CameraAccess:FaceLandmarkMosaic"

        private const val MODEL_ASSET_PATH =
            "face_landmarker.task"

        /*
         * 사람 crop 하나에는 보통 얼굴이 하나이므로 1개만 찾는다.
         * 모자·측면·작은 얼굴에서 검출률을 높이기 위해 기본값보다 낮게 시작한다.
         */
        private const val MAX_NUM_FACES = 1
        private const val MIN_FACE_DETECTION_CONFIDENCE = 0.25f
        private const val MIN_FACE_PRESENCE_CONFIDENCE = 0.25f
        private const val MIN_TRACKING_CONFIDENCE = 0.25f

        private const val MIN_HEAD_ROI_SIDE_PX = 96
        private const val TARGET_HEAD_ROI_SHORT_SIDE_PX = 512
        private const val MAX_HEAD_ROI_SCALE = 4.0f

        private const val MIN_MOSAIC_WIDTH_PX = 10
        private const val MIN_MOSAIC_HEIGHT_PX = 10

        private const val POSE_FACE_KEYPOINT_CONFIDENCE = 0.20f
        private const val POSE_NOSE_CONFIDENCE = 0.22f

        /*
         * Face Landmarker 468-landmark 체계에서 눈·코·입 중심부에 해당하는 점들.
         * 이 점들의 전체 bounding box 하나를 만들어 얼굴 중앙부를 확실하게 가린다.
         */
        private val SENSITIVE_FACE_LANDMARK_INDICES =
            intArrayOf(
                // 왼쪽 눈
                33, 133, 157, 158, 159, 160, 173,
                144, 145, 153, 154, 155,

                // 오른쪽 눈
                263, 362, 384, 385, 386, 387, 398,
                373, 374, 380, 381, 382, 390,

                // 코
                1, 2, 4, 5, 6, 19, 94, 97, 98,
                168, 195, 197,

                // 입
                0, 13, 14, 17, 37, 39, 40, 61, 78,
                80, 81, 82, 84, 87, 88, 91, 95,
                146, 178, 181, 185, 191,
                267, 269, 270, 291, 308, 310, 311,
                312, 314, 317, 318, 321, 324, 375,
                402, 405, 409, 415,
            )

        /*
         * 눈 위쪽은 조금만 넓혀 이마·모자를 보존하고,
         * 좌우와 아래쪽은 넉넉히 넓혀 식별 가능한 얼굴 중앙부를 가린다.
         */
        private const val LANDMARK_HORIZONTAL_PADDING_RATIO = 0.20f
        private const val LANDMARK_TOP_PADDING_RATIO = 0.16f
        private const val LANDMARK_BOTTOM_PADDING_RATIO = 0.28f
    }

    data class MosaicResult(
        val detectedFaceCount: Int,
        val mosaickedFaceCount: Int,
        val mosaickedRegionCount: Int,
        val usedPoseFallback: Boolean,
        val strategy: String,
    )

    private data class DetectionAttempt(
        val landmarks: List<NormalizedLandmark>,
        val sourceRect: Rect,
        val strategy: String,
    )

    private val faceLandmarker: FaceLandmarker

    init {
        val baseOptions =
            BaseOptions.builder()
                .setDelegate(Delegate.CPU)
                .setModelAssetPath(MODEL_ASSET_PATH)
                .build()

        val options =
            FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.IMAGE)
                .setNumFaces(MAX_NUM_FACES)
                .setMinFaceDetectionConfidence(
                    MIN_FACE_DETECTION_CONFIDENCE,
                )
                .setMinFacePresenceConfidence(
                    MIN_FACE_PRESENCE_CONFIDENCE,
                )
                .setMinTrackingConfidence(
                    MIN_TRACKING_CONFIDENCE,
                )
                .setOutputFaceBlendshapes(false)
                .setOutputFacialTransformationMatrixes(false)
                .build()

        faceLandmarker =
            FaceLandmarker.createFromOptions(
                context.applicationContext,
                options,
            )

        Log.d(
            TAG,
            "Face Landmarker initialized: " +
                    "model=$MODEL_ASSET_PATH, " +
                    "detectionThreshold=$MIN_FACE_DETECTION_CONFIDENCE",
        )
    }

    /*
     * bitmap은 mutable ARGB_8888이어야 한다.
     * poseKeypoints는 사람 crop 좌표계의 YOLO11 pose 17개 키포인트다.
     */
    @Synchronized
    fun mosaicFaces(
        bitmap: Bitmap,
        poseKeypoints: List<PoseKeypoint> = emptyList(),
    ): MosaicResult {
        require(
            bitmap.config == Bitmap.Config.ARGB_8888,
        ) {
            "Face landmark mosaic requires ARGB_8888 Bitmap"
        }

        require(bitmap.isMutable) {
            "Face landmark mosaic requires a mutable Bitmap"
        }

        val headRoi =
            createHeadRoi(
                bitmapWidth = bitmap.width,
                bitmapHeight = bitmap.height,
                poseKeypoints = poseKeypoints,
            )

        val headAttempt =
            detectInRegion(
                bitmap = bitmap,
                sourceRect = headRoi,
                strategy = "HEAD_ROI",
                enlargeSmallRegion = true,
            )

        val fullAttempt =
            if (headAttempt == null) {
                detectInRegion(
                    bitmap = bitmap,
                    sourceRect =
                        Rect(
                            0,
                            0,
                            bitmap.width,
                            bitmap.height,
                        ),
                    strategy = "FULL_PERSON_CROP",
                    enlargeSmallRegion = false,
                )
            } else {
                null
            }

        val successfulAttempt =
            headAttempt ?: fullAttempt

        if (successfulAttempt != null) {
            val sensitiveRect =
                createSensitiveFaceRect(
                    landmarks = successfulAttempt.landmarks,
                    sourceRect = successfulAttempt.sourceRect,
                    bitmapWidth = bitmap.width,
                    bitmapHeight = bitmap.height,
                )

            if (sensitiveRect != null) {
                applyPixelMosaic(
                    bitmap = bitmap,
                    region = sensitiveRect,
                )

                Log.d(
                    TAG,
                    "FACE_MOSAIC_APPLIED: " +
                            "strategy=${successfulAttempt.strategy}, " +
                            "headRoi=$headRoi, " +
                            "region=$sensitiveRect",
                )

                return MosaicResult(
                    detectedFaceCount = 1,
                    mosaickedFaceCount = 1,
                    mosaickedRegionCount = 1,
                    usedPoseFallback = false,
                    strategy = successfulAttempt.strategy,
                )
            }
        }

        /*
         * 모자·측면 때문에 Face Landmarker가 실패해도,
         * YOLO에서 코와 최소 한쪽 눈이 보이면 얼굴 중앙부만 가린다.
         * 뒷모습은 코/눈 confidence가 낮기 때문에 이 fallback을 통과하지 않는다.
         */
        val poseFallbackRect =
            createPoseFallbackRect(
                bitmapWidth = bitmap.width,
                bitmapHeight = bitmap.height,
                poseKeypoints = poseKeypoints,
            )

        if (poseFallbackRect != null) {
            applyPixelMosaic(
                bitmap = bitmap,
                region = poseFallbackRect,
            )

            Log.w(
                TAG,
                "FACE_MOSAIC_POSE_FALLBACK: " +
                        "headRoi=$headRoi, region=$poseFallbackRect",
            )

            return MosaicResult(
                detectedFaceCount = 0,
                mosaickedFaceCount = 1,
                mosaickedRegionCount = 1,
                usedPoseFallback = true,
                strategy = "YOLO_POSE_FALLBACK",
            )
        }

        Log.d(
            TAG,
            "FACE_MOSAIC_SKIPPED: " +
                    "no visible face landmarks or pose face keypoints",
        )

        return MosaicResult(
            detectedFaceCount = 0,
            mosaickedFaceCount = 0,
            mosaickedRegionCount = 0,
            usedPoseFallback = false,
            strategy = "NONE",
        )
    }

    private fun detectInRegion(
        bitmap: Bitmap,
        sourceRect: Rect,
        strategy: String,
        enlargeSmallRegion: Boolean,
    ): DetectionAttempt? {
        if (
            sourceRect.width() < 2 ||
            sourceRect.height() < 2
        ) {
            return null
        }

        val regionBitmap =
            Bitmap.createBitmap(
                bitmap,
                sourceRect.left,
                sourceRect.top,
                sourceRect.width(),
                sourceRect.height(),
            )

        val inferenceBitmap =
            if (enlargeSmallRegion) {
                enlargeForFaceLandmarker(regionBitmap)
            } else {
                regionBitmap
            }

        try {
            val mpImage =
                BitmapImageBuilder(inferenceBitmap)
                    .build()

            val result =
                faceLandmarker.detect(mpImage)

            val landmarks =
                result.faceLandmarks()
                    .firstOrNull()
                    ?: return null

            return DetectionAttempt(
                landmarks = landmarks,
                sourceRect = sourceRect,
                strategy = strategy,
            )
        } finally {
            if (
                inferenceBitmap !== regionBitmap &&
                !inferenceBitmap.isRecycled
            ) {
                inferenceBitmap.recycle()
            }

            if (!regionBitmap.isRecycled) {
                regionBitmap.recycle()
            }
        }
    }

    private fun enlargeForFaceLandmarker(
        source: Bitmap,
    ): Bitmap {
        val shortSide =
            min(
                source.width,
                source.height,
            )

        if (shortSide <= 0) {
            return source
        }

        val requestedScale =
            TARGET_HEAD_ROI_SHORT_SIDE_PX.toFloat() /
                    shortSide.toFloat()

        val scale =
            requestedScale
                .coerceIn(
                    1.0f,
                    MAX_HEAD_ROI_SCALE,
                )

        if (scale <= 1.01f) {
            return source
        }

        return Bitmap.createScaledBitmap(
            source,
            max(
                1,
                (source.width * scale).roundToInt(),
            ),
            max(
                1,
                (source.height * scale).roundToInt(),
            ),
            true,
        )
    }

    private fun createHeadRoi(
        bitmapWidth: Int,
        bitmapHeight: Int,
        poseKeypoints: List<PoseKeypoint>,
    ): Rect {
        val facePoints =
            listOfNotNull(
                posePoint(poseKeypoints, 0), // nose
                posePoint(poseKeypoints, 1), // left eye
                posePoint(poseKeypoints, 2), // right eye
                posePoint(poseKeypoints, 3), // left ear
                posePoint(poseKeypoints, 4), // right ear
            )

        val shoulderPoints =
            listOfNotNull(
                posePoint(poseKeypoints, 5),
                posePoint(poseKeypoints, 6),
            )

        if (facePoints.isNotEmpty()) {
            val minX =
                facePoints.minOf { it.x }

            val maxX =
                facePoints.maxOf { it.x }

            val minY =
                facePoints.minOf { it.y }

            val maxY =
                facePoints.maxOf { it.y }

            val visibleFaceWidth =
                max(
                    maxX - minX,
                    bitmapWidth * 0.10f,
                )

            val shoulderY =
                if (shoulderPoints.isNotEmpty()) {
                    shoulderPoints
                        .map { it.y }
                        .average()
                        .toFloat()
                } else {
                    maxY + visibleFaceWidth * 1.5f
                }

            val centerX =
                facePoints
                    .map { it.x }
                    .average()
                    .toFloat()

            val roiWidth =
                max(
                    visibleFaceWidth * 3.2f,
                    bitmapWidth * 0.32f,
                )

            val estimatedHeight =
                max(
                    shoulderY - minY,
                    roiWidth * 0.85f,
                )

            val roiHeight =
                max(
                    estimatedHeight * 1.35f,
                    roiWidth * 1.05f,
                )

            val left =
                (centerX - roiWidth / 2f)
                    .roundToInt()
                    .coerceIn(
                        0,
                        bitmapWidth - 1,
                    )

            val top =
                (minY - roiHeight * 0.32f)
                    .roundToInt()
                    .coerceIn(
                        0,
                        bitmapHeight - 1,
                    )

            val right =
                (centerX + roiWidth / 2f)
                    .roundToInt()
                    .coerceIn(
                        left + 1,
                        bitmapWidth,
                    )

            val bottom =
                (top + roiHeight)
                    .roundToInt()
                    .coerceIn(
                        top + 1,
                        bitmapHeight,
                    )

            if (
                right - left >= MIN_HEAD_ROI_SIDE_PX &&
                bottom - top >= MIN_HEAD_ROI_SIDE_PX
            ) {
                return Rect(
                    left,
                    top,
                    right,
                    bottom,
                )
            }
        }

        /*
         * pose 얼굴 키포인트가 약하면 사람 crop 상단 48%를 머리/상체 ROI로 사용한다.
         */
        val fallbackBottom =
            max(
                1,
                (bitmapHeight * 0.48f).roundToInt(),
            )

        return Rect(
            0,
            0,
            bitmapWidth,
            fallbackBottom.coerceAtMost(bitmapHeight),
        )
    }

    private fun createSensitiveFaceRect(
        landmarks: List<NormalizedLandmark>,
        sourceRect: Rect,
        bitmapWidth: Int,
        bitmapHeight: Int,
    ): Rect? {
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE
        var validCount = 0

        SENSITIVE_FACE_LANDMARK_INDICES.forEach { index ->
            val landmark =
                landmarks.getOrNull(index)
                    ?: return@forEach

            val x = landmark.x()
            val y = landmark.y()

            if (!x.isFinite() || !y.isFinite()) {
                return@forEach
            }

            minX = min(minX, x)
            minY = min(minY, y)
            maxX = max(maxX, x)
            maxY = max(maxY, y)
            validCount += 1
        }

        if (validCount < 8) {
            return null
        }

        val normalizedWidth =
            maxX - minX

        val normalizedHeight =
            maxY - minY

        if (
            normalizedWidth <= 0f ||
            normalizedHeight <= 0f
        ) {
            return null
        }

        val expandedLeft =
            (
                    minX -
                            normalizedWidth *
                            LANDMARK_HORIZONTAL_PADDING_RATIO
                    ).coerceIn(0f, 1f)

        val expandedRight =
            (
                    maxX +
                            normalizedWidth *
                            LANDMARK_HORIZONTAL_PADDING_RATIO
                    ).coerceIn(0f, 1f)

        val expandedTop =
            (
                    minY -
                            normalizedHeight *
                            LANDMARK_TOP_PADDING_RATIO
                    ).coerceIn(0f, 1f)

        val expandedBottom =
            (
                    maxY +
                            normalizedHeight *
                            LANDMARK_BOTTOM_PADDING_RATIO
                    ).coerceIn(0f, 1f)

        val left =
            (
                    sourceRect.left +
                            expandedLeft * sourceRect.width()
                    )
                .roundToInt()
                .coerceIn(
                    0,
                    bitmapWidth - 1,
                )

        val top =
            (
                    sourceRect.top +
                            expandedTop * sourceRect.height()
                    )
                .roundToInt()
                .coerceIn(
                    0,
                    bitmapHeight - 1,
                )

        val right =
            (
                    sourceRect.left +
                            expandedRight * sourceRect.width()
                    )
                .roundToInt()
                .coerceIn(
                    left + 1,
                    bitmapWidth,
                )

        val bottom =
            (
                    sourceRect.top +
                            expandedBottom * sourceRect.height()
                    )
                .roundToInt()
                .coerceIn(
                    top + 1,
                    bitmapHeight,
                )

        if (
            right - left < MIN_MOSAIC_WIDTH_PX ||
            bottom - top < MIN_MOSAIC_HEIGHT_PX
        ) {
            return null
        }

        return Rect(
            left,
            top,
            right,
            bottom,
        )
    }

    private fun createPoseFallbackRect(
        bitmapWidth: Int,
        bitmapHeight: Int,
        poseKeypoints: List<PoseKeypoint>,
    ): Rect? {
        val nose =
            poseKeypoints.getOrNull(0)

        val leftEye =
            poseKeypoints.getOrNull(1)

        val rightEye =
            poseKeypoints.getOrNull(2)

        val leftEar =
            poseKeypoints.getOrNull(3)

        val rightEar =
            poseKeypoints.getOrNull(4)

        if (
            nose == null ||
            nose.confidence < POSE_NOSE_CONFIDENCE
        ) {
            return null
        }

        val visibleEyes =
            listOfNotNull(
                leftEye?.takeIf {
                    it.confidence >=
                            POSE_FACE_KEYPOINT_CONFIDENCE
                },
                rightEye?.takeIf {
                    it.confidence >=
                            POSE_FACE_KEYPOINT_CONFIDENCE
                },
            )

        if (visibleEyes.isEmpty()) {
            return null
        }

        val visibleFacePoints =
            buildList {
                add(nose)
                addAll(visibleEyes)

                leftEar?.takeIf {
                    it.confidence >=
                            POSE_FACE_KEYPOINT_CONFIDENCE
                }?.let(::add)

                rightEar?.takeIf {
                    it.confidence >=
                            POSE_FACE_KEYPOINT_CONFIDENCE
                }?.let(::add)
            }

        val minX =
            visibleFacePoints.minOf { it.x }

        val maxX =
            visibleFacePoints.maxOf { it.x }

        val minY =
            visibleFacePoints.minOf { it.y }

        val maxY =
            visibleFacePoints.maxOf { it.y }

        val rawWidth =
            max(
                maxX - minX,
                bitmapWidth * 0.08f,
            )

        val rawHeight =
            max(
                maxY - minY,
                rawWidth * 0.50f,
            )

        val centerX =
            visibleFacePoints
                .map { it.x }
                .average()
                .toFloat()

        val centerY =
            (
                    visibleEyes
                        .map { it.y }
                        .average()
                        .toFloat() +
                            nose.y
                    ) / 2f

        val regionWidth =
            rawWidth * 2.1f

        val regionHeight =
            max(
                rawHeight * 2.5f,
                regionWidth * 0.85f,
            )

        val left =
            (centerX - regionWidth / 2f)
                .roundToInt()
                .coerceIn(
                    0,
                    bitmapWidth - 1,
                )

        /*
         * top을 눈보다 조금 위에 두되 모자·이마 전체까지 올라가지 않도록 제한한다.
         */
        val top =
            (centerY - regionHeight * 0.34f)
                .roundToInt()
                .coerceIn(
                    0,
                    bitmapHeight - 1,
                )

        val right =
            (centerX + regionWidth / 2f)
                .roundToInt()
                .coerceIn(
                    left + 1,
                    bitmapWidth,
                )

        val bottom =
            (centerY + regionHeight * 0.66f)
                .roundToInt()
                .coerceIn(
                    top + 1,
                    bitmapHeight,
                )

        if (
            right - left < MIN_MOSAIC_WIDTH_PX ||
            bottom - top < MIN_MOSAIC_HEIGHT_PX
        ) {
            return null
        }

        return Rect(
            left,
            top,
            right,
            bottom,
        )
    }

    private fun posePoint(
        poseKeypoints: List<PoseKeypoint>,
        index: Int,
    ): PoseKeypoint? {
        return poseKeypoints
            .getOrNull(index)
            ?.takeIf {
                it.confidence >=
                        POSE_FACE_KEYPOINT_CONFIDENCE
            }
    }

    private fun applyPixelMosaic(
        bitmap: Bitmap,
        region: Rect,
    ) {
        val regionWidth =
            region.width()

        val regionHeight =
            region.height()

        if (
            regionWidth <= 1 ||
            regionHeight <= 1
        ) {
            return
        }

        val sourceRegion =
            Bitmap.createBitmap(
                bitmap,
                region.left,
                region.top,
                regionWidth,
                regionHeight,
            )

        /*
         * 얼굴 크기에 비례해 블록 수를 정한다.
         * 짧은 변 기준 약 8~12개 블록이 남도록 축소한다.
         */
        val dynamicBlockSize =
            max(
                10,
                min(
                    regionWidth,
                    regionHeight,
                ) / 10,
            )

        val reducedWidth =
            max(
                1,
                regionWidth /
                        dynamicBlockSize,
            )

        val reducedHeight =
            max(
                1,
                regionHeight /
                        dynamicBlockSize,
            )

        val reducedBitmap =
            Bitmap.createScaledBitmap(
                sourceRegion,
                reducedWidth,
                reducedHeight,
                false,
            )

        try {
            val canvas =
                Canvas(bitmap)

            val paint =
                Paint().apply {
                    isAntiAlias = false
                    isFilterBitmap = false
                    isDither = false
                }

            canvas.drawBitmap(
                reducedBitmap,
                null,
                region,
                paint,
            )
        } finally {
            if (
                reducedBitmap !== sourceRegion &&
                !reducedBitmap.isRecycled
            ) {
                reducedBitmap.recycle()
            }

            if (!sourceRegion.isRecycled) {
                sourceRegion.recycle()
            }
        }
    }

    override fun close() {
        faceLandmarker.close()

        Log.d(
            TAG,
            "Face Landmarker closed",
        )
    }
}