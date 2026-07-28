package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import java.io.Closeable
import kotlin.math.max

/*
 * 사람 crop 내부의 얼굴을 검출하고 픽셀 모자이크를 적용한다.
 *
 * 이 클래스는 MediaPipe Face Detector를 IMAGE 모드로 사용한다.
 * 사람 crop은 전신에 가까운 이미지이므로 short-range보다
 * full-range BlazeFace 모델을 사용한다.
 */
class FaceMosaicProcessor(
    context: Context,
) : Closeable {

    companion object {
        private const val TAG =
            "CameraAccess:FaceMosaicProcessor"

        /*
         * app/src/main/assets 아래에 이 파일이 있어야 한다.
         */
        private const val MODEL_ASSET_PATH =
            "blaze_face_full_range.tflite"

        private const val MIN_DETECTION_CONFIDENCE =
            0.45f

        private const val MIN_SUPPRESSION_THRESHOLD =
            0.30f

        /*
         * 얼굴 바깥쪽까지 넉넉하게 가리기 위한 여백이다.
         */
        private const val FACE_HORIZONTAL_PADDING_RATIO =
            0.22f

        private const val FACE_TOP_PADDING_RATIO =
            0.32f

        private const val FACE_BOTTOM_PADDING_RATIO =
            0.22f

        /*
         * 모자이크 블록의 대략적인 크기다.
         * 값이 클수록 더 강하게 가려진다.
         */
        private const val MOSAIC_BLOCK_SIZE_PX =
            18

        /*
         * 얼굴 검출이 실패했을 때 crop 상단의 머리 영역을 가린다.
         * 검출 실패로 원본 얼굴이 저장되는 것을 막기 위한 fallback이다.
         */
        private const val FALLBACK_LEFT_RATIO =
            0.12f

        private const val FALLBACK_RIGHT_RATIO =
            0.88f

        private const val FALLBACK_BOTTOM_RATIO =
            0.32f
    }

    data class MosaicResult(
        val detectedFaceCount: Int,
        val mosaickedRegionCount: Int,
        val usedFallback: Boolean,
    )

    private val faceDetector: FaceDetector

    init {
        val baseOptions =
            BaseOptions.builder()
                .setDelegate(Delegate.CPU)
                .setModelAssetPath(MODEL_ASSET_PATH)
                .build()

        val options =
            FaceDetector.FaceDetectorOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinDetectionConfidence(
                    MIN_DETECTION_CONFIDENCE,
                )
                .setMinSuppressionThreshold(
                    MIN_SUPPRESSION_THRESHOLD,
                )
                .setRunningMode(RunningMode.IMAGE)
                .build()

        faceDetector =
            FaceDetector.createFromOptions(
                context.applicationContext,
                options,
            )

        Log.d(
            TAG,
            "Face detector initialized with $MODEL_ASSET_PATH",
        )
    }

    /*
     * FaceDetector 인스턴스를 동시에 여러 스레드에서 사용하지 않도록
     * synchronized로 직렬화한다.
     *
     * bitmap은 반드시 mutable ARGB_8888 Bitmap이어야 한다.
     * 모자이크는 전달받은 Bitmap 자체에 적용된다.
     */
    @Synchronized
    fun mosaicFaces(
        bitmap: Bitmap,
    ): MosaicResult {
        require(
            bitmap.config == Bitmap.Config.ARGB_8888,
        ) {
            "Face mosaic requires ARGB_8888 Bitmap"
        }

        require(bitmap.isMutable) {
            "Face mosaic requires a mutable Bitmap"
        }

        val mpImage =
            BitmapImageBuilder(bitmap).build()

        val result =
            faceDetector.detect(mpImage)

        val faceRects =
            result
                .detections()
                .mapNotNull { detection ->
                    expandAndClampFaceRect(
                        faceRect = detection.boundingBox(),
                        bitmapWidth = bitmap.width,
                        bitmapHeight = bitmap.height,
                    )
                }

        if (faceRects.isNotEmpty()) {
            faceRects.forEach { rect ->
                applyPixelMosaic(
                    bitmap = bitmap,
                    region = rect,
                )
            }

            return MosaicResult(
                detectedFaceCount = faceRects.size,
                mosaickedRegionCount = faceRects.size,
                usedFallback = false,
            )
        }

        /*
         * BlazeFace가 얼굴을 놓친 경우에도
         * 원본 얼굴이 그대로 저장되지 않도록 crop 상단을 가린다.
         */
        val fallbackRect =
            createFallbackHeadRect(
                bitmapWidth = bitmap.width,
                bitmapHeight = bitmap.height,
            )

        applyPixelMosaic(
            bitmap = bitmap,
            region = fallbackRect,
        )

        Log.w(
            TAG,
            "No face detected; applied fallback head mosaic: $fallbackRect",
        )

        return MosaicResult(
            detectedFaceCount = 0,
            mosaickedRegionCount = 1,
            usedFallback = true,
        )
    }

    private fun expandAndClampFaceRect(
        faceRect: RectF,
        bitmapWidth: Int,
        bitmapHeight: Int,
    ): Rect? {
        val faceWidth =
            faceRect.width()

        val faceHeight =
            faceRect.height()

        if (
            faceWidth <= 1f ||
            faceHeight <= 1f
        ) {
            return null
        }

        val left =
            (
                    faceRect.left -
                            faceWidth *
                            FACE_HORIZONTAL_PADDING_RATIO
                    )
                .toInt()
                .coerceIn(
                    0,
                    bitmapWidth - 1,
                )

        val top =
            (
                    faceRect.top -
                            faceHeight *
                            FACE_TOP_PADDING_RATIO
                    )
                .toInt()
                .coerceIn(
                    0,
                    bitmapHeight - 1,
                )

        val right =
            (
                    faceRect.right +
                            faceWidth *
                            FACE_HORIZONTAL_PADDING_RATIO
                    )
                .toInt()
                .coerceIn(
                    left + 1,
                    bitmapWidth,
                )

        val bottom =
            (
                    faceRect.bottom +
                            faceHeight *
                            FACE_BOTTOM_PADDING_RATIO
                    )
                .toInt()
                .coerceIn(
                    top + 1,
                    bitmapHeight,
                )

        return Rect(
            left,
            top,
            right,
            bottom,
        )
    }

    private fun createFallbackHeadRect(
        bitmapWidth: Int,
        bitmapHeight: Int,
    ): Rect {
        val left =
            (
                    bitmapWidth *
                            FALLBACK_LEFT_RATIO
                    )
                .toInt()
                .coerceIn(
                    0,
                    bitmapWidth - 1,
                )

        val top =
            0

        val right =
            (
                    bitmapWidth *
                            FALLBACK_RIGHT_RATIO
                    )
                .toInt()
                .coerceIn(
                    left + 1,
                    bitmapWidth,
                )

        val bottom =
            (
                    bitmapHeight *
                            FALLBACK_BOTTOM_RATIO
                    )
                .toInt()
                .coerceIn(
                    1,
                    bitmapHeight,
                )

        return Rect(
            left,
            top,
            right,
            bottom,
        )
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

        val reducedWidth =
            max(
                1,
                regionWidth /
                        MOSAIC_BLOCK_SIZE_PX,
            )

        val reducedHeight =
            max(
                1,
                regionHeight /
                        MOSAIC_BLOCK_SIZE_PX,
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
        faceDetector.close()

        Log.d(
            TAG,
            "Face detector closed",
        )
    }
}