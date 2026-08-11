package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.graphics.Bitmap
import com.meta.wearable.dat.camera.types.StreamState

data class StreamUiState(
    val streamState: StreamState = StreamState.STOPPED,
    val videoFrame: Bitmap? = null,
    val videoFrameCount: Int = 0,
    val capturedPhoto: Bitmap? = null,
    val isShareDialogVisible: Boolean = false,
    val isCapturing: Boolean = false,
    val isUploading: Boolean = false,
    val uploadStatusText: String? = null,
    val pendingUploadCount: Int = 0,
    val shouldNavigateAfterUpload: Boolean = false,

    // 데이터 수집 세션 설정
    val placeName: String = "",
    val gpsLatitude: Double? = null,
    val gpsLongitude: Double? = null,
    val gpsAccuracyM: Float? = null,
    val isLocationLoading: Boolean = false,
    val locationStatusText: String? = null,
    val isCollectionSessionActive: Boolean = false,
    val batchId: String? = null,
)