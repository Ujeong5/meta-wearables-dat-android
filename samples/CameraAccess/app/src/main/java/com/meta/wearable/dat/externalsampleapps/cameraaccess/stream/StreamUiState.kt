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
)