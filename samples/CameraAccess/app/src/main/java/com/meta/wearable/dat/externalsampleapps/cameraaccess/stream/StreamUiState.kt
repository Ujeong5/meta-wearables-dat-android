/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamUiState - DAT Camera Streaming UI State
//
// This data class manages UI state for camera streaming operations using the DAT API.

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

    // 현재 YOLO가 검출한 사람 목록
    val detections: List<PoseDetection> = emptyList(),

    // 검출에 사용된 원본 프레임 크기
    val detectionFrameWidth: Int = 0,
    val detectionFrameHeight: Int = 0,

    // 가장 최근 YOLO 전체 처리 시간
    val yoloElapsedMs: Long = 0L,
)
