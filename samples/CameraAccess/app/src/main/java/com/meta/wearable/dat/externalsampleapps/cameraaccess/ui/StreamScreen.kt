package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import androidx.activity.ComponentActivity
import androidx.activity.compose.LocalActivity
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.meta.wearable.dat.camera.types.StreamState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.R
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.StreamViewModel
import com.meta.wearable.dat.externalsampleapps.cameraaccess.wearables.WearablesViewModel
import kotlinx.coroutines.delay

@Composable
fun StreamScreen(
    wearablesViewModel: WearablesViewModel,
    modifier: Modifier = Modifier,
    streamViewModel: StreamViewModel =
        viewModel(
            factory =
                StreamViewModel.Factory(
                    application =
                        (LocalActivity.current as ComponentActivity)
                            .application,
                    wearablesViewModel = wearablesViewModel,
                ),
        ),
) {
    val streamUiState by
    streamViewModel.uiState.collectAsStateWithLifecycle()

    LaunchedEffect(Unit) {
        streamViewModel.startStream()
    }

    LaunchedEffect(
        streamUiState.shouldNavigateAfterUpload,
    ) {
        if (streamUiState.shouldNavigateAfterUpload) {
            // 성공 문구가 잠깐 보인 뒤 원래 기기 선택 화면으로 돌아간다.
            delay(700L)
            streamViewModel.consumeUploadNavigation()
            wearablesViewModel.navigateToDeviceSelection()
        }
    }

    val isStreamActive =
        streamUiState.streamState == StreamState.STARTING ||
                streamUiState.streamState == StreamState.STREAMING

    Box(
        modifier = modifier.fillMaxSize(),
    ) {
        streamUiState.videoFrame?.let { videoFrame ->
            key(streamUiState.videoFrameCount) {
                Image(
                    bitmap = videoFrame.asImageBitmap(),
                    contentDescription =
                        stringResource(R.string.live_stream),
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop,
                )
            }
        }

        if (
            streamUiState.streamState == StreamState.STARTING ||
            streamUiState.isUploading
        ) {
            CircularProgressIndicator(
                modifier = Modifier.align(Alignment.Center),
            )
        }

        streamUiState.uploadStatusText?.let { statusText ->
            Surface(
                modifier =
                    Modifier
                        .align(Alignment.TopCenter)
                        .padding(
                            start = 24.dp,
                            top = 48.dp,
                            end = 24.dp,
                        ),
                shape = MaterialTheme.shapes.medium,
                tonalElevation = 6.dp,
            ) {
                Text(
                    text = statusText,
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                )
            }
        }

        Column(
            modifier =
                Modifier
                    .align(Alignment.BottomCenter)
                    .navigationBarsPadding()
                    .fillMaxWidth()
                    .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier =
                    Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                SwitchButton(
                    label =
                        when {
                            streamUiState.isUploading ->
                                "사진 전송 중..."

                            isStreamActive ->
                                "스트리밍 중지 및 전송"

                            else ->
                                "저장된 사진 전송"
                        },
                    onClick = {
                        if (!streamUiState.isUploading) {
                            if (isStreamActive) {
                                streamViewModel.stopStreamAndUpload()
                            } else {
                                streamViewModel.uploadSavedPhotos()
                            }
                        }
                    },
                    isDestructive = isStreamActive,
                    modifier = Modifier.weight(1f),
                )

                if (
                    streamUiState.streamState ==
                    StreamState.STREAMING
                ) {
                    CaptureButton(
                        onClick = {
                            if (!streamUiState.isUploading) {
                                streamViewModel.capturePhoto()
                            }
                        },
                    )
                }
            }
        }
    }

    streamUiState.capturedPhoto?.let { photo ->
        if (streamUiState.isShareDialogVisible) {
            SharePhotoDialog(
                photo = photo,
                onDismiss = {
                    streamViewModel.hideShareDialog()
                },
                onShare = { bitmap ->
                    streamViewModel.sharePhoto(bitmap)
                    streamViewModel.hideShareDialog()
                },
            )
        }
    }
}