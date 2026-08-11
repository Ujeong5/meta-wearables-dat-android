package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
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

    val context =
        LocalContext.current

    val locationPermissionLauncher =
        rememberLauncherForActivityResult(
            contract = ActivityResultContracts.RequestPermission(),
        ) { granted ->
            if (granted) {
                streamViewModel.fetchCurrentGps()
            } else {
                streamViewModel.onLocationPermissionDenied()
            }
        }

    LaunchedEffect(
        streamUiState.shouldNavigateAfterUpload,
    ) {
        if (streamUiState.shouldNavigateAfterUpload) {
            delay(700L)
            streamViewModel.consumeUploadNavigation()
            wearablesViewModel.navigateToDeviceSelection()
        }
    }

    val isStreamActive =
        streamUiState.streamState == StreamState.STARTING ||
                streamUiState.streamState == StreamState.STREAMING

    if (!streamUiState.isCollectionSessionActive) {
        CollectionSetupScreen(
            modifier = modifier,
            placeName = streamUiState.placeName,
            gpsLatitude = streamUiState.gpsLatitude,
            gpsLongitude = streamUiState.gpsLongitude,
            gpsAccuracyM = streamUiState.gpsAccuracyM,
            isLocationLoading = streamUiState.isLocationLoading,
            locationStatusText = streamUiState.locationStatusText,
            onPlaceNameChange = streamViewModel::updatePlaceName,
            onRequestGps = {
                val fineLocationGranted =
                    ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.ACCESS_FINE_LOCATION,
                    ) == PackageManager.PERMISSION_GRANTED

                if (fineLocationGranted) {
                    streamViewModel.fetchCurrentGps()
                } else {
                    locationPermissionLauncher.launch(
                        Manifest.permission.ACCESS_FINE_LOCATION,
                    )
                }
            },
            onStartCollection = {
                streamViewModel.startCollectionSession()
            },
        )
        return
    }

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

        Surface(
            modifier =
                Modifier
                    .align(Alignment.TopCenter)
                    .padding(
                        start = 16.dp,
                        top = 40.dp,
                        end = 16.dp,
                    ),
            shape = MaterialTheme.shapes.medium,
            tonalElevation = 6.dp,
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                Text(
                    text = streamUiState.placeName,
                    style = MaterialTheme.typography.titleSmall,
                )

                streamUiState.batchId?.let { batchId ->
                    Text(
                        text = "batch: $batchId",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }

                Text(
                    text = "저장 대기 ${streamUiState.pendingUploadCount}장",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        streamUiState.uploadStatusText?.let { statusText ->
            Surface(
                modifier =
                    Modifier
                        .align(Alignment.Center)
                        .padding(24.dp),
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
                                "수집 종료 및 전송"

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

@Composable
private fun CollectionSetupScreen(
    modifier: Modifier,
    placeName: String,
    gpsLatitude: Double?,
    gpsLongitude: Double?,
    gpsAccuracyM: Float?,
    isLocationLoading: Boolean,
    locationStatusText: String?,
    onPlaceNameChange: (String) -> Unit,
    onRequestGps: () -> Unit,
    onStartCollection: () -> Unit,
) {
    Surface(
        modifier = modifier.fillMaxSize(),
    ) {
        Column(
            modifier =
                Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(24.dp),
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                text = "데이터 수집 세션",
                style = MaterialTheme.typography.headlineSmall,
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text =
                    "장소와 휴대폰 GPS를 먼저 저장한 뒤 스트리밍을 시작합니다. " +
                            "장소명은 한글로 입력해도 됩니다.",
                style = MaterialTheme.typography.bodyMedium,
            )

            Spacer(modifier = Modifier.height(20.dp))

            OutlinedTextField(
                value = placeName,
                onValueChange = onPlaceNameChange,
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                label = {
                    Text("장소명")
                },
                placeholder = {
                    Text("예: 성수")
                },
            )

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = onRequestGps,
                enabled = !isLocationLoading,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    if (isLocationLoading) {
                        "GPS 위치 가져오는 중..."
                    } else {
                        "현재 GPS 가져오기"
                    },
                )
            }

            Spacer(modifier = Modifier.height(12.dp))

            if (
                gpsLatitude != null &&
                gpsLongitude != null &&
                gpsAccuracyM != null
            ) {
                Text(
                    text =
                        "위도: ${"%.6f".format(gpsLatitude)}\n" +
                                "경도: ${"%.6f".format(gpsLongitude)}\n" +
                                "정확도: ±${"%.1f".format(gpsAccuracyM)} m",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }

            locationStatusText?.let { text ->
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = text,
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            Button(
                onClick = onStartCollection,
                enabled =
                    placeName.isNotBlank() &&
                            gpsLatitude != null &&
                            gpsLongitude != null &&
                            gpsAccuracyM != null &&
                            !isLocationLoading,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("수집 시작")
            }
        }
    }
}
