/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamViewModel - DAT Camera Streaming API Demo
//
// This ViewModel demonstrates the DAT Camera Streaming APIs for:
// - Creating and managing stream sessions with wearable devices
// - Receiving video frames from device cameras
// - Capturing photos during streaming sessions
// - Handling different video qualities and formats
// - Processing raw video data (I420 -> ARGB conversion)

package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.annotation.SuppressLint
import android.app.Application
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.util.Log
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.meta.wearable.dat.camera.Stream
import com.meta.wearable.dat.camera.addStream
import com.meta.wearable.dat.camera.types.PhotoData
import com.meta.wearable.dat.camera.types.StreamConfiguration
import com.meta.wearable.dat.camera.types.StreamError
import com.meta.wearable.dat.camera.types.StreamState
import com.meta.wearable.dat.camera.types.VideoFrame
import com.meta.wearable.dat.camera.types.VideoQuality
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.DeviceSelector
import com.meta.wearable.dat.core.session.DeviceSession
import com.meta.wearable.dat.core.session.DeviceSessionState
import com.meta.wearable.dat.core.types.DeviceSessionError
import com.meta.wearable.dat.externalsampleapps.cameraaccess.wearables.WearablesViewModel
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import android.os.Environment
import kotlinx.coroutines.Dispatchers
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import kotlinx.coroutines.withContext
import kotlin.math.max
import kotlin.math.roundToInt
import android.os.SystemClock
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.coroutines.CancellationException


@SuppressLint("AutoCloseableUse")
class StreamViewModel(
    application: Application,
    private val wearablesViewModel: WearablesViewModel,
) : AndroidViewModel(application) {

  companion object {
    private const val TAG = "CameraAccess:StreamViewModel"
    private val INITIAL_STATE = StreamUiState()
    private val SESSION_TERMINAL_STATES = setOf(StreamState.CLOSED)

    // YOLO는 500ms마다 한 번, 즉 초당 약 2회 실행한다.
    private const val YOLO_INTERVAL_MS = 500L
  }

  private val yoloPoseModel =
    YoloPoseModel(getApplication())
  // 현재 YOLO 추론이 실행 중인지 나타낸다.
  private val isYoloRunning =
    AtomicBoolean(false)

  // 마지막으로 YOLO를 시작한 시간이다.
  private var lastYoloStartedAtMs =
    0L

  // 현재 실행 중인 YOLO 코루틴이다.
  private var yoloJob: Job? =
    null

  // 가장 최근 프레임의 검출 결과다.
  @Volatile
  private var latestDetections: List<PoseDetection> =
    emptyList()

  // 검출에 사용된 이미지 크기다.
  @Volatile
  private var latestDetectionWidth =
    0

  @Volatile
  private var latestDetectionHeight =
    0
  private var hasSavedDebugFrame = false
  private val deviceSelector: DeviceSelector = wearablesViewModel.deviceSelector
  private var session: DeviceSession? = null

  private val _uiState = MutableStateFlow(INITIAL_STATE)
  val uiState: StateFlow<StreamUiState> = _uiState.asStateFlow()

  private var videoJob: Job? = null
  private var stateJob: Job? = null
  private var errorJob: Job? = null
  private var sessionErrorJob: Job? = null
  private var sessionStateJob: Job? = null
  private var stream: Stream? = null
  private var previousDeviceSessionState: DeviceSessionState? = null

  // Presentation queue for buffering frames after color conversion
  private var presentationQueue: PresentationQueue? = null

  fun startStream() {
    videoJob?.cancel()
    stateJob?.cancel()
    errorJob?.cancel()
    sessionErrorJob?.cancel()
    sessionStateJob?.cancel()
    presentationQueue?.stop()
    presentationQueue = null
    previousDeviceSessionState = null

    // Initialize presentation queue - frames are presented based on timestamp, not arrival time
    // Uses IntArray pooling for efficiency - cheaper than Bitmap.copy()
    val queue =
        PresentationQueue(
            bufferDelayMs = 100L,
            maxQueueSize = 15,
            onFrameReady = { frame ->
              // This is called from the presentation thread at regular intervals
              // when a frame's presentation time has arrived
              viewModelScope.launch(kotlinx.coroutines.Dispatchers.Main) {
                _uiState.update {
                  it.copy(videoFrame = frame.bitmap, videoFrameCount = it.videoFrameCount + 1)
                }
              }
            },
        )
    presentationQueue = queue
    queue.start()
    if (session == null) {
      previousDeviceSessionState = null
      Wearables.createSession(deviceSelector)
          .onSuccess { createdSession ->
            session = createdSession
            sessionErrorJob = viewModelScope.launch {
              createdSession.errors.collect { error -> handleSessionError(error) }
            }
            session?.start()
          }
          .onFailure { error, _ ->
            Log.e(TAG, "Failed to create session: ${error.description}")
            handleSessionError(error)
          }
      if (session == null) return
    }
    startStreamInternal()
  }

  private fun startStreamInternal() {
    Log.d(TAG, "startStreamInternal() - collecting session state")
    sessionStateJob = viewModelScope.launch {
      session?.state?.collect { currentState ->
        val prevState = previousDeviceSessionState
        previousDeviceSessionState = currentState

        if (currentState == DeviceSessionState.STARTED) {
          wearablesViewModel.setDatAppUpdateRequired(false)
          if (prevState == DeviceSessionState.PAUSED && stream != null) {
            // PAUSED → STARTED: device-initiated resume (tap gesture).
            // The SDK handles resume internally via requestCameraOn() → resumeStreaming().
            // Do NOT recreate the stream — just let the SDK resume it.
            Log.d(TAG, "Session resumed from PAUSED — stream stays alive")
            return@collect
          }

          videoJob?.cancel()
          stateJob?.cancel()
          errorJob?.cancel()
          stream?.stop()
          stream = null
          session
              ?.addStream(StreamConfiguration(videoQuality = VideoQuality.HIGH, frameRate = 2))
              ?.onSuccess { addedStream ->
                stream = addedStream
                videoJob = viewModelScope.launch {
                  Log.d(TAG, "Collecting video frames from stream")
                  stream?.videoStream?.collect { handleVideoFrame(it) }
                  Log.d(TAG, "Video stream collection ended")
                }
                stateJob = viewModelScope.launch {
                  stream?.state?.collect { streamState ->
                    val prevStreamState = _uiState.value.streamState
                    Log.d(TAG, "Stream state changed: $prevStreamState -> $streamState")
                    _uiState.update { it.copy(streamState = streamState) }

                    val wasActive = prevStreamState !in SESSION_TERMINAL_STATES
                    val isTerminated = streamState in SESSION_TERMINAL_STATES
                    if (wasActive && isTerminated) {
                      Log.d(TAG, "Terminal state reached, navigating back")
                      stopStream()
                      wearablesViewModel.navigateToDeviceSelection()
                    }
                  }
                }
                errorJob = viewModelScope.launch {
                  stream?.errorStream?.collect { error ->
                    Log.d(TAG, "Stream error received: $error (description: ${error.description})")
                    if (error == StreamError.STREAM_ERROR) {
                      Log.d(TAG, "Non-critical error, stream continues")
                      return@collect
                    }
                    stopStream()
                    wearablesViewModel.navigateToDeviceSelection()
                    // Use `getLocalizedDescription(context)` for user-facing text —
                    // `description` is always English and intended for logs.
                    wearablesViewModel.setRecentError(
                        error.getLocalizedDescription(getApplication())
                    )
                  }
                }
                stream?.start()
              }
              ?.onFailure { error, _ ->
                Log.e(TAG, "Failed to add stream to session: ${error.description}")
              }
        } else if (currentState == DeviceSessionState.PAUSED) {
          // Tap gesture paused the session — keep the stream alive.
          // The SDK transitions StreamState to PAUSED internally.
          Log.d(TAG, "Session paused (tap gesture) — keeping stream alive for resume")
        }
      }
    }
  }

  fun stopStream() {
    videoJob?.cancel()
    videoJob = null
    yoloJob?.cancel()
    yoloJob = null

    latestDetections =
      emptyList()

    latestDetectionWidth =
      0

    latestDetectionHeight =
      0

    lastYoloStartedAtMs =
      0L
    stateJob?.cancel()
    stateJob = null
    errorJob?.cancel()
    errorJob = null
    sessionErrorJob?.cancel()
    sessionErrorJob = null
    sessionStateJob?.cancel()
    sessionStateJob = null
    presentationQueue?.stop()
    presentationQueue = null
    _uiState.update { INITIAL_STATE }
    stream?.stop()
    stream = null
    session?.stop()
    session = null

  }

  private fun handleSessionError(error: DeviceSessionError) {
    Log.e(TAG, "Session error: ${error.description}")
    val alreadyShowingUpdateRequired =
        wearablesViewModel.uiState.value.isFirmwareUpdateRequired ||
            wearablesViewModel.uiState.value.isDatAppUpdateRequired

    if (error == DeviceSessionError.DAT_APP_ON_THE_GLASSES_UPDATE_REQUIRED) {
      wearablesViewModel.setDatAppUpdateRequired(true)
    }
    if (alreadyShowingUpdateRequired && error == DeviceSessionError.SESSION_ENDED_BY_DEVICE) {
      stopStream()
      wearablesViewModel.navigateToDeviceSelection()
      return
    }

    wearablesViewModel.setRecentError(error.getLocalizedDescription(getApplication()))
    stopStream()
    wearablesViewModel.navigateToDeviceSelection()
  }

  fun capturePhoto() {
    if (uiState.value.isCapturing) {
      Log.d(TAG, "Photo capture already in progress, ignoring request")
      return
    }

    if (uiState.value.streamState == StreamState.STREAMING) {
      Log.d(TAG, "Starting photo capture")
      _uiState.update { it.copy(isCapturing = true) }

      viewModelScope.launch {
        stream
            ?.capturePhoto()
            ?.onSuccess { photoData ->
              Log.d(TAG, "Photo capture successful")
              handlePhotoData(photoData)
              _uiState.update { it.copy(isCapturing = false) }
            }
            ?.onFailure { error, _ ->
              Log.e(TAG, "Photo capture failed: ${error.description}")
              _uiState.update { it.copy(isCapturing = false) }
            }
      }
    } else {
      Log.w(
          TAG,
          "Cannot capture photo: stream not active (state=${uiState.value.streamState})",
      )
    }
  }

  fun showShareDialog() {
    _uiState.update { it.copy(isShareDialogVisible = true) }
  }

  fun hideShareDialog() {
    _uiState.update { it.copy(isShareDialogVisible = false) }
  }

  fun sharePhoto(bitmap: Bitmap) {
    val context = getApplication<Application>()
    val imagesFolder = File(context.cacheDir, "images")
    try {
      imagesFolder.mkdirs()
      val file = File(imagesFolder, "shared_image.png")
      FileOutputStream(file).use { stream ->
        bitmap.compress(Bitmap.CompressFormat.PNG, 90, stream)
      }

      val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
      val intent = Intent(Intent.ACTION_SEND)
      intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
      intent.putExtra(Intent.EXTRA_STREAM, uri)
      intent.type = "image/png"
      intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

      val chooser = Intent.createChooser(intent, "Share Image")
      chooser.flags = Intent.FLAG_ACTIVITY_NEW_TASK
      context.startActivity(chooser)
    } catch (e: IOException) {
      Log.e("StreamViewModel", "Failed to share photo", e)
    }
  }

  private fun handleVideoFrame(
    videoFrame: VideoFrame,
  ) {
    val bitmap =
      YuvToBitmapConverter.convert(
        videoFrame.buffer,
        videoFrame.width,
        videoFrame.height,
      )

    if (bitmap == null) {
      Log.e(
        TAG,
        "Failed to convert YUV to bitmap",
      )
      return
    }

    /*
     * 반드시 박스를 그리기 전에 YOLO용 프레임을 복사해야 한다.
     * 그렇지 않으면 YOLO가 초록색 박스까지 포함된 이미지를 분석한다.
     */
    startRealtimeYoloIfNeeded(bitmap)

    /*
     * 가장 최근 YOLO 결과를 현재 프레임에 표시한다.
     * 검출 결과는 약 0.5초마다 갱신된다.
     */
    drawLatestDetectionsInPlace(bitmap)

    /*
     * PresentationQueue가 Bitmap을 복제한 후
     * 스마트폰 미리보기에 표시한다.
     */
    presentationQueue?.enqueue(
      bitmap,
      videoFrame.presentationTimeUs,
    )
  }
  private fun saveAnnotatedYoloFrame(
    bitmap: Bitmap,
  ) {
    try {
      val context =
        getApplication<Application>()

      val baseDirectory =
        context.getExternalFilesDir(
          Environment.DIRECTORY_PICTURES,
        ) ?: context.filesDir

      val resultDirectory =
        File(
          baseDirectory,
          "yolo_results",
        ).apply {
          mkdirs()
        }

      val outputFile =
        File(
          resultDirectory,
          "yolo_result_${System.currentTimeMillis()}.jpg",
        )

      FileOutputStream(outputFile).use { outputStream ->
        val success =
          bitmap.compress(
            Bitmap.CompressFormat.JPEG,
            95,
            outputStream,
          )

        if (!success) {
          throw IllegalStateException(
            "YOLO 결과 이미지 압축에 실패했습니다.",
          )
        }
      }

      Log.d(
        TAG,
        "Annotated YOLO frame saved: " +
                "${outputFile.absolutePath}, " +
                "size=${bitmap.width}x${bitmap.height}",
      )
    } catch (exception: Exception) {
      Log.e(
        TAG,
        "Annotated YOLO frame save failed",
        exception,
      )
    }
  }
  private fun startRealtimeYoloIfNeeded(
    rawBitmap: Bitmap,
  ) {
    val currentTime =
      SystemClock.elapsedRealtime()

    // 이전 추론을 시작한 후 500ms가 지나지 않았다면 건너뛴다.
    if (
      currentTime - lastYoloStartedAtMs <
      YOLO_INTERVAL_MS
    ) {
      return
    }

    // 이전 YOLO 추론이 아직 진행 중이면 현재 프레임을 버린다.
    if (!isYoloRunning.compareAndSet(false, true)) {
      return
    }

    /*
     * YuvToBitmapConverter가 만든 Bitmap은 다음 프레임에서
     * 다시 사용되므로, 비동기 추론용 복사본이 반드시 필요하다.
     */
    val inferenceBitmap =
      rawBitmap.copy(
        Bitmap.Config.ARGB_8888,
        false,
      )

    if (inferenceBitmap == null) {
      Log.e(
        TAG,
        "실시간 YOLO용 Bitmap 복사에 실패했습니다.",
      )

      isYoloRunning.set(false)
      return
    }

    lastYoloStartedAtMs =
      currentTime

    yoloJob =
      viewModelScope.launch(Dispatchers.Default) {
        val totalStartTime =
          SystemClock.elapsedRealtime()

        try {
          val detections =
            yoloPoseModel.detect(inferenceBitmap)

          // 완성된 결과를 한 번에 교체한다.
          latestDetections =
            detections

          latestDetectionWidth =
            inferenceBitmap.width

          latestDetectionHeight =
            inferenceBitmap.height

          val totalElapsedTime =
            SystemClock.elapsedRealtime() -
                    totalStartTime

          Log.d(
            TAG,
            "Realtime YOLO: " +
                    "total=${totalElapsedTime}ms, " +
                    "people=${detections.size}",
          )
        } catch (exception: CancellationException) {
          throw exception
        } catch (exception: Exception) {
          Log.e(
            TAG,
            "Realtime YOLO failed",
            exception,
          )
        } finally {
          inferenceBitmap.recycle()
          isYoloRunning.set(false)
        }
      }
  }
  private fun drawLatestDetectionsInPlace(
    bitmap: Bitmap,
  ) {
    val detections =
      latestDetections

    if (detections.isEmpty()) {
      return
    }

    /*
     * 검출에 사용한 프레임과 현재 프레임의 해상도가 다르면
     * 좌표가 맞지 않으므로 이번 프레임에는 그리지 않는다.
     */
    if (
      latestDetectionWidth != bitmap.width ||
      latestDetectionHeight != bitmap.height
    ) {
      return
    }

    val canvas =
      Canvas(bitmap)

    val strokeWidth =
      max(
        3f,
        bitmap.width / 180f,
      )

    val boxPaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        this.strokeWidth = strokeWidth
      }

    val skeletonPaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        this.strokeWidth = strokeWidth
      }

    val keypointPaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.FILL
      }

    val textPaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize =
          max(
            28f,
            bitmap.width / 18f,
          )
        style = Paint.Style.FILL
      }

    val textBackgroundPaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        style = Paint.Style.FILL
      }

    val skeletonConnections =
      arrayOf(
        intArrayOf(5, 6),   // 양쪽 어깨
        intArrayOf(5, 7),   // 왼쪽 어깨-팔꿈치
        intArrayOf(7, 9),   // 왼쪽 팔꿈치-손목
        intArrayOf(6, 8),   // 오른쪽 어깨-팔꿈치
        intArrayOf(8, 10),  // 오른쪽 팔꿈치-손목
        intArrayOf(5, 11),  // 왼쪽 어깨-골반
        intArrayOf(6, 12),  // 오른쪽 어깨-골반
        intArrayOf(11, 12), // 양쪽 골반
        intArrayOf(11, 13), // 왼쪽 골반-무릎
        intArrayOf(13, 15), // 왼쪽 무릎-발목
        intArrayOf(12, 14), // 오른쪽 골반-무릎
        intArrayOf(14, 16), // 오른쪽 무릎-발목
      )

    detections.forEach { detection ->
      // 사람 Bounding Box
      canvas.drawRect(
        detection.box,
        boxPaint,
      )

      // 신뢰도 문구
      val confidencePercent =
        (detection.confidence * 100f)
          .roundToInt()

      val label =
        "person $confidencePercent%"

      val labelWidth =
        textPaint.measureText(label)

      val labelBottom =
        max(
          textPaint.textSize,
          detection.box.top,
        )

      canvas.drawRect(
        detection.box.left,
        labelBottom - textPaint.textSize,
        detection.box.left + labelWidth + 12f,
        labelBottom + 8f,
        textBackgroundPaint,
      )

      canvas.drawText(
        label,
        detection.box.left + 6f,
        labelBottom,
        textPaint,
      )

      // 신체 관절 연결선
      skeletonConnections.forEach { connection ->
        val first =
          detection.keypoints[connection[0]]

        val second =
          detection.keypoints[connection[1]]

        if (
          first.confidence >= 0.25f &&
          second.confidence >= 0.25f
        ) {
          canvas.drawLine(
            first.x,
            first.y,
            second.x,
            second.y,
            skeletonPaint,
          )
        }
      }

      // 신체 관절점
      detection.keypoints.forEach { keypoint ->
        if (keypoint.confidence >= 0.25f) {
          canvas.drawCircle(
            keypoint.x,
            keypoint.y,
            strokeWidth * 1.5f,
            keypointPaint,
          )
        }
      }
    }
  }
  private fun drawPoseDetections(
    sourceBitmap: Bitmap,
    detections: List<PoseDetection>,
  ): Bitmap {
    // 그림을 그릴 수 있는 mutable Bitmap을 만든다.
    val resultBitmap =
      sourceBitmap.copy(
        Bitmap.Config.ARGB_8888,
        true,
      ) ?: throw IllegalStateException(
        "결과 이미지용 Bitmap 생성에 실패했습니다.",
      )

    val canvas =
      Canvas(resultBitmap)

    // 이미지 해상도에 따라 선 굵기를 조절한다.
    val baseStrokeWidth =
      max(
        3f,
        resultBitmap.width / 180f,
      )

    val boxPaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = baseStrokeWidth
      }

    val skeletonPaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = baseStrokeWidth
      }

    val keypointPaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.FILL
      }

    val textPaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize =
          max(
            28f,
            resultBitmap.width / 18f,
          )
        style = Paint.Style.FILL
      }

    val textBackgroundPaint =
      Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.BLACK
        style = Paint.Style.FILL
      }

    /*
     * COCO Pose 신체 연결 관계다.
     *
     * 5: 왼쪽 어깨
     * 6: 오른쪽 어깨
     * 7, 8: 팔꿈치
     * 9, 10: 손목
     * 11, 12: 골반
     * 13, 14: 무릎
     * 15, 16: 발목
     */
    val skeletonConnections =
      arrayOf(
        intArrayOf(5, 6),
        intArrayOf(5, 7),
        intArrayOf(7, 9),
        intArrayOf(6, 8),
        intArrayOf(8, 10),
        intArrayOf(5, 11),
        intArrayOf(6, 12),
        intArrayOf(11, 12),
        intArrayOf(11, 13),
        intArrayOf(13, 15),
        intArrayOf(12, 14),
        intArrayOf(14, 16),
      )

    detections.forEach { detection ->
      // 사람 Bounding Box를 그린다.
      canvas.drawRect(
        detection.box,
        boxPaint,
      )

      // 검출 신뢰도를 표시한다.
      val confidencePercent =
        (detection.confidence * 100f)
          .roundToInt()

      val label =
        "person $confidencePercent%"

      val textWidth =
        textPaint.measureText(label)

      val textHeight =
        textPaint.textSize

      val labelLeft =
        detection.box.left

      val labelBottom =
        max(
          textHeight,
          detection.box.top,
        )

      canvas.drawRect(
        labelLeft,
        labelBottom - textHeight,
        labelLeft + textWidth + 12f,
        labelBottom + 8f,
        textBackgroundPaint,
      )

      canvas.drawText(
        label,
        labelLeft + 6f,
        labelBottom,
        textPaint,
      )

      // 신뢰도가 충분한 관절 사이에 선을 그린다.
      skeletonConnections.forEach { connection ->
        val firstKeypoint =
          detection.keypoints[connection[0]]

        val secondKeypoint =
          detection.keypoints[connection[1]]

        if (
          firstKeypoint.confidence >= 0.25f &&
          secondKeypoint.confidence >= 0.25f
        ) {
          canvas.drawLine(
            firstKeypoint.x,
            firstKeypoint.y,
            secondKeypoint.x,
            secondKeypoint.y,
            skeletonPaint,
          )
        }
      }

      // 17개 관절점을 점으로 그린다.
      detection.keypoints.forEach { keypoint ->
        if (keypoint.confidence >= 0.25f) {
          canvas.drawCircle(
            keypoint.x,
            keypoint.y,
            baseStrokeWidth * 1.5f,
            keypointPaint,
          )
        }
      }
    }

    return resultBitmap
  }
  private fun saveDebugFrame(bitmap: Bitmap) {
    try {
      val context = getApplication<Application>()

      // 앱 전용 Pictures 폴더를 가져온다.
      val baseDirectory =
        context.getExternalFilesDir(Environment.DIRECTORY_PICTURES)
          ?: context.filesDir

      // debug_frames 폴더를 만든다.
      val debugDirectory =
        File(baseDirectory, "debug_frames").apply {
          mkdirs()
        }

      // 현재 시간을 이용해 파일 이름을 만든다.
      val outputFile =
        File(
          debugDirectory,
          "rayban_frame_${System.currentTimeMillis()}.jpg",
        )

      // Bitmap을 JPEG 파일로 저장한다.
      FileOutputStream(outputFile).use { outputStream ->
        val success =
          bitmap.compress(
            Bitmap.CompressFormat.JPEG,
            95,
            outputStream,
          )

        if (!success) {
          throw IOException("Bitmap JPEG 압축에 실패했습니다.")
        }
      }

      Log.d(
        TAG,
        "Debug frame saved: ${outputFile.absolutePath}, " +
                "size=${bitmap.width}x${bitmap.height}",
      )
    } catch (exception: Exception) {
      Log.e(TAG, "Debug frame save failed", exception)
    } finally {
      bitmap.recycle()
    }
  }
  private fun handlePhotoData(photo: PhotoData) {
    val capturedPhoto =
        when (photo) {
          is PhotoData.Bitmap -> photo.bitmap
          is PhotoData.HEIC -> {
            val byteArray = ByteArray(photo.data.remaining())
            photo.data.get(byteArray)

            // Extract EXIF transformation matrix and apply to bitmap
            val exifInfo = getExifInfo(byteArray)
            val transform = getTransform(exifInfo)
            decodeHeic(byteArray, transform)
          }
        }
    _uiState.update { it.copy(capturedPhoto = capturedPhoto, isShareDialogVisible = true) }
  }

  // HEIC Decoding with EXIF transformation
  private fun decodeHeic(heicBytes: ByteArray, transform: Matrix): Bitmap {
    val bitmap = BitmapFactory.decodeByteArray(heicBytes, 0, heicBytes.size)
    return applyTransform(bitmap, transform)
  }

  private fun getExifInfo(heicBytes: ByteArray): ExifInterface? {
    return try {
      ByteArrayInputStream(heicBytes).use { inputStream -> ExifInterface(inputStream) }
    } catch (e: IOException) {
      Log.w(TAG, "Failed to read EXIF from HEIC", e)
      null
    }
  }

  private fun getTransform(exifInfo: ExifInterface?): Matrix {
    val matrix = Matrix()

    if (exifInfo == null) {
      return matrix // Identity matrix (no transformation)
    }

    when (
        exifInfo.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL,
        )
    ) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> {
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_180 -> {
        matrix.postRotate(180f)
      }
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
        matrix.postScale(1f, -1f)
      }
      ExifInterface.ORIENTATION_TRANSPOSE -> {
        matrix.postRotate(90f)
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_90 -> {
        matrix.postRotate(90f)
      }
      ExifInterface.ORIENTATION_TRANSVERSE -> {
        matrix.postRotate(270f)
        matrix.postScale(-1f, 1f)
      }
      ExifInterface.ORIENTATION_ROTATE_270 -> {
        matrix.postRotate(270f)
      }
      ExifInterface.ORIENTATION_NORMAL,
      ExifInterface.ORIENTATION_UNDEFINED -> {
        // No transformation needed
      }
    }

    return matrix
  }

  private fun applyTransform(bitmap: Bitmap, matrix: Matrix): Bitmap {
    if (matrix.isIdentity) {
      return bitmap
    }

    return try {
      val transformed = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
      if (transformed != bitmap) {
        bitmap.recycle()
      }
      transformed
    } catch (e: OutOfMemoryError) {
      Log.e(TAG, "Failed to apply transformation due to memory", e)
      bitmap
    }
  }

  override fun onCleared() {
    stopStream()
    yoloPoseModel.close()
    super.onCleared()
    session?.stop()
    session = null
  }

  class Factory(
      private val application: Application,
      private val wearablesViewModel: WearablesViewModel,
  ) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
      if (modelClass.isAssignableFrom(StreamViewModel::class.java)) {
        @Suppress("UNCHECKED_CAST", "KotlinGenericsCast")
        return StreamViewModel(
            application = application,
            wearablesViewModel = wearablesViewModel,
        )
            as T
      }
      throw IllegalArgumentException("Unknown ViewModel class")
    }
  }
}
