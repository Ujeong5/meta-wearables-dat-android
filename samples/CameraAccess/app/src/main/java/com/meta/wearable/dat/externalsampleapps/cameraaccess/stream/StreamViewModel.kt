package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.annotation.SuppressLint
import android.app.Application
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.os.SystemClock
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
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.roundToInt
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

@SuppressLint("AutoCloseableUse")
class StreamViewModel(
  application: Application,
  private val wearablesViewModel: WearablesViewModel,
) : AndroidViewModel(application) {

  companion object {
    private const val TAG = "CameraAccess:StreamViewModel"

    private val INITIAL_STATE = StreamUiState()

    private val SESSION_TERMINAL_STATES =
      setOf(StreamState.CLOSED)

    // YOLO를 500ms마다 최대 한 번 실행한다.
    private const val YOLO_INTERVAL_MS = 500L

    private const val MIN_PERSON_CONFIDENCE = 0.45f
    private const val MIN_KEYPOINT_CONFIDENCE = 0.35f

    private const val MIN_BOX_WIDTH_RATIO = 0.08f
    private const val MIN_BOX_HEIGHT_RATIO = 0.20f
  }

  private val yoloPoseModel =
    YoloPoseModel(getApplication())

  // 현재 YOLO 추론이 실행 중인지 나타낸다.
  private val isYoloRunning =
    AtomicBoolean(false)

  // 마지막 YOLO 시작 시각이다.
  private var lastYoloStartedAtMs =
    0L

  // 현재 실행 중인 YOLO 코루틴이다.
  private var yoloJob: Job? =
    null

  // 가장 최근 프레임의 사람 검출 결과다.
  @Volatile
  private var latestDetections: List<PoseDetection> =
    emptyList()

  // 검출에 사용된 프레임 크기다.
  @Volatile
  private var latestDetectionWidth =
    0

  @Volatile
  private var latestDetectionHeight =
    0

  private val deviceSelector: DeviceSelector =
    wearablesViewModel.deviceSelector

  private var session: DeviceSession? =
    null

  private val _uiState =
    MutableStateFlow(INITIAL_STATE)

  val uiState: StateFlow<StreamUiState> =
    _uiState.asStateFlow()

  private var videoJob: Job? =
    null

  private var stateJob: Job? =
    null

  private var errorJob: Job? =
    null

  private var sessionErrorJob: Job? =
    null

  private var sessionStateJob: Job? =
    null

  private var stream: Stream? =
    null

  private var previousDeviceSessionState: DeviceSessionState? =
    null

  // 색 변환이 끝난 프레임을 표시 시각에 맞춰 전달하는 큐다.
  private var presentationQueue: PresentationQueue? =
    null

  fun startStream() {
    videoJob?.cancel()
    stateJob?.cancel()
    errorJob?.cancel()
    sessionErrorJob?.cancel()
    sessionStateJob?.cancel()

    yoloJob?.cancel()
    yoloJob = null
    isYoloRunning.set(false)
    lastYoloStartedAtMs = 0L
    latestDetections = emptyList()
    latestDetectionWidth = 0
    latestDetectionHeight = 0

    presentationQueue?.stop()
    presentationQueue = null

    previousDeviceSessionState = null

    val queue =
      PresentationQueue(
        bufferDelayMs = 100L,
        maxQueueSize = 15,
        onFrameReady = { frame ->
          viewModelScope.launch(Dispatchers.Main) {
            _uiState.update {
              it.copy(
                videoFrame = frame.bitmap,
                videoFrameCount = it.videoFrameCount + 1,
              )
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

          sessionErrorJob =
            viewModelScope.launch {
              createdSession.errors.collect { error ->
                handleSessionError(error)
              }
            }

          createdSession.start()
        }
        .onFailure { error, _ ->
          Log.e(
            TAG,
            "Failed to create session: ${error.description}",
          )

          handleSessionError(error)
        }

      if (session == null) {
        return
      }
    }

    startStreamInternal()
  }

  private fun startStreamInternal() {
    Log.d(
      TAG,
      "startStreamInternal() - collecting session state",
    )

    sessionStateJob =
      viewModelScope.launch {
        session?.state?.collect { currentState ->
          val previousState =
            previousDeviceSessionState

          previousDeviceSessionState =
            currentState

          if (currentState == DeviceSessionState.STARTED) {
            wearablesViewModel.setDatAppUpdateRequired(false)

            if (
              previousState == DeviceSessionState.PAUSED &&
              stream != null
            ) {
              // 기기 탭 제스처로 PAUSED → STARTED가 된 경우
              // 기존 스트림을 다시 만들지 않는다.
              Log.d(
                TAG,
                "Session resumed from PAUSED — stream stays alive",
              )
              return@collect
            }

            videoJob?.cancel()
            stateJob?.cancel()
            errorJob?.cancel()

            stream?.stop()
            stream = null

            session
              ?.addStream(
                StreamConfiguration(
                  videoQuality = VideoQuality.HIGH,
                  frameRate = 2,
                ),
              )
              ?.onSuccess { addedStream ->
                stream = addedStream

                videoJob =
                  viewModelScope.launch {
                    Log.d(
                      TAG,
                      "Collecting video frames from stream",
                    )

                    addedStream.videoStream.collect { videoFrame ->
                      handleVideoFrame(videoFrame)
                    }

                    Log.d(
                      TAG,
                      "Video stream collection ended",
                    )
                  }

                stateJob =
                  viewModelScope.launch {
                    addedStream.state.collect { streamState ->
                      val previousStreamState =
                        _uiState.value.streamState

                      Log.d(
                        TAG,
                        "Stream state changed: " +
                                "$previousStreamState -> $streamState",
                      )

                      _uiState.update {
                        it.copy(streamState = streamState)
                      }

                      val wasActive =
                        previousStreamState !in
                                SESSION_TERMINAL_STATES

                      val isTerminated =
                        streamState in
                                SESSION_TERMINAL_STATES

                      if (wasActive && isTerminated) {
                        Log.d(
                          TAG,
                          "Terminal state reached, navigating back",
                        )

                        stopStream()
                        wearablesViewModel
                          .navigateToDeviceSelection()
                      }
                    }
                  }

                errorJob =
                  viewModelScope.launch {
                    addedStream.errorStream.collect { error ->
                      Log.d(
                        TAG,
                        "Stream error received: $error " +
                                "(description: ${error.description})",
                      )

                      if (error == StreamError.STREAM_ERROR) {
                        Log.d(
                          TAG,
                          "Non-critical error, stream continues",
                        )
                        return@collect
                      }

                      stopStream()

                      wearablesViewModel
                        .navigateToDeviceSelection()

                      wearablesViewModel.setRecentError(
                        error.getLocalizedDescription(
                          getApplication(),
                        ),
                      )
                    }
                  }

                addedStream.start()
              }
              ?.onFailure { error, _ ->
                Log.e(
                  TAG,
                  "Failed to add stream to session: " +
                          error.description,
                )
              }
          } else if (
            currentState == DeviceSessionState.PAUSED
          ) {
            Log.d(
              TAG,
              "Session paused (tap gesture) — " +
                      "keeping stream alive for resume",
            )
          }
        }
      }
  }

  fun stopStream() {
    videoJob?.cancel()
    videoJob = null

    yoloJob?.cancel()
    yoloJob = null

    isYoloRunning.set(false)
    lastYoloStartedAtMs = 0L

    latestDetections = emptyList()
    latestDetectionWidth = 0
    latestDetectionHeight = 0

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

    _uiState.value = INITIAL_STATE

    stream?.stop()
    stream = null

    session?.stop()
    session = null

    previousDeviceSessionState = null
  }

  private fun handleSessionError(
    error: DeviceSessionError,
  ) {
    Log.e(
      TAG,
      "Session error: ${error.description}",
    )

    val alreadyShowingUpdateRequired =
      wearablesViewModel.uiState.value.isFirmwareUpdateRequired ||
              wearablesViewModel.uiState.value.isDatAppUpdateRequired

    if (
      error ==
      DeviceSessionError
        .DAT_APP_ON_THE_GLASSES_UPDATE_REQUIRED
    ) {
      wearablesViewModel.setDatAppUpdateRequired(true)
    }

    if (
      alreadyShowingUpdateRequired &&
      error == DeviceSessionError.SESSION_ENDED_BY_DEVICE
    ) {
      stopStream()
      wearablesViewModel.navigateToDeviceSelection()
      return
    }

    wearablesViewModel.setRecentError(
      error.getLocalizedDescription(
        getApplication(),
      ),
    )

    stopStream()
    wearablesViewModel.navigateToDeviceSelection()
  }

  fun capturePhoto() {
    if (_uiState.value.isCapturing) {
      Log.d(
        TAG,
        "Photo capture already in progress, ignoring request",
      )
      return
    }

    if (
      _uiState.value.streamState ==
      StreamState.STREAMING
    ) {
      Log.d(
        TAG,
        "Starting photo capture",
      )

      _uiState.update {
        it.copy(isCapturing = true)
      }

      viewModelScope.launch {
        stream
          ?.capturePhoto()
          ?.onSuccess { photoData ->
            Log.d(
              TAG,
              "Photo capture successful",
            )

            handlePhotoData(photoData)

            _uiState.update {
              it.copy(isCapturing = false)
            }
          }
          ?.onFailure { error, _ ->
            Log.e(
              TAG,
              "Photo capture failed: ${error.description}",
            )

            _uiState.update {
              it.copy(isCapturing = false)
            }
          }
      }
    } else {
      Log.w(
        TAG,
        "Cannot capture photo: stream not active " +
                "(state=${_uiState.value.streamState})",
      )
    }
  }

  fun showShareDialog() {
    _uiState.update {
      it.copy(isShareDialogVisible = true)
    }
  }

  fun hideShareDialog() {
    _uiState.update {
      it.copy(isShareDialogVisible = false)
    }
  }

  fun sharePhoto(
    bitmap: Bitmap,
  ) {
    val context =
      getApplication<Application>()

    val imagesFolder =
      File(
        context.cacheDir,
        "images",
      )

    try {
      imagesFolder.mkdirs()

      val file =
        File(
          imagesFolder,
          "shared_image.png",
        )

      FileOutputStream(file).use { outputStream ->
        bitmap.compress(
          Bitmap.CompressFormat.PNG,
          90,
          outputStream,
        )
      }

      val uri =
        FileProvider.getUriForFile(
          context,
          "${context.packageName}.fileprovider",
          file,
        )

      val intent =
        Intent(Intent.ACTION_SEND).apply {
          flags = Intent.FLAG_ACTIVITY_NEW_TASK
          putExtra(Intent.EXTRA_STREAM, uri)
          type = "image/png"
          addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }

      val chooser =
        Intent.createChooser(
          intent,
          "Share Image",
        ).apply {
          flags = Intent.FLAG_ACTIVITY_NEW_TASK
        }

      context.startActivity(chooser)
    } catch (exception: IOException) {
      Log.e(
        TAG,
        "Failed to share photo",
        exception,
      )
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
     * YOLO는 화면에 박스를 그리기 전의 원본 프레임을 분석해야 한다.
     * 따라서 이 호출이 drawLatestDetectionsInPlace()보다 먼저 와야 한다.
     */
    startRealtimeYoloIfNeeded(bitmap)

    /*
     * 스마트폰 미리보기에는 사람 Bounding Box와 신뢰도만 표시한다.
     * 관절점과 skeleton 선은 그리지 않는다.
     */
    drawLatestDetectionsInPlace(bitmap)

    presentationQueue?.enqueue(
      bitmap,
      videoFrame.presentationTimeUs,
    )
  }

  private fun startRealtimeYoloIfNeeded(
    rawBitmap: Bitmap,
  ) {
    val currentTime =
      SystemClock.elapsedRealtime()

    if (
      currentTime - lastYoloStartedAtMs <
      YOLO_INTERVAL_MS
    ) {
      return
    }

    // 이전 추론이 끝나지 않았다면 새 프레임은 버린다.
    if (
      !isYoloRunning.compareAndSet(
        false,
        true,
      )
    ) {
      return
    }

    /*
     * rawBitmap은 PresentationQueue에서도 사용하므로
     * 비동기 YOLO 추론용 복사본을 만든다.
     */
    val inferenceBitmap =
      rawBitmap.copy(
        Bitmap.Config.ARGB_8888,
        false,
      )

    if (inferenceBitmap == null) {
      Log.e(
        TAG,
        "Failed to copy Bitmap for realtime YOLO",
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
          val rawDetections =
            yoloPoseModel.detect(
              inferenceBitmap,
            )

          val detections =
            filterValidPersonDetections(
              detections = rawDetections,
              frameWidth = inferenceBitmap.width,
              frameHeight = inferenceBitmap.height,
            )

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
                    "raw=${rawDetections.size}, " +
                    "valid=${detections.size}",
          )
        } catch (
          exception: CancellationException
        ) {
          throw exception
        } catch (
          exception: Exception
        ) {
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

  private fun filterValidPersonDetections(
    detections: List<PoseDetection>,
    frameWidth: Int,
    frameHeight: Int,
  ): List<PoseDetection> {
    return detections.filter { detection ->
      // YOLO의 사람 검출 신뢰도가 너무 낮으면 제외한다.
      if (detection.confidence < MIN_PERSON_CONFIDENCE) {
        return@filter false
      }

      val boxWidth =
        detection.box.width()

      val boxHeight =
        detection.box.height()

      // 지나치게 작은 영역은 사람으로 인정하지 않는다.
      if (
        boxWidth < frameWidth * MIN_BOX_WIDTH_RATIO ||
        boxHeight < frameHeight * MIN_BOX_HEIGHT_RATIO
      ) {
        return@filter false
      }

      // COCO Pose 관절 번호
      // 5: 왼쪽 어깨
      // 6: 오른쪽 어깨
      // 11: 왼쪽 골반
      // 12: 오른쪽 골반
      val leftShoulder =
        detection.keypoints.getOrNull(5)?.confidence ?: 0f

      val rightShoulder =
        detection.keypoints.getOrNull(6)?.confidence ?: 0f

      val leftHip =
        detection.keypoints.getOrNull(11)?.confidence ?: 0f

      val rightHip =
        detection.keypoints.getOrNull(12)?.confidence ?: 0f

      val visibleShoulderCount =
        listOf(
          leftShoulder,
          rightShoulder,
        ).count {
          it >= MIN_KEYPOINT_CONFIDENCE
        }

      val visibleHipCount =
        listOf(
          leftHip,
          rightHip,
        ).count {
          it >= MIN_KEYPOINT_CONFIDENCE
        }

      val visibleTorsoPointCount =
        listOf(
          leftShoulder,
          rightShoulder,
          leftHip,
          rightHip,
        ).count {
          it >= MIN_KEYPOINT_CONFIDENCE
        }

      /*
       * 최소한 어깨 하나, 골반 하나,
       * 그리고 몸통 관절 총 3개 이상이 확인돼야 사람으로 인정한다.
       *
       * 손이나 팔만 보이는 경우에는 이 조건을 통과하기 어렵다.
       */
      visibleShoulderCount >= 1 &&
              visibleHipCount >= 1 &&
              visibleTorsoPointCount >= 3
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

    detections.forEach { detection ->
      // 사람 Bounding Box
      canvas.drawRect(
        detection.box,
        boxPaint,
      )

      // 사람 검출 신뢰도
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
    }
  }

  private fun handlePhotoData(
    photo: PhotoData,
  ) {
    val capturedPhoto =
      when (photo) {
        is PhotoData.Bitmap ->
          photo.bitmap

        is PhotoData.HEIC -> {
          val byteArray =
            ByteArray(
              photo.data.remaining(),
            )

          photo.data.get(byteArray)

          val exifInfo =
            getExifInfo(byteArray)

          val transform =
            getTransform(exifInfo)

          decodeHeic(
            byteArray,
            transform,
          )
        }
      }

    _uiState.update {
      it.copy(
        capturedPhoto = capturedPhoto,
        isShareDialogVisible = true,
      )
    }
  }

  private fun decodeHeic(
    heicBytes: ByteArray,
    transform: Matrix,
  ): Bitmap {
    val bitmap =
      BitmapFactory.decodeByteArray(
        heicBytes,
        0,
        heicBytes.size,
      )

    return applyTransform(
      bitmap,
      transform,
    )
  }

  private fun getExifInfo(
    heicBytes: ByteArray,
  ): ExifInterface? {
    return try {
      ByteArrayInputStream(heicBytes).use { inputStream ->
        ExifInterface(inputStream)
      }
    } catch (exception: IOException) {
      Log.w(
        TAG,
        "Failed to read EXIF from HEIC",
        exception,
      )
      null
    }
  }

  private fun getTransform(
    exifInfo: ExifInterface?,
  ): Matrix {
    val matrix =
      Matrix()

    if (exifInfo == null) {
      return matrix
    }

    when (
      exifInfo.getAttributeInt(
        ExifInterface.TAG_ORIENTATION,
        ExifInterface.ORIENTATION_NORMAL,
      )
    ) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> {
        matrix.postScale(
          -1f,
          1f,
        )
      }

      ExifInterface.ORIENTATION_ROTATE_180 -> {
        matrix.postRotate(180f)
      }

      ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
        matrix.postScale(
          1f,
          -1f,
        )
      }

      ExifInterface.ORIENTATION_TRANSPOSE -> {
        matrix.postRotate(90f)
        matrix.postScale(
          -1f,
          1f,
        )
      }

      ExifInterface.ORIENTATION_ROTATE_90 -> {
        matrix.postRotate(90f)
      }

      ExifInterface.ORIENTATION_TRANSVERSE -> {
        matrix.postRotate(270f)
        matrix.postScale(
          -1f,
          1f,
        )
      }

      ExifInterface.ORIENTATION_ROTATE_270 -> {
        matrix.postRotate(270f)
      }

      ExifInterface.ORIENTATION_NORMAL,
      ExifInterface.ORIENTATION_UNDEFINED,
        -> {
        // 변환하지 않는다.
      }
    }

    return matrix
  }

  private fun applyTransform(
    bitmap: Bitmap,
    matrix: Matrix,
  ): Bitmap {
    if (matrix.isIdentity) {
      return bitmap
    }

    return try {
      val transformed =
        Bitmap.createBitmap(
          bitmap,
          0,
          0,
          bitmap.width,
          bitmap.height,
          matrix,
          true,
        )

      if (transformed != bitmap) {
        bitmap.recycle()
      }

      transformed
    } catch (error: OutOfMemoryError) {
      Log.e(
        TAG,
        "Failed to apply transformation due to memory",
        error,
      )
      bitmap
    }
  }

  override fun onCleared() {
    stopStream()
    yoloPoseModel.close()
    super.onCleared()
  }

  class Factory(
    private val application: Application,
    private val wearablesViewModel: WearablesViewModel,
  ) : ViewModelProvider.Factory {

    override fun <T : ViewModel> create(
      modelClass: Class<T>,
    ): T {
      if (
        modelClass.isAssignableFrom(
          StreamViewModel::class.java,
        )
      ) {
        @Suppress(
          "UNCHECKED_CAST",
          "KotlinGenericsCast",
        )
        return StreamViewModel(
          application = application,
          wearablesViewModel = wearablesViewModel,
        ) as T
      }

      throw IllegalArgumentException(
        "Unknown ViewModel class",
      )
    }
  }
}