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
import android.os.Environment
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
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

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

    // 스트림 프레임의 YOLO 분석 간격
    private const val YOLO_INTERVAL_MS = 500L

    // 자동 사진 촬영 요청 사이의 최소 간격
    //private const val AUTO_CAPTURE_COOLDOWN_MS = 6_000L

    // 유효한 사람 판단 기준
    private const val MIN_PERSON_CONFIDENCE = 0.45f
    private const val MIN_KEYPOINT_CONFIDENCE = 0.35f
    private const val MIN_BOX_WIDTH_RATIO = 0.08f
    private const val MIN_BOX_HEIGHT_RATIO = 0.20f

    // 사람 crop에 추가할 바깥 여백
    private const val PERSON_CROP_PADDING_RATIO = 0.08f
    private const val PERSON_CROP_JPEG_QUALITY = 95

    /*
     * 한 프레임의 순간 오검출로 바로 촬영하지 않도록
     * 유효한 사람을 연속 2회 확인한 뒤 촬영한다.
     */
    private const val REQUIRED_CONSECUTIVE_VALID_DETECTIONS = 2

    /*
     * 같은 사람을 반복 촬영하지 않는다.
     * 사람이 연속 3회 사라진 뒤에만 다음 자동 촬영을 허용한다.
     */
    private const val REQUIRED_CONSECUTIVE_EMPTY_DETECTIONS_TO_REARM = 3

    /*
     * 고해상도 Bitmap이 메모리에 무제한 쌓이는 것을 방지한다.
     * 현재 처리 중인 사진과 대기 중인 사진을 합쳐 최대 2장만 유지한다.
     */
    private const val MAX_PENDING_PHOTO_PROCESSING = 2
  }

  private val yoloPoseModel =
    YoloPoseModel(getApplication())

  /*
   * 사람 crop에서 얼굴을 검출하고 모자이크한다.
   * 얼굴을 찾지 못하면 privacy fallback으로 crop 상단의 머리 영역을 모자이크한다.
   */
  private val faceMosaicProcessor =
    FaceMosaicProcessor(getApplication())

  // 저장된 모자이크 crop 전체를 데스크톱으로 전송한다.
  private val batchCropUploader =
    BatchCropUploader(
      context = getApplication<Application>(),
      serverBaseUrl = UploadConfig.SERVER_BASE_URL,
    )

  // 스트림 종료 또는 일괄 전송이 실행 중인지 나타낸다.
  private val isStoppingOrUploading =
    AtomicBoolean(false)

  private var uploadJob: Job? =
    null

  // 현재 YOLO 추론이 실행 중인지 나타낸다.
  private val isYoloRunning =
    AtomicBoolean(false)

  // 마지막 YOLO 시작 시각이다.
  private var lastYoloStartedAtMs =
    0L

  // 현재 실행 중인 YOLO 코루틴이다.
  private var yoloJob: Job? =
    null

  // 현재 사진 촬영 요청이 실행 중인지 나타낸다.
  private val isPhotoCaptureRunning =
    AtomicBoolean(false)

  // 현재 사진 촬영 코루틴이다.
  private var photoCaptureJob: Job? =
    null

  /*
   * 촬영된 사진의 처리 작업들을 보관한다.
   * 사진 촬영과 이전 사진의 YOLO/crop/저장을 병렬로 진행할 수 있다.
   */
  private val photoProcessingJobs =
    mutableSetOf<Job>()

  // 처리 중이거나 대기 중인 고해상도 사진 수다.
  private val pendingPhotoProcessingCount =
    AtomicInteger(0)

  /*
   * 스트림 YOLO와 고해상도 사진 YOLO가 같은 모델을 동시에 사용하지 않도록 한다.
   * 고해상도 사진 YOLO가 끝나면 lock이 풀리므로,
   * 이전 사진의 JPEG 저장과 다음 스트림 YOLO는 겹쳐서 실행될 수 있다.
   */
  private val yoloMutex =
    Mutex()

  // 고해상도 사진 YOLO를 기다리거나 실행 중인 작업 수다.
  private val highResYoloDemandCount =
    AtomicInteger(0)

  // 마지막 자동 촬영 요청 시각
  private var lastAutoCaptureRequestedAtMs =
    0L

  // 유효한 사람이 연속으로 검출된 횟수다.
  private var consecutiveValidDetectionCount =
    0

  // 유효한 사람이 연속으로 검출되지 않은 횟수다.
  private var consecutiveEmptyDetectionCount =
    0

  /*
   * true일 때만 자동 촬영할 수 있다.
   * 한 번 촬영한 뒤 사람이 화면에서 사라질 때까지 false로 유지한다.
   */
  private var autoCaptureArmed =
    true

  // 가장 최근 프레임의 유효한 사람 검출 결과다.
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
    if (isStoppingOrUploading.get()) {
      Log.w(
        TAG,
        "Cannot start stream while stopping or uploading",
      )
      return
    }

    _uiState.update {
      it.copy(
        uploadStatusText = null,
        pendingUploadCount =
          batchCropUploader.countPendingFiles(),
        shouldNavigateAfterUpload = false,
      )
    }

    videoJob?.cancel()
    stateJob?.cancel()
    errorJob?.cancel()
    sessionErrorJob?.cancel()
    sessionStateJob?.cancel()

    yoloJob?.cancel()
    yoloJob = null

    photoCaptureJob?.cancel()
    photoCaptureJob = null

    cancelPhotoProcessingJobs()

    isYoloRunning.set(false)
    isPhotoCaptureRunning.set(false)

    pendingPhotoProcessingCount.set(0)
    highResYoloDemandCount.set(0)

    lastYoloStartedAtMs = 0L
    lastAutoCaptureRequestedAtMs = 0L

    consecutiveValidDetectionCount = 0
    consecutiveEmptyDetectionCount = 0
    autoCaptureArmed = true

    latestDetections = emptyList()
    latestDetectionWidth = 0
    latestDetectionHeight = 0

    presentationQueue?.stop()
    presentationQueue = null

    previousDeviceSessionState = null

    val queue =
      PresentationQueue(
        /*
         * 실시간 미리보기에 인위적인 지연을 추가하지 않는다.
         * 오래된 프레임 대신 가능한 한 최신 프레임을 바로 표시한다.
         */
        bufferDelayMs = 0L,
        maxQueueSize = 2,
        onFrameReady = { frame ->
          /*
           * MutableStateFlow.update()는 스레드 안전하다.
           * 매 프레임마다 Main 코루틴을 새로 만들지 않아
           * UI 작업이 뒤늦게 몰리는 현상을 줄인다.
           */
          _uiState.update {
            it.copy(
              videoFrame = frame.bitmap,
              videoFrameCount = it.videoFrameCount + 1,
            )
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
                  videoQuality = VideoQuality.MEDIUM,
                  frameRate = 24,
                ),
              )
              ?.onSuccess { addedStream ->
                stream = addedStream

                videoJob =
                  viewModelScope.launch(Dispatchers.Default) {
                    Log.d(
                      TAG,
                      "Collecting latest video frames " +
                              "on background dispatcher",
                    )

                    /*
                     * YUV 변환, Bitmap 복사, Bounding Box 그리기를
                     * Main 스레드에서 실행하지 않는다.
                     *
                     * conflate()는 처리 도중 여러 프레임이 들어오면
                     * 중간의 오래된 프레임을 버리고 최신 프레임을 남긴다.
                     */
                    addedStream.videoStream
                      .conflate()
                      .collect { videoFrame ->
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
                        if (isStoppingOrUploading.get()) {
                          return@collect
                        }

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
                      if (isStoppingOrUploading.get()) {
                        return@collect
                      }

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

  /**
   * 오류, 화면 종료 등 일반적인 정리에서 사용한다.
   * 진행 중인 사진 처리와 업로드도 취소한다.
   */
  fun stopStream() {
    uploadJob?.cancel()
    uploadJob = null

    isStoppingOrUploading.set(false)

    videoJob?.cancel()
    videoJob = null

    yoloJob?.cancel()
    yoloJob = null

    photoCaptureJob?.cancel()
    photoCaptureJob = null

    cancelPhotoProcessingJobs()

    isYoloRunning.set(false)
    isPhotoCaptureRunning.set(false)

    pendingPhotoProcessingCount.set(0)
    highResYoloDemandCount.set(0)

    resetDetectionState()

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

  /**
   * 사용자가 "스트리밍 중지 및 전송"을 눌렀을 때 실행한다.
   *
   * 1. 새 프레임 분석과 새 자동 촬영을 막는다.
   * 2. 이미 시작된 사진 촬영과 crop 저장을 기다린다.
   * 3. 스트림을 닫는다.
   * 4. 저장된 JPEG 전체를 하나의 multipart 요청으로 전송한다.
   */
  fun stopStreamAndUpload() {
    if (
      !isStoppingOrUploading.compareAndSet(
        false,
        true,
      )
    ) {
      return
    }

    uploadJob =
      viewModelScope.launch {
        try {
          _uiState.update {
            it.copy(
              isUploading = true,
              uploadStatusText =
                "마지막 사진 처리를 기다리는 중입니다...",
              shouldNavigateAfterUpload = false,
            )
          }

          // 새 프레임 YOLO와 새 자동 촬영을 즉시 막는다.
          videoJob?.cancel()
          videoJob = null

          yoloJob?.cancel()
          yoloJob = null

          isYoloRunning.set(false)

          presentationQueue?.stop()
          presentationQueue = null

          // 이미 시작된 고해상도 촬영은 최대 30초 기다린다.
          val captureCompleted =
            withTimeoutOrNull(30_000L) {
              photoCaptureJob?.join()
              true
            } ?: false

          if (!captureCompleted) {
            Log.w(
              TAG,
              "Photo capture did not finish before upload timeout",
            )

            val timedOutCaptureJob =
              photoCaptureJob

            timedOutCaptureJob?.cancel()
            timedOutCaptureJob?.join()
            photoCaptureJob = null
            finishPhotoCapture()
          }

          // 촬영 완료 후 생성된 crop/모자이크/저장 작업까지 기다린다.
          val processingJobs =
            synchronized(photoProcessingJobs) {
              photoProcessingJobs.toList()
            }

          val processingCompleted =
            withTimeoutOrNull(90_000L) {
              processingJobs.joinAll()
              true
            } ?: false

          if (!processingCompleted) {
            Log.w(
              TAG,
              "Photo processing did not finish before upload timeout",
            )

            processingJobs.forEach { job ->
              job.cancel()
            }
            processingJobs.joinAll()
          }

          pendingPhotoProcessingCount.set(0)
          highResYoloDemandCount.set(0)
          isPhotoCaptureRunning.set(false)

          closeStreamSessionForUpload()

          uploadAllSavedPhotosInternal()
        } catch (exception: CancellationException) {
          throw exception
        } catch (exception: Exception) {
          Log.e(
            TAG,
            "Stop and upload failed",
            exception,
          )

          _uiState.update {
            it.copy(
              isUploading = false,
              uploadStatusText =
                "중지 또는 전송 중 오류가 발생했습니다: " +
                        (exception.message
                          ?: exception.javaClass.simpleName),
              pendingUploadCount =
                batchCropUploader.countPendingFiles(),
            )
          }
        } finally {
          isStoppingOrUploading.set(false)
          uploadJob = null
        }
      }
  }

  /**
   * 스트림이 이미 멈춘 뒤 전송에 실패한 파일을 다시 보낼 때 사용한다.
   */
  fun uploadSavedPhotos() {
    if (
      !isStoppingOrUploading.compareAndSet(
        false,
        true,
      )
    ) {
      return
    }

    uploadJob =
      viewModelScope.launch {
        try {
          _uiState.update {
            it.copy(
              isUploading = true,
              uploadStatusText = "저장된 사진을 전송하는 중입니다...",
              shouldNavigateAfterUpload = false,
            )
          }

          uploadAllSavedPhotosInternal()
        } catch (exception: CancellationException) {
          throw exception
        } catch (exception: Exception) {
          Log.e(
            TAG,
            "Saved photo upload failed",
            exception,
          )

          _uiState.update {
            it.copy(
              isUploading = false,
              uploadStatusText =
                "전송 중 오류가 발생했습니다: " +
                        (exception.message
                          ?: exception.javaClass.simpleName),
              pendingUploadCount =
                batchCropUploader.countPendingFiles(),
            )
          }
        } finally {
          isStoppingOrUploading.set(false)
          uploadJob = null
        }
      }
  }

  fun consumeUploadNavigation() {
    _uiState.update {
      it.copy(
        shouldNavigateAfterUpload = false,
      )
    }
  }

  private suspend fun uploadAllSavedPhotosInternal() {
    val pendingCount =
      batchCropUploader.countPendingFiles()

    _uiState.update {
      it.copy(
        isUploading = true,
        pendingUploadCount = pendingCount,
        uploadStatusText =
          if (pendingCount == 0) {
            "전송할 사진이 없습니다."
          } else {
            "사진 ${pendingCount}장을 한 번에 전송하는 중입니다..."
          },
      )
    }

    val result =
      batchCropUploader.uploadPendingCrops()

    val remainingCount =
      batchCropUploader.countPendingFiles()

    _uiState.update {
      it.copy(
        isUploading = false,
        uploadStatusText = result.message,
        pendingUploadCount = remainingCount,
        shouldNavigateAfterUpload = result.success,
      )
    }

    if (result.success) {
      Log.d(
        TAG,
        "Upload finished successfully: " +
                "batchId=${result.batchId}, " +
                "count=${result.transferredCount}",
      )
    } else {
      Log.e(
        TAG,
        "Upload failed: ${result.message}",
      )
    }
  }

  /**
   * 사진 처리를 취소하지 않고 스트림과 세션만 닫는다.
   * 호출 전에 진행 중인 사진 작업을 기다린다.
   */
  private fun closeStreamSessionForUpload() {
    stateJob?.cancel()
    stateJob = null

    errorJob?.cancel()
    errorJob = null

    sessionErrorJob?.cancel()
    sessionErrorJob = null

    sessionStateJob?.cancel()
    sessionStateJob = null

    stream?.stop()
    stream = null

    session?.stop()
    session = null

    previousDeviceSessionState = null

    resetDetectionState()

    _uiState.update {
      it.copy(
        streamState = StreamState.STOPPED,
        videoFrame = null,
        isCapturing = false,
      )
    }
  }

  private fun resetDetectionState() {
    lastYoloStartedAtMs = 0L
    lastAutoCaptureRequestedAtMs = 0L

    consecutiveValidDetectionCount = 0
    consecutiveEmptyDetectionCount = 0
    autoCaptureArmed = true

    latestDetections = emptyList()
    latestDetectionWidth = 0
    latestDetectionHeight = 0
  }

  private fun handleSessionError(
    error: DeviceSessionError,
  ) {
    if (isStoppingOrUploading.get()) {
      Log.d(
        TAG,
        "Ignoring session error during intentional stop/upload: " +
                error.description,
      )
      return
    }

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

  /*
   * 기존 촬영 버튼에서 호출한다.
   * 수동 촬영 결과는 기존처럼 공유 다이얼로그에 표시한다.
   */
  fun capturePhoto() {
    requestPhotoCapture(
      isAutomatic = false,
    )
  }

  private fun requestAutoPhotoCapture() {
    requestPhotoCapture(
      isAutomatic = true,
    )
  }

  private fun requestPhotoCapture(
    isAutomatic: Boolean,
  ) {
    if (isStoppingOrUploading.get()) {
      Log.d(
        TAG,
        "Photo capture skipped while stopping or uploading",
      )
      return
    }

    if (
      _uiState.value.streamState !=
      StreamState.STREAMING
    ) {
      Log.w(
        TAG,
        "Cannot capture photo: stream not active " +
                "(state=${_uiState.value.streamState})",
      )
      return
    }

    val currentTime =
      SystemClock.elapsedRealtime()

    /*if (
      isAutomatic &&
      currentTime - lastAutoCaptureRequestedAtMs <
      AUTO_CAPTURE_COOLDOWN_MS
    ) {
      return
    }*/

    /*
     * 실제 capturePhoto() 호출끼리만 겹치지 않게 막는다.
     * 이전 사진의 YOLO/crop/저장이 실행 중이어도 다음 사진 촬영은 가능하다.
     */
    if (
      !isPhotoCaptureRunning.compareAndSet(
        false,
        true,
      )
    ) {
      Log.d(
        TAG,
        "Photo capture already running, ignoring request",
      )
      return
    }

    val activeStream =
      stream

    if (activeStream == null) {
      finishPhotoCapture()
      return
    }

    if (isAutomatic) {
      lastAutoCaptureRequestedAtMs =
        currentTime

      Log.d(
        TAG,
        "Automatic high-resolution photo capture requested",
      )
    } else {
      Log.d(
        TAG,
        "Manual photo capture requested",
      )
    }

    _uiState.update {
      it.copy(isCapturing = true)
    }

    photoCaptureJob =
      viewModelScope.launch {
        activeStream
          .capturePhoto()
          .onSuccess { photoData ->
            val capturedBitmap =
              try {
                /*
                 * SDK가 제공한 PhotoData에서 독립적인 Bitmap을 만든다.
                 * 이 복사가 끝나는 즉시 촬영 lock을 해제한다.
                 */
                copyPhotoDataToBitmap(
                  photoData,
                )
              } catch (exception: Exception) {
                Log.e(
                  TAG,
                  "Failed to copy captured photo",
                  exception,
                )

                finishPhotoCapture()
                return@onSuccess
              }

            Log.d(
              TAG,
              if (isAutomatic) {
                "Automatic photo capture successful: " +
                        "${capturedBitmap.width}x${capturedBitmap.height}"
              } else {
                "Manual photo capture successful: " +
                        "${capturedBitmap.width}x${capturedBitmap.height}"
              },
            )

            /*
             * 여기서 촬영 lock을 먼저 해제한다.
             * 이후 YOLO/crop/저장은 별도 작업에서 계속된다.
             */
            finishPhotoCapture()

            launchCapturedPhotoProcessing(
              capturedBitmap = capturedBitmap,
              isAutomatic = isAutomatic,
            )
          }
          .onFailure { error, _ ->
            Log.e(
              TAG,
              "Photo capture failed: ${error.description}",
            )

            finishPhotoCapture()
          }
      }
  }

  private fun finishPhotoCapture() {
    _uiState.update {
      it.copy(isCapturing = false)
    }

    isPhotoCaptureRunning.set(false)
  }

  private fun launchCapturedPhotoProcessing(
    capturedBitmap: Bitmap,
    isAutomatic: Boolean,
  ) {
    while (true) {
      val currentCount =
        pendingPhotoProcessingCount.get()

      if (
        currentCount >=
        MAX_PENDING_PHOTO_PROCESSING
      ) {
        Log.w(
          TAG,
          "Photo processing queue is full; dropping captured photo",
        )

        if (!capturedBitmap.isRecycled) {
          capturedBitmap.recycle()
        }
        return
      }

      if (
        pendingPhotoProcessingCount.compareAndSet(
          currentCount,
          currentCount + 1,
        )
      ) {
        break
      }
    }

    val processingJob =
      viewModelScope.launch(
        context = Dispatchers.Default,
        start = CoroutineStart.LAZY,
      ) {
        try {
          if (isAutomatic) {
            processAutoCapturedPhoto(
              fullPhoto = capturedBitmap,
            )
          } else {
            processManualCapturedPhoto(
              capturedPhoto = capturedBitmap,
            )
          }
        } catch (
          exception: CancellationException
        ) {
          if (!capturedBitmap.isRecycled) {
            capturedBitmap.recycle()
          }

          throw exception
        } catch (
          exception: Exception
        ) {
          Log.e(
            TAG,
            "Captured photo processing failed",
            exception,
          )

          if (!capturedBitmap.isRecycled) {
            capturedBitmap.recycle()
          }
        } finally {
          decrementAtomicNonNegative(
            pendingPhotoProcessingCount,
          )
        }
      }

    synchronized(photoProcessingJobs) {
      photoProcessingJobs.add(
        processingJob,
      )
    }

    processingJob.invokeOnCompletion {
      synchronized(photoProcessingJobs) {
        photoProcessingJobs.remove(
          processingJob,
        )
      }
    }

    processingJob.start()
  }

  private fun decrementAtomicNonNegative(
    counter: AtomicInteger,
  ) {
    while (true) {
      val currentValue =
        counter.get()

      if (currentValue <= 0) {
        return
      }

      if (
        counter.compareAndSet(
          currentValue,
          currentValue - 1,
        )
      ) {
        return
      }
    }
  }

  private fun cancelPhotoProcessingJobs() {
    val jobsToCancel =
      synchronized(photoProcessingJobs) {
        photoProcessingJobs
          .toList()
          .also {
            photoProcessingJobs.clear()
          }
      }

    jobsToCancel.forEach { job ->
      job.cancel()
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
     * 스마트폰 미리보기에는 유효한 사람의 Bounding Box만 표시한다.
     * 글자, 퍼센트, 관절점, skeleton 선은 표시하지 않는다.
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
    if (isStoppingOrUploading.get()) {
      return
    }

    /*
     * 고해상도 사진 YOLO가 기다리거나 실행 중이면
     * 스트림 YOLO를 잠시 건너뛰어 고해상도 처리를 우선한다.
     *
     * 사진 촬영 자체와 crop 저장 중에는 스트림 YOLO가 계속 가능하다.
     */
    if (highResYoloDemandCount.get() > 0) {
      return
    }

    val currentTime =
      SystemClock.elapsedRealtime()

    if (
      currentTime - lastYoloStartedAtMs <
      YOLO_INTERVAL_MS
    ) {
      return
    }

    if (
      !isYoloRunning.compareAndSet(
        false,
        true,
      )
    ) {
      return
    }

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

        var yoloLockAcquired =
          false

        try {
          /*
           * 다른 YOLO가 모델을 사용 중이면 이 스트림 프레임은 버린다.
           * 스트림 프레임을 대기열에 쌓지 않기 위한 동작이다.
           */
          yoloLockAcquired =
            yoloMutex.tryLock()

          if (!yoloLockAcquired) {
            return@launch
          }

          /*
           * lock을 얻는 순간 고해상도 YOLO 요청이 생겼다면
           * 이 스트림 추론을 취소하고 lock을 넘긴다.
           */
          if (highResYoloDemandCount.get() > 0) {
            return@launch
          }

          val rawDetections =
            yoloPoseModel.detect(
              inferenceBitmap,
            )

          val validDetections =
            filterValidPersonDetections(
              detections = rawDetections,
              frameWidth = inferenceBitmap.width,
              frameHeight = inferenceBitmap.height,
            )

          latestDetections =
            validDetections

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
                    "valid=${validDetections.size}",
          )

          updateAutoCaptureState(
            validDetections = validDetections,
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
          if (yoloLockAcquired) {
            yoloMutex.unlock()
          }

          inferenceBitmap.recycle()
          isYoloRunning.set(false)
        }
      }
  }

  /*
   * 순간 오검출과 같은 사람의 반복 촬영을 막는다.
   *
   * 1. 유효한 사람을 연속 2회 확인해야 촬영한다.
   * 2. 촬영 후에는 자동 촬영을 잠근다.
   * 3. 사람이 연속 3회 사라져야 다시 촬영 가능 상태가 된다.
   */
  private fun updateAutoCaptureState(
    validDetections: List<PoseDetection>,
  ) {
    if (validDetections.isNotEmpty()) {
      consecutiveEmptyDetectionCount = 0

      if (!autoCaptureArmed) {
        consecutiveValidDetectionCount = 0
        return
      }

      consecutiveValidDetectionCount += 1

      if (
        consecutiveValidDetectionCount <
        REQUIRED_CONSECUTIVE_VALID_DETECTIONS
      ) {
        return
      }

      autoCaptureArmed = false
      consecutiveValidDetectionCount = 0

      Log.d(
        TAG,
        "Stable person detection confirmed; auto capture armed -> locked",
      )

      requestAutoPhotoCapture()
      return
    }

    consecutiveValidDetectionCount = 0

    if (autoCaptureArmed) {
      consecutiveEmptyDetectionCount = 0
      return
    }

    consecutiveEmptyDetectionCount += 1

    if (
      consecutiveEmptyDetectionCount >=
      REQUIRED_CONSECUTIVE_EMPTY_DETECTIONS_TO_REARM
    ) {
      consecutiveEmptyDetectionCount = 0
      autoCaptureArmed = true

      Log.d(
        TAG,
        "Person disappearance confirmed; automatic capture re-armed",
      )
    }
  }

  private fun filterValidPersonDetections(
    detections: List<PoseDetection>,
    frameWidth: Int,
    frameHeight: Int,
  ): List<PoseDetection> {
    return detections.filter { detection ->
      if (
        detection.confidence <
        MIN_PERSON_CONFIDENCE
      ) {
        return@filter false
      }

      val boxWidth =
        detection.box.width()

      val boxHeight =
        detection.box.height()

      if (
        boxWidth <
        frameWidth * MIN_BOX_WIDTH_RATIO ||
        boxHeight <
        frameHeight * MIN_BOX_HEIGHT_RATIO
      ) {
        return@filter false
      }

      // COCO Pose: 5/6은 어깨, 11/12는 골반이다.
      val leftShoulder =
        detection.keypoints
          .getOrNull(5)
          ?.confidence
          ?: 0f

      val rightShoulder =
        detection.keypoints
          .getOrNull(6)
          ?.confidence
          ?: 0f

      val leftHip =
        detection.keypoints
          .getOrNull(11)
          ?.confidence
          ?: 0f

      val rightHip =
        detection.keypoints
          .getOrNull(12)
          ?.confidence
          ?: 0f

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

      visibleShoulderCount >= 1 &&
              visibleHipCount >= 1 &&
              visibleTorsoPointCount >= 3
    }
  }

  /*
   * 휴대폰 미리보기 Bitmap에 유효한 사람의 Bounding Box만 그린다.
   * person 글자, 신뢰도 퍼센트, 관절점, skeleton 선은 표시하지 않는다.
   */
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

    detections.forEach { detection ->
      canvas.drawRect(
        detection.box,
        boxPaint,
      )
    }
  }

  /*
   * 자동 촬영된 고해상도 사진:
   * 1. 고해상도 YOLO는 한 번에 하나씩 실행
   * 2. 유효한 사람 중 Bounding Box가 가장 큰 한 명 선택
   * 3. 사람 영역 crop
   * 4. JPEG 저장
   *
   * YOLO lock은 검출 직후 해제하므로,
   * 이 사진의 crop 저장과 다음 사진의 YOLO는 겹쳐서 실행될 수 있다.
   */
  private suspend fun processAutoCapturedPhoto(
    fullPhoto: Bitmap,
  ) {
    try {
      Log.d(
        TAG,
        "High-resolution photo processing started: " +
                "${fullPhoto.width}x${fullPhoto.height}",
      )

      highResYoloDemandCount
        .incrementAndGet()

      val rawDetections =
        try {
          yoloMutex.withLock {
            yoloPoseModel.detect(
              fullPhoto,
            )
          }
        } finally {
          decrementAtomicNonNegative(
            highResYoloDemandCount,
          )
        }

      val validDetections =
        filterValidPersonDetections(
          detections = rawDetections,
          frameWidth = fullPhoto.width,
          frameHeight = fullPhoto.height,
        )

      Log.d(
        TAG,
        "High-resolution YOLO: " +
                "raw=${rawDetections.size}, " +
                "valid=${validDetections.size}",
      )

      val targetPerson =
        selectLargestPerson(
          detections = validDetections,
        )

      if (targetPerson == null) {
        Log.d(
          TAG,
          "No valid person remained in high-resolution photo",
        )
        return
      }

      val personCrop =
        cropPersonBitmap(
          sourceBitmap = fullPhoto,
          detection = targetPerson,
        )

      try {
        /*
         * 저장 전에 사람 crop 내부의 얼굴을 검출하고 모자이크한다.
         * 얼굴 검출이 0개이면 crop 상단 머리 영역에 fallback 모자이크를 적용한다.
         */
        val mosaicResult =
          faceMosaicProcessor.mosaicFaces(
            bitmap = personCrop,
          )

        Log.d(
          TAG,
          "Face mosaic completed: " +
                  "detectedFaces=${mosaicResult.detectedFaceCount}, " +
                  "mosaickedRegions=${mosaicResult.mosaickedRegionCount}, " +
                  "fallback=${mosaicResult.usedFallback}",
        )

        /*
         * 저장은 IO 스레드에서 실행한다.
         * 이 시점에는 YOLO lock이 이미 풀렸으므로
         * 다음 사진 YOLO 또는 스트림 YOLO와 병렬로 실행될 수 있다.
         */
        val savedFile =
          withContext(Dispatchers.IO) {
            savePersonCrop(
              bitmap = personCrop,
            )
          }

        Log.d(
          TAG,
          "Mosaicked person crop saved: " +
                  "${savedFile.absolutePath}, " +
                  "size=${personCrop.width}x${personCrop.height}",
        )
      } finally {
        if (!personCrop.isRecycled) {
          personCrop.recycle()
        }
      }
    } finally {
      if (!fullPhoto.isRecycled) {
        fullPhoto.recycle()
      }
    }
  }

  /*
   * 수동 촬영은 기존 샘플 동작을 유지한다.
   * capturedPhoto의 소유권을 UI 상태로 넘기므로 여기서는 recycle하지 않는다.
   */
  private fun processManualCapturedPhoto(
    capturedPhoto: Bitmap,
  ) {
    _uiState.update {
      it.copy(
        capturedPhoto = capturedPhoto,
        isShareDialogVisible = true,
      )
    }
  }

  /*
   * SDK PhotoData와 독립적으로 사용할 Bitmap 복사본을 만든다.
   * 자동 촬영 사진은 처리 완료 후 recycle되고,
   * 수동 촬영 사진은 UI 상태가 사용한다.
   */
  private fun copyPhotoDataToBitmap(
    photo: PhotoData,
  ): Bitmap {
    return when (photo) {
      is PhotoData.Bitmap -> {
        photo.bitmap.copy(
          Bitmap.Config.ARGB_8888,
          false,
        ) ?: throw IOException(
          "Failed to copy captured Bitmap",
        )
      }

      is PhotoData.HEIC -> {
        val sourceBuffer =
          photo.data.duplicate()

        val byteArray =
          ByteArray(
            sourceBuffer.remaining(),
          )

        sourceBuffer.get(
          byteArray,
        )

        val exifInfo =
          getExifInfo(
            byteArray,
          )

        val transform =
          getTransform(
            exifInfo,
          )

        decodeHeic(
          heicBytes = byteArray,
          transform = transform,
        )
      }
    }
  }

  private fun selectLargestPerson(
    detections: List<PoseDetection>,
  ): PoseDetection? {
    return detections.maxByOrNull { detection ->
      detection.box.width() *
              detection.box.height()
    }
  }

  private fun cropPersonBitmap(
    sourceBitmap: Bitmap,
    detection: PoseDetection,
  ): Bitmap {
    val box =
      detection.box

    val horizontalPadding =
      box.width() *
              PERSON_CROP_PADDING_RATIO

    val verticalPadding =
      box.height() *
              PERSON_CROP_PADDING_RATIO

    val left =
      (box.left - horizontalPadding)
        .toInt()
        .coerceIn(
          0,
          sourceBitmap.width - 1,
        )

    val top =
      (box.top - verticalPadding)
        .toInt()
        .coerceIn(
          0,
          sourceBitmap.height - 1,
        )

    val right =
      (box.right + horizontalPadding)
        .toInt()
        .coerceIn(
          left + 1,
          sourceBitmap.width,
        )

    val bottom =
      (box.bottom + verticalPadding)
        .toInt()
        .coerceIn(
          top + 1,
          sourceBitmap.height,
        )

    val croppedBitmap =
      Bitmap.createBitmap(
        sourceBitmap,
        left,
        top,
        right - left,
        bottom - top,
      )

    val copiedBitmap =
      croppedBitmap.copy(
        Bitmap.Config.ARGB_8888,
        true,
      ) ?: throw IllegalStateException(
        "Failed to copy person crop Bitmap",
      )

    if (
      croppedBitmap !== sourceBitmap &&
      !croppedBitmap.isRecycled
    ) {
      croppedBitmap.recycle()
    }

    return copiedBitmap
  }

  private fun savePersonCrop(
    bitmap: Bitmap,
  ): File {
    val context =
      getApplication<Application>()

    val baseDirectory =
      context.getExternalFilesDir(
        Environment.DIRECTORY_PICTURES,
      ) ?: context.filesDir

    val cropDirectory =
      File(
        baseDirectory,
        "person_crops_mosaicked",
      ).apply {
        if (!exists() && !mkdirs()) {
          throw IOException(
            "Failed to create crop directory: $absolutePath",
          )
        }
      }

    val outputFile =
      File(
        cropDirectory,
        "person_mosaic_${System.currentTimeMillis()}.jpg",
      )

    FileOutputStream(outputFile).use { outputStream ->
      val success =
        bitmap.compress(
          Bitmap.CompressFormat.JPEG,
          PERSON_CROP_JPEG_QUALITY,
          outputStream,
        )

      if (!success) {
        throw IOException(
          "Failed to compress person crop",
        )
      }
    }

    return outputFile
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
      ) ?: throw IOException(
        "Failed to decode HEIC photo",
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
    batchCropUploader.cancelAll()
    stopStream()
    faceMosaicProcessor.close()
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