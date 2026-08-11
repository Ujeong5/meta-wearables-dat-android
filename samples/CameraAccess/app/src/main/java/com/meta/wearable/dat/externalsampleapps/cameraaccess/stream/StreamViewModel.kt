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
import java.util.ArrayDeque
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlin.math.sqrt
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

    /*
     * 2fps 스트림에 맞춰 최대 500ms마다 최신 프레임에 YOLO를 실행한다.
     * 추론 중 들어온 프레임은 conflate/isYoloRunning으로 쌓지 않는다.
     */
    private const val YOLO_INTERVAL_MS = 500L

    /*
     * 실시간 촬영 후보와 고해상도 저장 후보에 동일하게 적용한다.
     * 어깨·골반·무릎·발목 중 각 관절 쌍에서 한 점 이상 보여야 통과한다.
     */
    private const val MIN_PERSON_CONFIDENCE = 0.45f
    private const val MIN_KEYPOINT_CONFIDENCE = 0.35f
    private const val MIN_BOX_WIDTH_RATIO = 0.05f
    private const val MIN_BOX_HEIGHT_RATIO = 0.20f

    // 사람 crop에 추가할 바깥 여백
    private const val PERSON_CROP_PADDING_RATIO = 0.08f
    private const val PERSON_CROP_JPEG_QUALITY = 95

    /*
     * 처리 중인 사진까지 포함해 메모리에 유지할 자동 고해상도 사진 수다.
     * 슬롯이 가득 차면 이미 촬영한 사진을 버리는 대신 새 촬영 요청만 잠시 보류한다.
     */
    private const val MAX_PENDING_AUTO_PHOTOS = 3
  }

  /*
   * 실시간 스트림과 고해상도 사진이 서로 YOLO 사용권을 기다리지 않도록
   * ONNX Runtime 세션을 두 개 분리한다.
   */
  private val realtimeYoloPoseModel =
    YoloPoseModel(getApplication())

  private val highResYoloPoseModel =
    YoloPoseModel(getApplication())

  /*
   * Face Landmarker로 눈·코·입 중심부만 모자이크한다.
   * 얼굴이 검출되지 않는 뒷모습에는 모자이크를 적용하지 않는다.
   */
  private val faceLandmarkMosaicProcessor =
    FaceLandmarkMosaicProcessor(getApplication())

  /*
   * 익명화된 사람 crop에서 OSNet 특징을 추출하고,
   * 이미 저장한 사람과 중복인지 판단한다.
   */
  private var osNetReIdentifier =
    OsNetReIdentifier(getApplication())

  private var hasStartedCollectionSession =
    false

  /*
   * 수집 세션/샘플 metadata는 작은 JSON sidecar로만 기록한다.
   * Android에서는 CSV나 SQLite를 쓰지 않는다.
   */
  private val collectionMetadataStore =
    CollectionMetadataStore(getApplication())

  private val phoneLocationProvider =
    PhoneLocationProvider(getApplication())

  private var activeCollectionSession: CollectionSession? =
    null

  /*
   * OSNet 자체의 중복 판정 로직은 건드리지 않는다.
   * 저장된 신규 인물 embedding만 이 목록에 복사해 두고
   * metadata용 max cosine similarity를 별도로 계산한다.
   */
  private val metadataEmbeddingGallery =
    mutableListOf<FloatArray>()

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
   * 수동 촬영 후 공유 화면으로 넘기는 작업들을 보관한다.
   * 자동 촬영 사진은 아래의 단일 순차 처리 큐에서 관리한다.
   */
  private val photoProcessingJobs =
    mutableSetOf<Job>()

  /*
   * 자동 촬영 사진 처리 대기열.
   *
   * 촬영은 이전 사진의 후처리와 병렬로 진행할 수 있지만,
   * YOLO → Face Landmarker → OSNet → 저장은 한 장씩 순서대로 실행한다.
   * 이렇게 해야 같은 사람이 연속 사진에서 동시에 신규로 판정되는 문제를 줄일 수 있다.
   */
  private val autoPhotoQueueLock =
    Any()

  private data class QueuedAutoPhoto(
    val bitmap: Bitmap,
    val capturedAtEpochMs: Long,
  )

  private val autoPhotoQueue =
    ArrayDeque<QueuedAutoPhoto>()

  private var autoPhotoProcessorJob: Job? =
    null

  /*
   * 현재 촬영 예약, 처리 대기, 처리 중인 자동 사진의 총합이다.
   * 이 값이 MAX_PENDING_AUTO_PHOTOS에 도달하면 새 촬영만 잠시 보류한다.
   */
  private val pendingAutoPhotoCount =
    AtomicInteger(0)

  /*
   * 자동 촬영 간격을 실제로 측정하기 위한 단조 증가 시각이다.
   * 코드에서 6초를 강제하지 않고, 요청·성공 간격을 Logcat에 기록한다.
   */
  private var lastAutoCaptureRequestAtMs =
    0L

  private var lastAutoCaptureSuccessAtMs =
    0L

  private val autoCaptureAttemptCount =
    AtomicInteger(0)

  private val autoCaptureSuccessCount =
    AtomicInteger(0)

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

  fun updatePlaceName(value: String) {
    if (_uiState.value.isCollectionSessionActive) {
      return
    }

    _uiState.update {
      it.copy(placeName = value)
    }
  }

  fun onLocationPermissionDenied() {
    _uiState.update {
      it.copy(
        isLocationLoading = false,
        locationStatusText =
          "정확한 위치 권한이 필요합니다. 권한을 허용한 뒤 다시 시도하세요.",
      )
    }
  }

  fun fetchCurrentGps() {
    if (_uiState.value.isCollectionSessionActive) {
      return
    }

    _uiState.update {
      it.copy(
        isLocationLoading = true,
        locationStatusText = "GPS 위치를 확인하는 중입니다...",
      )
    }

    viewModelScope.launch {
      val result =
        phoneLocationProvider.getCurrentGpsLocation()

      result
        .onSuccess { location ->
          _uiState.update {
            it.copy(
              gpsLatitude = location.latitude,
              gpsLongitude = location.longitude,
              gpsAccuracyM = location.accuracyMeters,
              isLocationLoading = false,
              locationStatusText =
                "GPS 위치를 저장했습니다. 정확도 ±${"%.1f".format(location.accuracyMeters)} m",
            )
          }
        }
        .onFailure { error ->
          _uiState.update {
            it.copy(
              isLocationLoading = false,
              locationStatusText =
                error.message ?: "GPS 위치를 가져오지 못했습니다.",
            )
          }
        }
    }
  }

  fun startCollectionSession() {
    if (_uiState.value.isCollectionSessionActive) {
      return
    }

    val state = _uiState.value
    val latitude = state.gpsLatitude
    val longitude = state.gpsLongitude
    val accuracy = state.gpsAccuracyM

    if (state.placeName.isBlank()) {
      _uiState.update {
        it.copy(locationStatusText = "장소명을 입력하세요.")
      }
      return
    }

    if (latitude == null || longitude == null || accuracy == null) {
      _uiState.update {
        it.copy(locationStatusText = "먼저 현재 GPS를 가져오세요.")
      }
      return
    }

    val session =
      try {
        collectionMetadataStore.createSession(
          placeName = state.placeName,
          location =
            LocationSnapshot(
              latitude = latitude,
              longitude = longitude,
              accuracyMeters = accuracy,
            ),
        )
      } catch (exception: Exception) {
        _uiState.update {
          it.copy(
            locationStatusText =
              "수집 세션을 만들지 못했습니다: " +
                      (exception.message ?: exception.javaClass.simpleName),
          )
        }
        return
      }

    if (hasStartedCollectionSession) {
      osNetReIdentifier.close()
      osNetReIdentifier =
        OsNetReIdentifier(getApplication())
    }

    hasStartedCollectionSession = true
    activeCollectionSession = session
    metadataEmbeddingGallery.clear()

    _uiState.update {
      it.copy(
        isCollectionSessionActive = true,
        batchId = session.batchId,
        placeName = session.placeName,
        gpsLatitude = session.gpsLatitude,
        gpsLongitude = session.gpsLongitude,
        gpsAccuracyM = session.gpsAccuracyM,
        locationStatusText = null,
        uploadStatusText = null,
        pendingUploadCount = 0,
      )
    }

    startStream()
  }

  private fun countPendingForActiveSession(): Int {
    val batchId = activeCollectionSession?.batchId
    return if (batchId == null) {
      0
    } else {
      batchCropUploader.countPendingFiles(batchId)
    }
  }

  fun startStream() {
    if (activeCollectionSession == null) {
      _uiState.update {
        it.copy(
          uploadStatusText = "장소와 GPS를 설정한 뒤 수집을 시작하세요.",
        )
      }
      return
    }
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
          countPendingForActiveSession(),
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
    cancelAutoPhotoProcessingQueue()

    isYoloRunning.set(false)
    isPhotoCaptureRunning.set(false)

    pendingAutoPhotoCount.set(0)

    lastYoloStartedAtMs = 0L
    lastAutoCaptureRequestAtMs = 0L
    lastAutoCaptureSuccessAtMs = 0L
    autoCaptureAttemptCount.set(0)
    autoCaptureSuccessCount.set(0)

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
                  frameRate = 2,
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
    cancelAutoPhotoProcessingQueue()

    isYoloRunning.set(false)
    isPhotoCaptureRunning.set(false)

    pendingAutoPhotoCount.set(0)

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

          /*
           * 자동 촬영 큐는 한 장씩 순서대로 처리한다.
           * 최대 3장만 유지하므로, 중지 시 큐 전체가 끝날 때까지 기다린다.
           */
          val autoProcessingCompleted =
            withTimeoutOrNull(180_000L) {
              autoPhotoProcessorJob?.join()
              true
            } ?: false

          if (!autoProcessingCompleted) {
            Log.w(
              TAG,
              "Automatic photo queue did not finish before upload timeout",
            )

            cancelAutoPhotoProcessingQueue()
          }

          // 수동 촬영 후처리 작업도 기다린다.
          val processingJobs =
            synchronized(photoProcessingJobs) {
              photoProcessingJobs.toList()
            }

          val manualProcessingCompleted =
            withTimeoutOrNull(30_000L) {
              processingJobs.joinAll()
              true
            } ?: false

          if (!manualProcessingCompleted) {
            Log.w(
              TAG,
              "Manual photo processing did not finish before upload timeout",
            )

            processingJobs.forEach { job ->
              job.cancel()
            }
            processingJobs.joinAll()
          }

          isPhotoCaptureRunning.set(false)
          pendingAutoPhotoCount.set(0)

          val currentSession =
            activeCollectionSession
              ?: throw IllegalStateException(
                "활성 수집 세션이 없습니다.",
              )

          activeCollectionSession =
            withContext(Dispatchers.IO) {
              collectionMetadataStore.finalizeSession(currentSession)
            }

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
                countPendingForActiveSession(),
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
                countPendingForActiveSession(),
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
      countPendingForActiveSession()

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

    val currentSession =
      activeCollectionSession

    if (currentSession == null) {
      _uiState.update {
        it.copy(
          isUploading = false,
          uploadStatusText = "활성 수집 세션이 없습니다.",
        )
      }
      return
    }

    val finalizedSession =
      if (currentSession.sessionEndTime == null) {
        withContext(Dispatchers.IO) {
          collectionMetadataStore.finalizeSession(currentSession)
        }.also { activeCollectionSession = it }
      } else {
        currentSession
      }

    val result =
      batchCropUploader.uploadPendingCrops(finalizedSession)

    val remainingCount =
      countPendingForActiveSession()

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
    lastAutoCaptureRequestAtMs = 0L
    lastAutoCaptureSuccessAtMs = 0L
    autoCaptureAttemptCount.set(0)
    autoCaptureSuccessCount.set(0)

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

    /*
     * 자동 사진은 처리 중인 사진까지 포함해 최대 3장만 유지한다.
     * 큐가 가득 차면 이미 촬영한 사진을 버리지 않고 새 촬영 요청만 보류한다.
     */
    val autoSlotReserved =
      if (isAutomatic) {
        tryReserveAutoPhotoSlot()
      } else {
        false
      }

    if (isAutomatic && !autoSlotReserved) {
      Log.d(
        TAG,
        "Automatic capture skipped: " +
                "photo queue is full " +
                "(${pendingAutoPhotoCount.get()}/$MAX_PENDING_AUTO_PHOTOS)",
      )
      return
    }

    /*
     * 실제 capturePhoto() 호출끼리만 겹치지 않게 막는다.
     * 이전 사진의 YOLO, Face Landmarker, OSNet, 저장이 실행 중이어도
     * 큐 슬롯이 남아 있으면 다음 사진 촬영을 시작할 수 있다.
     */
    if (
      !isPhotoCaptureRunning.compareAndSet(
        false,
        true,
      )
    ) {
      if (autoSlotReserved) {
        releaseAutoPhotoSlot()
      }

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

      if (autoSlotReserved) {
        releaseAutoPhotoSlot()
      }
      return
    }

    val requestStartedAtMs =
      SystemClock.elapsedRealtime()

    val attemptNumber =
      if (isAutomatic) {
        autoCaptureAttemptCount.incrementAndGet()
      } else {
        0
      }

    val requestToRequestMs =
      if (
        isAutomatic &&
        lastAutoCaptureRequestAtMs > 0L
      ) {
        requestStartedAtMs -
                lastAutoCaptureRequestAtMs
      } else {
        -1L
      }

    if (isAutomatic) {
      lastAutoCaptureRequestAtMs =
        requestStartedAtMs
    }

    Log.d(
      TAG,
      if (isAutomatic) {
        "AUTO_CAPTURE_REQUEST: " +
                "attempt=$attemptNumber, " +
                "requestToRequestMs=$requestToRequestMs, " +
                "pending=${pendingAutoPhotoCount.get()}/$MAX_PENDING_AUTO_PHOTOS"
      } else {
        "Manual photo capture requested"
      },
    )

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

                if (autoSlotReserved) {
                  releaseAutoPhotoSlot()
                }
                return@onSuccess
              }

            val capturedAtEpochMs =
              System.currentTimeMillis()

            val successAtMs =
              SystemClock.elapsedRealtime()

            val requestToSuccessMs =
              successAtMs -
                      requestStartedAtMs

            val successToSuccessMs =
              if (
                isAutomatic &&
                lastAutoCaptureSuccessAtMs > 0L
              ) {
                successAtMs -
                        lastAutoCaptureSuccessAtMs
              } else {
                -1L
              }

            val successNumber =
              if (isAutomatic) {
                autoCaptureSuccessCount.incrementAndGet()
              } else {
                0
              }

            if (isAutomatic) {
              lastAutoCaptureSuccessAtMs =
                successAtMs
            }

            Log.d(
              TAG,
              if (isAutomatic) {
                "AUTO_CAPTURE_SUCCESS: " +
                        "success=$successNumber, " +
                        "attempt=$attemptNumber, " +
                        "requestToSuccessMs=$requestToSuccessMs, " +
                        "successToSuccessMs=$successToSuccessMs, " +
                        "size=${capturedBitmap.width}x${capturedBitmap.height}, " +
                        "bitmapBytes=${capturedBitmap.allocationByteCount}, " +
                        "pending=${pendingAutoPhotoCount.get()}/$MAX_PENDING_AUTO_PHOTOS"
              } else {
                "Manual photo capture successful: " +
                        "${capturedBitmap.width}x${capturedBitmap.height}"
              },
            )

            /*
             * PhotoData를 독립적인 Bitmap으로 복사한 순간
             * 촬영 API 잠금을 해제한다.
             * 이후 고해상도 처리는 자동 사진 큐에서 별도로 진행된다.
             */
            finishPhotoCapture()

            if (isAutomatic) {
              enqueueAutoCapturedPhoto(
                capturedBitmap = capturedBitmap,
                capturedAtEpochMs = capturedAtEpochMs,
              )
            } else {
              launchManualCapturedPhotoProcessing(
                capturedBitmap = capturedBitmap,
              )
            }
          }
          .onFailure { error, _ ->
            val failedAtMs =
              SystemClock.elapsedRealtime()

            Log.e(
              TAG,
              if (isAutomatic) {
                "AUTO_CAPTURE_FAILURE: " +
                        "attempt=$attemptNumber, " +
                        "requestToFailureMs=${failedAtMs - requestStartedAtMs}, " +
                        "error=${error.description}"
              } else {
                "Photo capture failed: ${error.description}"
              },
            )

            finishPhotoCapture()

            if (autoSlotReserved) {
              releaseAutoPhotoSlot()
            }
          }
      }
  }

  private fun finishPhotoCapture() {
    _uiState.update {
      it.copy(isCapturing = false)
    }

    isPhotoCaptureRunning.set(false)
  }

  /*
   * 자동 촬영 사진을 메모리 큐에 추가한다.
   * 슬롯은 촬영 요청 전에 이미 예약되었으므로 여기서는 사진을 버리지 않는다.
   */
  private fun enqueueAutoCapturedPhoto(
    capturedBitmap: Bitmap,
    capturedAtEpochMs: Long,
  ) {
    val queueSize =
      synchronized(autoPhotoQueueLock) {
        autoPhotoQueue.addLast(
          QueuedAutoPhoto(
            bitmap = capturedBitmap,
            capturedAtEpochMs = capturedAtEpochMs,
          ),
        )

        autoPhotoQueue.size
      }

    Log.d(
      TAG,
      "Automatic photo queued: " +
              "queueSize=$queueSize, " +
              "pending=${pendingAutoPhotoCount.get()}/$MAX_PENDING_AUTO_PHOTOS",
    )

    startAutoPhotoProcessorIfNeeded()
  }

  /*
   * 자동 사진은 반드시 한 장씩 순서대로 처리한다.
   * 촬영은 이 작업과 병렬로 계속 진행할 수 있다.
   */
  private fun startAutoPhotoProcessorIfNeeded() {
    val jobToStart: Job?

    synchronized(autoPhotoQueueLock) {
      if (
        autoPhotoProcessorJob?.isActive ==
        true
      ) {
        return
      }

      if (autoPhotoQueue.isEmpty()) {
        return
      }

      val newJob =
        viewModelScope.launch(
          context = Dispatchers.Default,
          start = CoroutineStart.LAZY,
        ) {
          while (true) {
            val nextPhoto =
              synchronized(autoPhotoQueueLock) {
                if (autoPhotoQueue.isEmpty()) {
                  null
                } else {
                  autoPhotoQueue.removeFirst()
                }
              } ?: break

            try {
              processAutoCapturedPhoto(
                fullPhoto = nextPhoto.bitmap,
                captureTimestampMs = nextPhoto.capturedAtEpochMs,
              )
            } catch (
              exception: CancellationException
            ) {
              if (!nextPhoto.bitmap.isRecycled) {
                nextPhoto.bitmap.recycle()
              }

              throw exception
            } catch (
              exception: Exception
            ) {
              Log.e(
                TAG,
                "Automatic photo processing failed",
                exception,
              )

              if (!nextPhoto.bitmap.isRecycled) {
                nextPhoto.bitmap.recycle()
              }
            } finally {
              releaseAutoPhotoSlot()

              Log.d(
                TAG,
                "Automatic photo slot released: " +
                        "pending=${pendingAutoPhotoCount.get()}/$MAX_PENDING_AUTO_PHOTOS",
              )
            }
          }
        }

      autoPhotoProcessorJob =
        newJob

      newJob.invokeOnCompletion {
        var shouldRestart =
          false

        synchronized(autoPhotoQueueLock) {
          if (
            autoPhotoProcessorJob ===
            newJob
          ) {
            autoPhotoProcessorJob =
              null
          }

          shouldRestart =
            autoPhotoQueue.isNotEmpty()
        }

        if (shouldRestart) {
          startAutoPhotoProcessorIfNeeded()
        }
      }

      jobToStart =
        newJob
    }

    jobToStart?.start()
  }

  /*
   * 수동 촬영은 기존 공유 다이얼로그 동작을 유지한다.
   */
  private fun launchManualCapturedPhotoProcessing(
    capturedBitmap: Bitmap,
  ) {
    val processingJob =
      viewModelScope.launch(
        context = Dispatchers.Default,
        start = CoroutineStart.LAZY,
      ) {
        try {
          processManualCapturedPhoto(
            capturedPhoto = capturedBitmap,
          )
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
            "Manual photo processing failed",
            exception,
          )

          if (!capturedBitmap.isRecycled) {
            capturedBitmap.recycle()
          }
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

  private fun tryReserveAutoPhotoSlot(): Boolean {
    while (true) {
      val currentCount =
        pendingAutoPhotoCount.get()

      if (
        currentCount >=
        MAX_PENDING_AUTO_PHOTOS
      ) {
        return false
      }

      if (
        pendingAutoPhotoCount.compareAndSet(
          currentCount,
          currentCount + 1,
        )
      ) {
        return true
      }
    }
  }

  private fun releaseAutoPhotoSlot() {
    decrementAtomicNonNegative(
      pendingAutoPhotoCount,
    )
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


  private fun cancelAutoPhotoProcessingQueue() {
    autoPhotoProcessorJob?.cancel()
    autoPhotoProcessorJob = null

    val queuedPhotos =
      synchronized(autoPhotoQueueLock) {
        buildList {
          while (autoPhotoQueue.isNotEmpty()) {
            add(
              autoPhotoQueue.removeFirst(),
            )
          }
        }
      }

    queuedPhotos.forEach { queuedPhoto ->
      if (!queuedPhoto.bitmap.isRecycled) {
        queuedPhoto.bitmap.recycle()
      }
    }

    pendingAutoPhotoCount.set(0)

    Log.d(
      TAG,
      "Automatic photo processing queue cancelled and cleared",
    )
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
        val startedAtMs =
          SystemClock.elapsedRealtime()

        try {
          /*
           * 고해상도 사진은 별도의 YOLO 세션을 사용하므로,
           * 실시간 검출이 사진 후처리 때문에 대기하거나 굶지 않는다.
           */
          val rawDetections =
            realtimeYoloPoseModel.detect(
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

          val elapsedMs =
            SystemClock.elapsedRealtime() -
                    startedAtMs

          Log.d(
            TAG,
            "REALTIME_YOLO_RESULT: " +
                    "elapsedMs=$elapsedMs, " +
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
          inferenceBitmap.recycle()
          isYoloRunning.set(false)
        }
      }
  }

  /*
   * 첫 번째 OSNet은 사용하지 않는다.
   * 실시간 스트림에서 어깨·골반·무릎·발목 조건까지 통과한 사람이
   * 한 명 이상이면 고해상도 사진 촬영을 요청한다.
   */
  private fun updateAutoCaptureState(
    validDetections: List<PoseDetection>,
  ) {
    if (validDetections.isNotEmpty()) {
      requestAutoPhotoCapture()
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

      // COCO Pose index: 어깨 5/6, 골반 11/12, 무릎 13/14, 발목 15/16.
      fun keypointConfidence(index: Int): Float {
        return detection.keypoints
          .getOrNull(index)
          ?.confidence
          ?: 0f
      }

      val shoulderVisible =
        keypointConfidence(5) >= MIN_KEYPOINT_CONFIDENCE ||
                keypointConfidence(6) >= MIN_KEYPOINT_CONFIDENCE

      val hipVisible =
        keypointConfidence(11) >= MIN_KEYPOINT_CONFIDENCE ||
                keypointConfidence(12) >= MIN_KEYPOINT_CONFIDENCE

      val kneeVisible =
        keypointConfidence(13) >= MIN_KEYPOINT_CONFIDENCE ||
                keypointConfidence(14) >= MIN_KEYPOINT_CONFIDENCE

      val ankleVisible =
        keypointConfidence(15) >= MIN_KEYPOINT_CONFIDENCE ||
                keypointConfidence(16) >= MIN_KEYPOINT_CONFIDENCE

      shoulderVisible &&
              hipVisible &&
              kneeVisible &&
              ankleVisible
    }
  }

  /*
   * 휴대폰 미리보기 Bitmap에 전신 조건을 통과한 사람의 Bounding Box만 그린다.
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
   * 자동 촬영된 고해상도 사진 처리:
   * 1. 사진 전체에 YOLO11n-pose를 다시 실행한다.
   * 2. 전신·크기 조건을 통과한 모든 사람을 각각 crop한다.
   * 3. Face Landmarker로 눈·코·입만 모자이크한다.
   * 4. 익명화된 crop에서 OSNet embedding을 추출한다.
   * 5. 이미 저장한 사람은 제외하고 신규 인물만 JPEG로 저장한다.
   * 6. 저장 성공 후에만 OSNet gallery에 등록한다.
   */
  private suspend fun processAutoCapturedPhoto(
    fullPhoto: Bitmap,
    captureTimestampMs: Long,
  ) {
    try {
      Log.d(
        TAG,
        "High-resolution photo processing started: " +
                "${fullPhoto.width}x${fullPhoto.height}",
      )

      /*
       * 고해상도 사진은 실시간 스트림과 별도의 YOLO 세션으로 처리한다.
       * 따라서 사진 후처리 중에도 실시간 박스와 촬영 트리거가 계속 갱신된다.
       */
      val rawDetections =
        highResYoloPoseModel.detect(
          fullPhoto,
        )

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

      if (validDetections.isEmpty()) {
        Log.d(
          TAG,
          "No valid full-body person remained in high-resolution photo",
        )
        return
      }

      val captureId =
        captureTimestampMs

      val collectionSession =
        activeCollectionSession
          ?: throw IllegalStateException(
            "활성 수집 세션이 없습니다.",
          )

      var savedCount = 0
      var duplicateCount = 0
      var failedCount = 0

      validDetections.forEachIndexed { personIndex, detection ->

        val personCropResult =
          try {
            cropPersonBitmap(
              sourceBitmap = fullPhoto,
              detection = detection,
            )
          } catch (exception: Exception) {
            failedCount += 1

            Log.e(
              TAG,
              "Failed to crop person index=$personIndex",
              exception,
            )
            return@forEachIndexed
          }

        val personCrop =
          personCropResult.bitmap

        try {
          val mosaicResult =
            faceLandmarkMosaicProcessor.mosaicFaces(
              bitmap = personCrop,
              poseKeypoints =
                personCropResult.poseKeypoints,
            )

          Log.d(
            TAG,
            "Face landmark mosaic: " +
                    "personIndex=$personIndex, " +
                    "detectedFaces=${mosaicResult.detectedFaceCount}, " +
                    "mosaickedFaces=${mosaicResult.mosaickedFaceCount}, " +
                    "mosaickedRegions=${mosaicResult.mosaickedRegionCount}",
          )

          /*
           * OSNet은 익명화 뒤에 실행한다.
           * 얼굴이 보인 경우 눈·코·입 정보가 제거된 crop을 사용하므로
           * 착장 중심 중복 판정이라는 목적에 더 잘 맞는다.
           */
          val embedding =
            osNetReIdentifier.extractEmbedding(
              personBitmap = personCrop,
            )

          val osnetMaxSimilarity =
            computeMaxCosineSimilarity(embedding)

          val duplicateMatch =
            osNetReIdentifier.findDuplicate(
              embedding = embedding,
            )

          if (duplicateMatch != null) {
            duplicateCount += 1

            Log.d(
              TAG,
              "Duplicate person skipped: " +
                      "personIndex=$personIndex, " +
                      "matchedPersonId=${duplicateMatch.personId}, " +
                      "similarity=${duplicateMatch.similarity}, " +
                      "file=${duplicateMatch.savedFileName}",
            )
            return@forEachIndexed
          }

          val savedFile =
            withContext(Dispatchers.IO) {
              savePersonCrop(
                bitmap = personCrop,
                captureId = captureId,
                personIndex = personIndex,
                batchId = collectionSession.batchId,
              )
            }

          try {
            val sampleMetadata =
              buildSampleMetadata(
                session = collectionSession,
                imageFilename = savedFile.name,
                captureTimestampMs = captureId,
                sourceWidth = fullPhoto.width,
                sourceHeight = fullPhoto.height,
                cropWidth = personCrop.width,
                cropHeight = personCrop.height,
                detection = detection,
                osnetMaxSimilarity = osnetMaxSimilarity,
              )

            withContext(Dispatchers.IO) {
              collectionMetadataStore.saveSampleMetadata(sampleMetadata)
            }
          } catch (exception: Exception) {
            withContext(Dispatchers.IO) {
              savedFile.delete()
            }
            throw exception
          }

          val personId =
            osNetReIdentifier.registerSavedPerson(
              embedding = embedding,
              savedFileName = savedFile.name,
            )

          metadataEmbeddingGallery.add(
            embedding.copyOf(),
          )

          savedCount += 1

          Log.d(
            TAG,
            "New anonymized person crop saved: " +
                    "personIndex=$personIndex, " +
                    "personId=$personId, " +
                    "path=${savedFile.absolutePath}, " +
                    "size=${personCrop.width}x${personCrop.height}",
          )
        } catch (exception: Exception) {
          failedCount += 1

          Log.e(
            TAG,
            "Person processing failed: index=$personIndex",
            exception,
          )
        } finally {
          if (!personCrop.isRecycled) {
            personCrop.recycle()
          }
        }
      }

      _uiState.update {
        it.copy(
          pendingUploadCount =
            countPendingForActiveSession(),
        )
      }

      Log.d(
        TAG,
        "High-resolution photo processing completed: " +
                "valid=${validDetections.size}, " +
                "saved=$savedCount, " +
                "duplicates=$duplicateCount, " +
                "failed=$failedCount, " +
                "gallery=${osNetReIdentifier.getRegisteredPersonCount()}",
      )
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


  private data class PersonCropResult(
    val bitmap: Bitmap,
    val poseKeypoints: List<PoseKeypoint>,
  )

  private fun cropPersonBitmap(
    sourceBitmap: Bitmap,
    detection: PoseDetection,
  ): PersonCropResult {
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

    /*
     * Face Landmarker가 모자나 측면 얼굴을 놓칠 때 사용할 수 있도록
     * YOLO pose 얼굴 키포인트를 사람 crop 좌표로 변환한다.
     */
    val mappedKeypoints =
      detection.keypoints.map { keypoint ->
        PoseKeypoint(
          x =
            (keypoint.x - left)
              .coerceIn(
                0f,
                copiedBitmap.width.toFloat(),
              ),
          y =
            (keypoint.y - top)
              .coerceIn(
                0f,
                copiedBitmap.height.toFloat(),
              ),
          confidence =
            keypoint.confidence,
        )
      }

    return PersonCropResult(
      bitmap = copiedBitmap,
      poseKeypoints = mappedKeypoints,
    )
  }

  private fun savePersonCrop(
    bitmap: Bitmap,
    captureId: Long,
    personIndex: Int,
    batchId: String,
  ): File {
    val context =
      getApplication<Application>()

    val baseDirectory =
      context.getExternalFilesDir(
        Environment.DIRECTORY_PICTURES,
      ) ?: context.filesDir

    val cropDirectory =
      File(
        File(
          baseDirectory,
          "person_crops_mosaicked",
        ),
        batchId,
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
        "capture_${captureId}_person_${personIndex.toString().padStart(2, '0')}.jpg",
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

  private fun buildSampleMetadata(
    session: CollectionSession,
    imageFilename: String,
    captureTimestampMs: Long,
    sourceWidth: Int,
    sourceHeight: Int,
    cropWidth: Int,
    cropHeight: Int,
    detection: PoseDetection,
    osnetMaxSimilarity: Float?,
  ): SampleMetadata {
    val sourceWidthF = sourceWidth.toFloat().coerceAtLeast(1f)
    val sourceHeightF = sourceHeight.toFloat().coerceAtLeast(1f)

    val x1 =
      (detection.box.left / sourceWidthF).coerceIn(0f, 1f)
    val y1 =
      (detection.box.top / sourceHeightF).coerceIn(0f, 1f)
    val x2 =
      (detection.box.right / sourceWidthF).coerceIn(0f, 1f)
    val y2 =
      (detection.box.bottom / sourceHeightF).coerceIn(0f, 1f)

    val areaRatio =
      ((detection.box.width().coerceAtLeast(0f) *
              detection.box.height().coerceAtLeast(0f)) /
              (sourceWidthF * sourceHeightF))
        .coerceIn(0f, 1f)

    fun confidence(index: Int): Float =
      detection.keypoints
        .getOrNull(index)
        ?.confidence
        ?: 0f

    /*
     * 좌/우를 각각 저장하지 않고 관절 그룹별 가장 잘 보이는 점 하나를 사용한다.
     * shoulders / hips / knees / ankles의 4개 품질값을 평균·최솟값으로 압축한다.
     */
    val poseGroupConfidences =
      listOf(
        max(confidence(5), confidence(6)),
        max(confidence(11), confidence(12)),
        max(confidence(13), confidence(14)),
        max(confidence(15), confidence(16)),
      )

    val poseMean =
      poseGroupConfidences.average().toFloat()

    val poseMin =
      poseGroupConfidences.minOrNull() ?: 0f

    return SampleMetadata(
      batchId = session.batchId,
      sampleId =
        collectionMetadataStore.nextSampleId(session.batchId),
      imageFilename = imageFilename,
      captureTimestamp =
        collectionMetadataStore.timestampFromEpochMillis(captureTimestampMs),
      sourceWidth = sourceWidth,
      sourceHeight = sourceHeight,
      cropWidth = cropWidth,
      cropHeight = cropHeight,
      yoloConfidence = detection.confidence,
      bboxX1Norm = x1,
      bboxY1Norm = y1,
      bboxX2Norm = x2,
      bboxY2Norm = y2,
      bboxAreaRatio = areaRatio,
      poseMeanConfidence = poseMean,
      poseMinConfidence = poseMin,
      osnetMaxSimilarity = osnetMaxSimilarity,
    )
  }

  private fun computeMaxCosineSimilarity(
    embedding: FloatArray,
  ): Float? {
    if (metadataEmbeddingGallery.isEmpty()) {
      return null
    }

    return metadataEmbeddingGallery
      .maxOfOrNull { previous ->
        cosineSimilarity(embedding, previous)
      }
  }

  private fun cosineSimilarity(
    first: FloatArray,
    second: FloatArray,
  ): Float {
    val size = minOf(first.size, second.size)
    if (size == 0) {
      return 0f
    }

    var dot = 0.0
    var firstNorm = 0.0
    var secondNorm = 0.0

    for (index in 0 until size) {
      val a = first[index].toDouble()
      val b = second[index].toDouble()
      dot += a * b
      firstNorm += a * a
      secondNorm += b * b
    }

    val denominator =
      sqrt(firstNorm) * sqrt(secondNorm)

    if (denominator <= 0.0) {
      return 0f
    }

    return (dot / denominator)
      .toFloat()
      .coerceIn(-1f, 1f)
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
    faceLandmarkMosaicProcessor.close()
    osNetReIdentifier.close()
    realtimeYoloPoseModel.close()
    highResYoloPoseModel.close()
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