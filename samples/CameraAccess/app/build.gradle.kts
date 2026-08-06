import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.jetbrains.kotlin.android)
  alias(libs.plugins.compose.compiler)
}

android {
  namespace =
    "com.meta.wearable.dat.externalsampleapps.cameraaccess"

  compileSdk = 36

  buildFeatures {
    buildConfig = true
  }

  defaultConfig {
    applicationId =
      "com.meta.wearable.dat.externalsampleapps.cameraaccess"

    minSdk = 31
    targetSdk = 36

    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner =
      "androidx.test.runner.AndroidJUnitRunner"

    /*
     * Meta Wearables Device Access Toolkit 설정.
     *
     * 현재 프로젝트에서 이미 정상 연결되고 있다면
     * 기존 값을 그대로 유지한다.
     */
    manifestPlaceholders["mwdat_application_id"] = "0"
    manifestPlaceholders["mwdat_client_token"] = "0"
  }

  buildTypes {
    release {
      isMinifyEnabled = false

      proguardFiles(
        getDefaultProguardFile(
          "proguard-android-optimize.txt",
        ),
        "proguard-rules.pro",
      )

      /*
       * 현재 개발·재현 단계에서는 기존 샘플과 동일하게
       * debug signingConfig를 사용한다.
       */
      signingConfig =
        signingConfigs.getByName("debug")
    }
  }

  compileOptions {
    sourceCompatibility =
      JavaVersion.VERSION_17

    targetCompatibility =
      JavaVersion.VERSION_17
  }

  /*
   * ONNX Runtime과 MediaPipe가 assets의 모델 파일을
   * 압축되지 않은 상태로 읽을 수 있도록 한다.
   */
  androidResources {
    noCompress +=
      listOf(
        "onnx",
        "task",
        "tflite",
      )
  }

  packaging {
    resources {
      excludes +=
        "/META-INF/{AL2.0,LGPL2.1}"
    }
  }

  signingConfigs {
    getByName("debug") {
      storeFile =
        file("sample.keystore")

      storePassword = "sample"
      keyAlias = "sample"
      keyPassword = "sample"
    }
  }
}

kotlin {
  compilerOptions {
    jvmTarget =
      JvmTarget.JVM_17
  }
}

dependencies {
  /*
   * Android / Compose
   */
  implementation(
    libs.androidx.activity.compose,
  )

  implementation(
    platform(libs.androidx.compose.bom),
  )

  implementation(
    libs.androidx.exifinterface,
  )

  implementation(
    libs.androidx.lifecycle.runtime.compose,
  )

  implementation(
    libs.androidx.lifecycle.viewmodel.compose,
  )

  implementation(
    libs.androidx.material.icons.extended,
  )

  implementation(
    libs.androidx.material3,
  )

  implementation(
    libs.kotlinx.collections.immutable,
  )

  /*
   * Meta Wearables DAT
   */
  implementation(
    libs.mwdat.core,
  )

  implementation(
    libs.mwdat.camera,
  )

  /*
   * 현재 프로젝트에 MockDeviceKit 화면과 패키지가 있으므로
   * 당장은 제거하지 않는다.
   *
   * 실제 기기 전용 앱으로 정리할 때 관련 화면과 함께 제거한다.
   */
  implementation(
    libs.mwdat.mockdevice,
  )

  /*
   * YOLO11n-pose와 OSNet이 공용으로 사용하는
   * ONNX Runtime Android.
   */
  implementation(
    "com.microsoft.onnxruntime:onnxruntime-android:1.27.0",
  )

  /*
   * Face Landmarker가 사용하는 MediaPipe Tasks Vision.
   */
  implementation(
    "com.google.mediapipe:tasks-vision:0.10.35",
  )

  /*
   * 저장된 JPEG crop을 multipart/form-data 형식으로
   * 데스크톱 서버에 전송한다.
   */
  implementation(
    "com.squareup.okhttp3:okhttp:4.12.0",
  )

  /*
   * Android 테스트
   */
  androidTestImplementation(
    libs.androidx.ui.test.junit4,
  )

  androidTestImplementation(
    libs.androidx.test.uiautomator,
  )

  androidTestImplementation(
    libs.androidx.test.rules,
  )
}