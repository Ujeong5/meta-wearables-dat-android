package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.content.Context
import android.os.Environment
import android.util.Log
import java.io.File
import java.io.IOException
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject

/**
 * person_crops_mosaicked 폴더의 모든 JPEG를 하나의 multipart/form-data 요청으로 전송한다.
 *
 * 서버가 모든 파일 저장 성공을 확인한 경우에만 휴대폰 파일을
 * person_crops_uploaded/<batchId> 폴더로 옮긴다.
 * 요청이 실패하면 원본 파일은 그대로 남아 재전송할 수 있다.
 */
class BatchCropUploader(
  private val context: Context,
  private val serverBaseUrl: String,
) {

  companion object {
    private const val TAG = "CameraAccess:BatchUploader"
    private const val PENDING_FOLDER_NAME = "person_crops_mosaicked"
    private const val UPLOADED_FOLDER_NAME = "person_crops_uploaded"
  }

  private val jpegMediaType =
    "image/jpeg".toMediaType()

  private val client =
    OkHttpClient.Builder()
      .connectTimeout(20, TimeUnit.SECONDS)
      .writeTimeout(10, TimeUnit.MINUTES)
      .readTimeout(10, TimeUnit.MINUTES)
      .retryOnConnectionFailure(false)
      .build()

  fun countPendingFiles(): Int {
    return getPendingFiles().size
  }

  suspend fun uploadPendingCrops(): BatchUploadResult =
    withContext(Dispatchers.IO) {
      val files =
        getPendingFiles()

      if (files.isEmpty()) {
        return@withContext BatchUploadResult(
          success = true,
          requestedCount = 0,
          transferredCount = 0,
          batchId = null,
          message = "전송할 저장 사진이 없습니다.",
        )
      }

      val batchId =
        createDeterministicBatchId(files)

      val multipartBuilder =
        MultipartBody.Builder()
          .setType(MultipartBody.FORM)
          .addFormDataPart(
            "batch_id",
            batchId,
          )

      files.forEach { file ->
        multipartBuilder.addFormDataPart(
          "files",
          file.name,
          file.asRequestBody(jpegMediaType),
        )
      }

      val request =
        Request.Builder()
          .url(
            "${serverBaseUrl.trimEnd('/')}/upload-batch",
          )
          .post(multipartBuilder.build())
          .build()

      Log.d(
        TAG,
        "Starting batch upload: " +
          "batchId=$batchId, files=${files.size}, " +
          "bytes=${files.sumOf { it.length() }}",
      )

      try {
        client.newCall(request).execute().use { response ->
          val responseText =
            response.body?.string().orEmpty()

          if (!response.isSuccessful) {
            return@withContext BatchUploadResult(
              success = false,
              requestedCount = files.size,
              transferredCount = 0,
              batchId = batchId,
              message =
                "서버가 HTTP ${response.code}로 응답했습니다. " +
                  responseText.take(300),
            )
          }

          val responseJson =
            try {
              JSONObject(responseText)
            } catch (exception: Exception) {
              return@withContext BatchUploadResult(
                success = false,
                requestedCount = files.size,
                transferredCount = 0,
                batchId = batchId,
                message = "서버 응답 JSON을 읽지 못했습니다.",
              )
            }

          val ok =
            responseJson.optBoolean("ok", false)

          val responseBatchId =
            responseJson.optString("batch_id", "")

          val savedCount =
            responseJson.optInt("saved_count", -1)

          val savedFilesArray =
            responseJson.optJSONArray("saved_files")

          val savedFileNames =
            buildSet {
              if (savedFilesArray != null) {
                for (index in 0 until savedFilesArray.length()) {
                  add(savedFilesArray.optString(index))
                }
              }
            }

          val expectedFileNames =
            files.map { it.name }.toSet()

          if (
            !ok ||
            responseBatchId != batchId ||
            savedCount != files.size ||
            savedFileNames != expectedFileNames
          ) {
            return@withContext BatchUploadResult(
              success = false,
              requestedCount = files.size,
              transferredCount = savedCount.coerceAtLeast(0),
              batchId = batchId,
              message =
                "서버의 저장 확인 내용이 휴대폰 파일 목록과 일치하지 않습니다.",
            )
          }

          moveUploadedFiles(
            files = files,
            batchId = batchId,
          )

          Log.d(
            TAG,
            "Batch upload completed: " +
              "batchId=$batchId, files=${files.size}",
          )

          BatchUploadResult(
            success = true,
            requestedCount = files.size,
            transferredCount = files.size,
            batchId = batchId,
            message = "사진 ${files.size}장을 데스크톱으로 전송했습니다.",
          )
        }
      } catch (exception: IOException) {
        Log.e(
          TAG,
          "Batch upload failed",
          exception,
        )

        BatchUploadResult(
          success = false,
          requestedCount = files.size,
          transferredCount = 0,
          batchId = batchId,
          message =
            "데스크톱 서버에 연결하지 못했습니다: " +
              (exception.message ?: exception.javaClass.simpleName),
        )
      }
    }

  fun cancelAll() {
    client.dispatcher.cancelAll()
  }

  private fun getPendingFiles(): List<File> {
    val directory =
      File(
        getPicturesBaseDirectory(),
        PENDING_FOLDER_NAME,
      )

    if (!directory.exists()) {
      return emptyList()
    }

    return directory
      .listFiles { file ->
        file.isFile &&
          file.extension.lowercase() in setOf("jpg", "jpeg")
      }
      ?.sortedBy { it.name }
      .orEmpty()
  }

  private fun getPicturesBaseDirectory(): File {
    return context.getExternalFilesDir(
      Environment.DIRECTORY_PICTURES,
    ) ?: context.filesDir
  }

  private fun createDeterministicBatchId(
    files: List<File>,
  ): String {
    val digest =
      MessageDigest.getInstance("SHA-256")

    files.forEach { file ->
      val description =
        "${file.name}:${file.length()}\n"

      digest.update(
        description.toByteArray(Charsets.UTF_8),
      )
    }

    return digest
      .digest()
      .joinToString(separator = "") { byte ->
        "%02x".format(byte)
      }
      .take(24)
  }

  private fun moveUploadedFiles(
    files: List<File>,
    batchId: String,
  ) {
    val uploadedDirectory =
      File(
        File(
          getPicturesBaseDirectory(),
          UPLOADED_FOLDER_NAME,
        ),
        batchId,
      )

    if (
      !uploadedDirectory.exists() &&
      !uploadedDirectory.mkdirs()
    ) {
      throw IOException(
        "전송 완료 폴더를 만들지 못했습니다: " +
          uploadedDirectory.absolutePath,
      )
    }

    files.forEach { sourceFile ->
      val targetFile =
        File(
          uploadedDirectory,
          sourceFile.name,
        )

      if (targetFile.exists() && !targetFile.delete()) {
        throw IOException(
          "기존 전송 완료 파일을 교체하지 못했습니다: " +
            targetFile.absolutePath,
        )
      }

      if (!sourceFile.renameTo(targetFile)) {
        sourceFile.copyTo(
          target = targetFile,
          overwrite = true,
        )

        if (!sourceFile.delete()) {
          targetFile.delete()
          throw IOException(
            "원본 파일을 전송 완료 폴더로 이동하지 못했습니다: " +
              sourceFile.absolutePath,
          )
        }
      }
    }
  }
}

data class BatchUploadResult(
  val success: Boolean,
  val requestedCount: Int,
  val transferredCount: Int,
  val batchId: String?,
  val message: String,
)
