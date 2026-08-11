package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.content.Context
import android.os.Environment
import android.util.Log
import java.io.File
import java.io.IOException
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
 * 한 수집 세션(batch)의 모자이크 crop과 metadata를 한 번에 전송한다.
 *
 * Android에서는 CSV를 쓰지 않는다.
 * 각 crop의 작은 JSON sidecar만 저장하고, CSV/SQLite는 Windows가 만든다.
 */
class BatchCropUploader(
  private val context: Context,
  private val serverBaseUrl: String,
) {

  companion object {
    private const val TAG = "CameraAccess:BatchUploader"
    private const val PENDING_FOLDER_NAME = "person_crops_mosaicked"
    private const val UPLOADED_FOLDER_NAME = "person_crops_uploaded"
    private const val METADATA_FOLDER_NAME = "person_crop_metadata"
    private const val UPLOADED_METADATA_FOLDER_NAME = "person_crop_metadata_uploaded"
  }

  private val metadataStore =
    CollectionMetadataStore(context)

  private val jpegMediaType =
    "image/jpeg".toMediaType()

  private val client =
    OkHttpClient.Builder()
      .connectTimeout(20, TimeUnit.SECONDS)
      .writeTimeout(10, TimeUnit.MINUTES)
      .readTimeout(10, TimeUnit.MINUTES)
      .retryOnConnectionFailure(false)
      .build()

  fun countPendingFiles(batchId: String? = null): Int {
    return if (batchId != null) {
      getPendingFiles(batchId).size
    } else {
      val root =
        File(
          getPicturesBaseDirectory(),
          PENDING_FOLDER_NAME,
        )

      if (!root.exists()) {
        0
      } else {
        root.walkTopDown().count { file ->
          file.isFile && file.extension.lowercase() in setOf("jpg", "jpeg")
        }
      }
    }
  }

  suspend fun uploadPendingCrops(
    session: CollectionSession,
  ): BatchUploadResult =
    withContext(Dispatchers.IO) {
      val files =
        getPendingFiles(session.batchId)

      if (files.isEmpty()) {
        return@withContext BatchUploadResult(
          success = true,
          requestedCount = 0,
          transferredCount = 0,
          batchId = session.batchId,
          message = "전송할 저장 사진이 없습니다.",
        )
      }

      val samples =
        metadataStore.loadSampleMetadata(session.batchId)

      val sampleByFilename =
        samples.associateBy { it.imageFilename }

      val expectedFileNames =
        files.map { it.name }.toSet()

      val metadataFileNames =
        sampleByFilename.keys

      if (metadataFileNames != expectedFileNames) {
        return@withContext BatchUploadResult(
          success = false,
          requestedCount = files.size,
          transferredCount = 0,
          batchId = session.batchId,
          message =
            "사진과 sample metadata가 일치하지 않습니다. " +
                    "images=${expectedFileNames.size}, metadata=${metadataFileNames.size}",
        )
      }

      val multipartBuilder =
        MultipartBody.Builder()
          .setType(MultipartBody.FORM)
          .addFormDataPart(
            "batch_id",
            session.batchId,
          )
          .addFormDataPart(
            "batch_metadata",
            session.toJson().toString(),
          )
          .addFormDataPart(
            "samples_json",
            samplesToJsonArray(samples).toString(),
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

      try {
        client.newCall(request).execute().use { response ->
          val responseText =
            response.body?.string().orEmpty()

          if (!response.isSuccessful) {
            return@withContext BatchUploadResult(
              success = false,
              requestedCount = files.size,
              transferredCount = 0,
              batchId = session.batchId,
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
                batchId = session.batchId,
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

          if (
            !ok ||
            responseBatchId != session.batchId ||
            savedCount != files.size ||
            savedFileNames != expectedFileNames
          ) {
            return@withContext BatchUploadResult(
              success = false,
              requestedCount = files.size,
              transferredCount = savedCount.coerceAtLeast(0),
              batchId = session.batchId,
              message =
                "서버의 저장 확인 내용이 휴대폰 파일 목록과 일치하지 않습니다.",
            )
          }

          moveUploadedBatch(
            batchId = session.batchId,
            files = files,
          )

          BatchUploadResult(
            success = true,
            requestedCount = files.size,
            transferredCount = files.size,
            batchId = session.batchId,
            message =
              "${session.placeName}: 사진 ${files.size}장을 데스크톱으로 전송했습니다.",
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
          batchId = session.batchId,
          message =
            "데스크톱 서버에 연결하지 못했습니다: " +
                    (exception.message ?: exception.javaClass.simpleName),
        )
      }
    }

  fun cancelAll() {
    client.dispatcher.cancelAll()
  }

  private fun getPendingFiles(batchId: String): List<File> {
    val directory =
      File(
        File(
          getPicturesBaseDirectory(),
          PENDING_FOLDER_NAME,
        ),
        batchId,
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

  private fun moveUploadedBatch(
    batchId: String,
    files: List<File>,
  ) {
    val base =
      getPicturesBaseDirectory()

    val uploadedDirectory =
      File(
        File(base, UPLOADED_FOLDER_NAME),
        batchId,
      )

    if (!uploadedDirectory.exists() && !uploadedDirectory.mkdirs()) {
      throw IOException(
        "전송 완료 폴더를 만들지 못했습니다: ${uploadedDirectory.absolutePath}",
      )
    }

    files.forEach { sourceFile ->
      moveFile(
        sourceFile,
        File(uploadedDirectory, sourceFile.name),
      )
    }

    val metadataSourceDirectory =
      File(
        File(base, METADATA_FOLDER_NAME),
        batchId,
      )

    if (metadataSourceDirectory.exists()) {
      val metadataTargetDirectory =
        File(
          File(base, UPLOADED_METADATA_FOLDER_NAME),
          batchId,
        )

      if (
        !metadataTargetDirectory.exists() &&
        !metadataTargetDirectory.mkdirs()
      ) {
        throw IOException(
          "metadata 전송 완료 폴더를 만들지 못했습니다: " +
                  metadataTargetDirectory.absolutePath,
        )
      }

      metadataSourceDirectory
        .listFiles()
        .orEmpty()
        .filter { it.isFile }
        .forEach { sourceFile ->
          moveFile(
            sourceFile,
            File(metadataTargetDirectory, sourceFile.name),
          )
        }

      metadataSourceDirectory.delete()
    }

    files.firstOrNull()?.parentFile?.delete()
  }

  private fun moveFile(
    sourceFile: File,
    targetFile: File,
  ) {
    if (targetFile.exists() && !targetFile.delete()) {
      throw IOException(
        "기존 파일을 교체하지 못했습니다: ${targetFile.absolutePath}",
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
          "원본 파일을 이동하지 못했습니다: ${sourceFile.absolutePath}",
        )
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
