package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.content.Context
import android.os.Environment
import java.io.File
import java.io.IOException
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.UUID
import org.json.JSONObject

class CollectionMetadataStore(
  private val context: Context,
) {
  companion object {
    private const val CROP_FOLDER_NAME = "person_crops_mosaicked"
    private const val CROP_METADATA_FOLDER_NAME = "person_crop_metadata"
    private const val SESSION_FOLDER_NAME = "collection_sessions"

    private val BATCH_TIME_FORMATTER =
      DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")
  }

  fun createSession(
    placeName: String,
    location: LocationSnapshot,
  ): CollectionSession {
    val trimmedPlace = placeName.trim()
    require(trimmedPlace.isNotBlank()) {
      "장소명을 입력해야 합니다."
    }

    val now = Instant.now()
    val batchId =
      BATCH_TIME_FORMATTER
        .withZone(ZoneId.systemDefault())
        .format(now) +
        "_" +
        UUID.randomUUID()
          .toString()
          .replace("-", "")
          .take(6)

    val session =
      CollectionSession(
        batchId = batchId,
        placeName = trimmedPlace,
        gpsLatitude = location.latitude,
        gpsLongitude = location.longitude,
        gpsAccuracyM = location.accuracyMeters,
        sessionStartTime = now.toString(),
      )

    saveSession(session)
    getPendingCropDirectory(batchId).mkdirs()
    getSampleMetadataDirectory(batchId).mkdirs()

    return session
  }

  fun finalizeSession(
    session: CollectionSession,
  ): CollectionSession {
    val finalized =
      if (session.sessionEndTime != null) {
        session
      } else {
        session.copy(
          sessionEndTime = Instant.now().toString(),
        )
      }

    saveSession(finalized)
    return finalized
  }

  fun saveSession(session: CollectionSession) {
    val directory = getSessionDirectory()
    ensureDirectory(directory)

    val file = File(directory, "${session.batchId}.json")
    file.writeText(
      session.toJson().toString(2),
      Charsets.UTF_8,
    )
  }

  fun loadSession(batchId: String): CollectionSession? {
    val file = File(getSessionDirectory(), "$batchId.json")
    if (!file.exists()) {
      return null
    }

    return CollectionSession.fromJson(
      JSONObject(file.readText(Charsets.UTF_8)),
    )
  }

  fun saveSampleMetadata(metadata: SampleMetadata) {
    val directory = getSampleMetadataDirectory(metadata.batchId)
    ensureDirectory(directory)

    val file =
      File(
        directory,
        "${File(metadata.imageFilename).nameWithoutExtension}.json",
      )

    file.writeText(
      metadata.toJson().toString(2),
      Charsets.UTF_8,
    )
  }

  fun loadSampleMetadata(batchId: String): List<SampleMetadata> {
    val directory = getSampleMetadataDirectory(batchId)
    if (!directory.exists()) {
      return emptyList()
    }

    return directory
      .listFiles { file ->
        file.isFile && file.extension.equals("json", ignoreCase = true)
      }
      ?.sortedBy { it.name }
      ?.map { file ->
        SampleMetadata.fromJson(
          JSONObject(file.readText(Charsets.UTF_8)),
        )
      }
      .orEmpty()
  }

  fun nextSampleId(batchId: String): String {
    val count =
      getSampleMetadataDirectory(batchId)
        .listFiles { file ->
          file.isFile && file.extension.equals("json", ignoreCase = true)
        }
        ?.size
        ?: 0

    return "S${(count + 1).toString().padStart(4, '0')}"
  }

  fun getPendingCropDirectory(batchId: String): File =
    File(
      File(getPicturesBaseDirectory(), CROP_FOLDER_NAME),
      batchId,
    )

  fun getSampleMetadataDirectory(batchId: String): File =
    File(
      File(getPicturesBaseDirectory(), CROP_METADATA_FOLDER_NAME),
      batchId,
    )

  fun timestampFromEpochMillis(epochMillis: Long): String =
    Instant.ofEpochMilli(epochMillis).toString()

  private fun getSessionDirectory(): File =
    File(
      getPicturesBaseDirectory(),
      SESSION_FOLDER_NAME,
    )

  private fun getPicturesBaseDirectory(): File =
    context.getExternalFilesDir(
      Environment.DIRECTORY_PICTURES,
    ) ?: context.filesDir

  private fun ensureDirectory(directory: File) {
    if (!directory.exists() && !directory.mkdirs()) {
      throw IOException(
        "폴더를 만들지 못했습니다: ${directory.absolutePath}",
      )
    }
  }
}
