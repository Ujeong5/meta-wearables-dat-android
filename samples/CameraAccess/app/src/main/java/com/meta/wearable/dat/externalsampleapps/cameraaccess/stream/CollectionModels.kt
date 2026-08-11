package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import org.json.JSONArray
import org.json.JSONObject

data class LocationSnapshot(
  val latitude: Double,
  val longitude: Double,
  val accuracyMeters: Float,
)

data class CollectionSession(
  val batchId: String,
  val placeName: String,
  val gpsLatitude: Double,
  val gpsLongitude: Double,
  val gpsAccuracyM: Float,
  val sessionStartTime: String,
  val sessionEndTime: String? = null,
) {
  fun toJson(): JSONObject =
    JSONObject().apply {
      put("batch_id", batchId)
      put("place_name", placeName)
      put("gps_latitude", gpsLatitude)
      put("gps_longitude", gpsLongitude)
      put("gps_accuracy_m", gpsAccuracyM)
      put("session_start_time", sessionStartTime)
      put(
        "session_end_time",
        sessionEndTime ?: JSONObject.NULL,
      )
    }

  companion object {
    fun fromJson(json: JSONObject): CollectionSession =
      CollectionSession(
        batchId = json.getString("batch_id"),
        placeName = json.getString("place_name"),
        gpsLatitude = json.getDouble("gps_latitude"),
        gpsLongitude = json.getDouble("gps_longitude"),
        gpsAccuracyM = json.getDouble("gps_accuracy_m").toFloat(),
        sessionStartTime = json.getString("session_start_time"),
        sessionEndTime =
          if (json.isNull("session_end_time")) {
            null
          } else {
            json.optString("session_end_time").takeIf { it.isNotBlank() }
          },
      )
  }
}

data class SampleMetadata(
  val batchId: String,
  val sampleId: String,
  val imageFilename: String,
  val captureTimestamp: String,
  val sourceWidth: Int,
  val sourceHeight: Int,
  val cropWidth: Int,
  val cropHeight: Int,
  val yoloConfidence: Float,
  val bboxX1Norm: Float,
  val bboxY1Norm: Float,
  val bboxX2Norm: Float,
  val bboxY2Norm: Float,
  val bboxAreaRatio: Float,
  val poseMeanConfidence: Float,
  val poseMinConfidence: Float,
  val osnetMaxSimilarity: Float?,
) {
  fun toJson(): JSONObject =
    JSONObject().apply {
      put("batch_id", batchId)
      put("sample_id", sampleId)
      put("image_filename", imageFilename)
      put("capture_timestamp", captureTimestamp)
      put("source_width", sourceWidth)
      put("source_height", sourceHeight)
      put("crop_width", cropWidth)
      put("crop_height", cropHeight)
      put("yolo_confidence", yoloConfidence)
      put("bbox_x1_norm", bboxX1Norm)
      put("bbox_y1_norm", bboxY1Norm)
      put("bbox_x2_norm", bboxX2Norm)
      put("bbox_y2_norm", bboxY2Norm)
      put("bbox_area_ratio", bboxAreaRatio)
      put("pose_mean_confidence", poseMeanConfidence)
      put("pose_min_confidence", poseMinConfidence)
      put(
        "osnet_max_similarity",
        osnetMaxSimilarity ?: JSONObject.NULL,
      )
    }

  companion object {
    fun fromJson(json: JSONObject): SampleMetadata =
      SampleMetadata(
        batchId = json.getString("batch_id"),
        sampleId = json.getString("sample_id"),
        imageFilename = json.getString("image_filename"),
        captureTimestamp = json.getString("capture_timestamp"),
        sourceWidth = json.getInt("source_width"),
        sourceHeight = json.getInt("source_height"),
        cropWidth = json.getInt("crop_width"),
        cropHeight = json.getInt("crop_height"),
        yoloConfidence = json.getDouble("yolo_confidence").toFloat(),
        bboxX1Norm = json.getDouble("bbox_x1_norm").toFloat(),
        bboxY1Norm = json.getDouble("bbox_y1_norm").toFloat(),
        bboxX2Norm = json.getDouble("bbox_x2_norm").toFloat(),
        bboxY2Norm = json.getDouble("bbox_y2_norm").toFloat(),
        bboxAreaRatio = json.getDouble("bbox_area_ratio").toFloat(),
        poseMeanConfidence = json.getDouble("pose_mean_confidence").toFloat(),
        poseMinConfidence = json.getDouble("pose_min_confidence").toFloat(),
        osnetMaxSimilarity =
          if (json.isNull("osnet_max_similarity")) {
            null
          } else {
            json.getDouble("osnet_max_similarity").toFloat()
          },
      )
  }
}

fun samplesToJsonArray(samples: List<SampleMetadata>): JSONArray =
  JSONArray().apply {
    samples.forEach { sample ->
      put(sample.toJson())
    }
  }
