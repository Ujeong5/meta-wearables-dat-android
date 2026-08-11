package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.CancellationSignal
import androidx.core.content.ContextCompat
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull

class PhoneLocationProvider(
  private val context: Context,
) {
  private val locationManager =
    context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

  suspend fun getCurrentGpsLocation(
    timeoutMs: Long = 20_000L,
  ): Result<LocationSnapshot> {
    if (
      ContextCompat.checkSelfPermission(
        context,
        Manifest.permission.ACCESS_FINE_LOCATION,
      ) != PackageManager.PERMISSION_GRANTED
    ) {
      return Result.failure(
        SecurityException("정확한 위치 권한이 필요합니다."),
      )
    }

    if (!locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
      return Result.failure(
        IllegalStateException(
          "휴대폰의 위치(GPS)가 꺼져 있습니다. 위치 기능을 켠 뒤 다시 시도하세요.",
        ),
      )
    }

    val snapshot =
      try {
        withTimeoutOrNull(timeoutMs) {
          suspendCancellableCoroutine<LocationSnapshot?> { continuation ->
            val cancellationSignal = CancellationSignal()

            continuation.invokeOnCancellation {
              cancellationSignal.cancel()
            }

            try {
              locationManager.getCurrentLocation(
                LocationManager.GPS_PROVIDER,
                cancellationSignal,
                context.mainExecutor,
              ) { location ->
                if (!continuation.isActive) {
                  return@getCurrentLocation
                }

                if (location == null) {
                  continuation.resume(null)
                } else {
                  continuation.resume(
                    LocationSnapshot(
                      latitude = location.latitude,
                      longitude = location.longitude,
                      accuracyMeters = location.accuracy,
                    ),
                  )
                }
              }
            } catch (exception: Exception) {
              if (continuation.isActive) {
                continuation.resumeWithException(exception)
              }
            }
          }
        }
      } catch (exception: Exception) {
        return Result.failure(exception)
      }

    return if (snapshot == null) {
      Result.failure(
        IllegalStateException(
          "20초 안에 GPS 위치를 얻지 못했습니다. 야외에서 다시 시도하세요.",
        ),
      )
    } else {
      Result.success(snapshot)
    }
  }
}
