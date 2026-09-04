package com.google.mediapipe.examples.poselandmarker.transport

import android.os.SystemClock
import com.google.mediapipe.examples.poselandmarker.transport.model.PoseFrame
import kotlin.math.PI
import kotlin.math.abs

/**
 * 对发送给OTT的第一个人体33个关键点进行One Euro滤波。
 * 输入滤波器前转换成像素坐标，输出时重新转换为MediaPipe归一化坐标。
 */
class OneEuroPoseFilter(
    private val initialIntervalSeconds: Float = 1f / 30f,
    private val minCutoff: Float = 1.0f,
    private val beta: Float = 0.01f,
    private val derivativeCutoff: Float = 30.0f,
    private val resetGapMs: Long = 400L
) {
    private var previousFilteredValues: FloatArray? = null
    private var previousFilteredDerivatives: FloatArray? = null

    private var lastTimestampMs = -1L
    private var previousImageWidth = 0
    private var previousImageHeight = 0

    /**
     * CameraX分析线程和相机切换可能分别调用filter/reset，
     * 使用Synchronized保证滤波状态不会同时被修改。
     */
    @Synchronized
    fun filter(frame: PoseFrame): PoseFrame {
        val person = frame.persons.firstOrNull()

        // 没检测到人体时清除旧状态，避免重新出现时从旧位置拖过来。
        if (person == null || person.landmarks.isEmpty()) {
            reset()
            return frame
        }

        if (frame.imageWidth <= 0 || frame.imageHeight <= 0) {
            reset()
            return frame
        }

        val nowMs = SystemClock.elapsedRealtime()

        // 输入尺寸发生变化，说明相机或旋转状态发生改变。
        if (previousImageWidth != 0 && (previousImageWidth != frame.imageWidth || previousImageHeight != frame.imageHeight)) {
            resetInternal()
        }

        // 长时间没有收到姿态，重新开始滤波。
        if (lastTimestampMs > 0L && nowMs - lastTimestampMs > resetGapMs) {
            resetInternal()
        }

        val rawValues = FloatArray(person.landmarks.size * 3)

        person.landmarks.forEachIndexed { index, landmark ->
            val valueIndex = index * 3
            // 使用像素单位。
            rawValues[valueIndex] = landmark.x * frame.imageWidth
            rawValues[valueIndex + 1] = landmark.y * frame.imageHeight
            // MediaPipe的z通常按图像宽度进行尺度转换。
            rawValues[valueIndex + 2] = landmark.z * frame.imageWidth
        }

        val filteredValues = filterValues(currentValues = rawValues, timestampMs = nowMs)

        previousImageWidth = frame.imageWidth
        previousImageHeight = frame.imageHeight

        val filteredLandmarks = person.landmarks.mapIndexed { index, landmark ->
                val valueIndex = index * 3
                landmark.copy(
                    x = filteredValues[valueIndex] / frame.imageWidth,
                    y = filteredValues[valueIndex + 1] / frame.imageHeight,
                    z = filteredValues[valueIndex + 2] / frame.imageWidth
                )
            }
        val filteredPerson = person.copy(landmarks = filteredLandmarks)

        // 当前OTT只使用第一个人。其他人物保持原始数据。
        val filteredPersons = frame.persons.toMutableList()
        filteredPersons[0] = filteredPerson
        return frame.copy(persons = filteredPersons)
    }

    private fun filterValues(
        currentValues: FloatArray,
        timestampMs: Long
    ): FloatArray {
        val previousValues = previousFilteredValues
        val previousDerivatives = previousFilteredDerivatives

        // 第一帧直接作为滤波初始值。
        if (previousValues == null || previousDerivatives == null || previousValues.size != currentValues.size
        ) {
            previousFilteredValues = currentValues.copyOf()
            previousFilteredDerivatives = FloatArray(currentValues.size)

            lastTimestampMs = timestampMs
            return currentValues.copyOf()
        }

        val measuredInterval = (timestampMs - lastTimestampMs) / 1000f

        // 防止时间间隔为0、过小或异常过大。
        val timeInterval = measuredInterval.coerceIn(1f / 120f, 0.2f)
        val derivativeAlpha = calculateAlpha(cutoff = derivativeCutoff, timeInterval = timeInterval)
        val filteredDerivatives = FloatArray(currentValues.size)
        val filteredValues = FloatArray(currentValues.size)

        for (index in currentValues.indices) {
            val currentDerivative = (currentValues[index] - previousValues[index]) / timeInterval
            val filteredDerivative = derivativeAlpha * currentDerivative + (1f - derivativeAlpha) * previousDerivatives[index]
            filteredDerivatives[index] = filteredDerivative

            // 运动越快，截止频率越高，减少快速动作的延迟。
            val adaptiveCutoff = minCutoff + beta * abs(filteredDerivative)
            val positionAlpha = calculateAlpha(cutoff = adaptiveCutoff, timeInterval = timeInterval)
            filteredValues[index] = positionAlpha * currentValues[index] + (1f - positionAlpha) * previousValues[index]
        }

        previousFilteredValues = filteredValues.copyOf()
        previousFilteredDerivatives = filteredDerivatives.copyOf()
        lastTimestampMs = timestampMs

        return filteredValues
    }

    private fun calculateAlpha(
        cutoff: Float,
        timeInterval: Float
    ): Float {
        val safeCutoff = cutoff.coerceAtLeast(0.0001f)
        val tau = (1.0 / (2.0 * PI * safeCutoff)).toFloat()

        return 1f / (1f + tau / timeInterval)
    }

    @Synchronized
    fun reset() {
        resetInternal()
    }

    private fun resetInternal() {
        previousFilteredValues = null
        previousFilteredDerivatives = null
        lastTimestampMs = -1L
        previousImageWidth = 0
        previousImageHeight = 0
    }
}