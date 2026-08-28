/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.io.File



class PoseLandmarkerHelper(
    var minPoseDetectionConfidence: Float = DEFAULT_POSE_DETECTION_CONFIDENCE,
    var minPoseTrackingConfidence: Float = DEFAULT_POSE_TRACKING_CONFIDENCE,
    var minPosePresenceConfidence: Float = DEFAULT_POSE_PRESENCE_CONFIDENCE,
    var currentModel: Int = MODEL_POSE_LANDMARKER_FULL,
    var currentDelegate: Int = DELEGATE_CPU,
    var runningMode: RunningMode = RunningMode.IMAGE,


    val context: Context,
    // 新增：保存文件夹
    private val saveDirectory: File,
    // this listener is only used when running in RunningMode.LIVE_STREAM
    val poseLandmarkerHelperListener: LandmarkerListener? = null
) {

    // For this example this needs to be a var so it can be reset on changes.
    // If the Pose Landmarker will not change, a lazy val would be preferable.
    private var poseLandmarker: PoseLandmarker? = null
//    private val poseData = StringBuilder()

    /**
     * CSV字段：
     *
     * system_timestamp_ms:
     *     当前真实系统时间，System.currentTimeMillis()，
     *     从1970-01-01 UTC开始累计的毫秒数。
     *
     * frame_timestamp_ms:
     *     当前输入帧在对应数据流中的时间。
     *
     * IMAGE模式：固定写0，因为单张图片没有视频时间轴。
     * VIDEO模式：视频中的帧时间，例如0、300、600 ms。
     * LIVE_STREAM模式：SystemClock.uptimeMillis()产生的单调递增帧时间。
     */
    private val csvHeader =
        "system_timestamp_ms," +
                "frame_timestamp_ms," +
                "person_id," +
                "landmark_id," +
                "x,y,z," +
                "world_x,world_y,world_z," +
                "visibility,presence\n"
    /**
     * StringBuilder不是线程安全的，因此增加同步锁。
     */
    private val poseDataLock = Any()

    /**
     * CSV数据缓存。
     */
    private val poseData = StringBuilder(csvHeader)//将字符头写入到poseData

    /**
     * 实际关键点数据行数。
     * 只有当该值大于0时才创建CSV文件。
     */
    private var poseDataRowCount = 0

    //它属于类的一部分。当根据这个类创建对象时，init中的代码会自动执行
    init {
        setupPoseLandmarker()
    }

    /**
     * MediaPipe中的visibility和presence是Optional<Float>。
     * 有值时写入数值；
     * 没有值时写入空字符串，而不是写null。
     */
    private fun optionalFloatToCsv(
        value: java.util.Optional<Float>
    ): String {

        return if (value.isPresent) {
            value.get().toString()
        } else {
            ""
        }
    }

    //每次识别都会生成不同的文件夹
    /**
     * 将关键点缓存保存为CSV文件。
     *
     * 没有任何关键点数据时不生成文件。
     */
    private fun savePoseDataToFile(): File? {

        try {

            /* 先检查实际数据行数。
             * poseDataRowCount为0时，poseData中只有表头，
             * 此时直接跳过文件创建。
             */
            val snapshot =
                synchronized(poseDataLock) {

                    if (poseDataRowCount <= 0) {

                        Log.i(
                            TAG,
                            "Skip CSV save: no landmark data."
                        )

                        return null
                    }

                    Pair(
                        poseData.toString(),
                        poseDataRowCount
                    )
                }

            val csvContent = snapshot.first
            val rowCount = snapshot.second

            // 创建保存目录
            if (!saveDirectory.exists()) {

                val created =
                    saveDirectory.mkdirs()

                Log.i(
                    TAG,
                    "Create directory result=$created, " +
                            "path=${saveDirectory.absolutePath}"
                )
            }

            if (
                !saveDirectory.exists() ||
                !saveDirectory.isDirectory
            ) {

                Log.e(
                    TAG,
                    "Save directory unavailable: " +
                            saveDirectory.absolutePath
                )

                return null
            }

            /*
             * 文件名增加毫秒SSS，
             * 避免一秒内多次保存时文件名重复。
             */
            val time =
                java.text.SimpleDateFormat(
                    "yyyyMMdd_HHmmss_SSS",
                    java.util.Locale.US
                ).format(java.util.Date())

            val file =
                File(
                    saveDirectory,
                    "pose_$time.csv"
                )

            file.bufferedWriter(
                Charsets.UTF_8
            ).use { writer ->

                writer.write(csvContent)
            }

            Log.i(
                TAG,
                "Pose CSV saved: " +
                        "path=${file.absolutePath}, " +
                        "rows=$rowCount, " +
                        "size=${file.length()} bytes"
            )

            /*
             * 只有文件成功写入以后才清空缓存。
             * 如果文件写入失败，旧数据仍然保留，
             * 不会直接丢失。
             */
            resetPoseData()

            return file

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Save pose CSV failed. " +
                        "directory=${saveDirectory.absolutePath}",
                e
            )

            return null
        }
    }

    /**
     * 关闭Pose Landmarker。
     *
     * @param saveCsv
     * true：关闭前后保存有效关键点数据；
     * false：只关闭模型，不保存CSV。
     */
    fun clearPoseLandmarker(
        saveCsv: Boolean = false
    ) {

        /*
         * 先关闭模型，避免实时回调在保存过程中
         * 继续向StringBuilder追加数据。
         */
        poseLandmarker?.close()

        poseLandmarker = null

        if (saveCsv) {

            savePoseDataToFile()
        }
    }

    // Return running status of PoseLandmarkerHelper
    fun isClose(): Boolean {
        return poseLandmarker == null
    }

    // Initialize the Pose landmarker using current settings on the
    // thread that is using it. CPU can be used with Landmarker
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the
    // Landmarker
    fun setupPoseLandmarker() {
        // Set general pose landmarker options
        val baseOptionBuilder = BaseOptions.builder()

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                baseOptionBuilder.setDelegate(Delegate.GPU)
            }
        }

        val modelName =
            when (currentModel) {
                MODEL_POSE_LANDMARKER_FULL -> "pose_landmarker_full.task"
                MODEL_POSE_LANDMARKER_LITE -> "pose_landmarker_lite.task"
                MODEL_POSE_LANDMARKER_HEAVY -> "pose_landmarker_heavy.task"
                else -> "pose_landmarker_full.task"
            }

        baseOptionBuilder.setModelAssetPath(modelName)

        // Check if runningMode is consistent with poseLandmarkerHelperListener
        when (runningMode) {
            RunningMode.LIVE_STREAM -> {
                if (poseLandmarkerHelperListener == null) {
                    throw IllegalStateException(
                        "poseLandmarkerHelperListener must be set when runningMode is LIVE_STREAM."
                    )
                }
            }
            else -> {
                // no-op
            }
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            // Create an option builder with base options and specific
            // options only use for Pose Landmarker.
            val optionsBuilder =
                PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    //设定识别人数
                    .setNumPoses(DEFAULT_NUM_POSES)
                    .setMinPoseDetectionConfidence(minPoseDetectionConfidence)
                    .setMinTrackingConfidence(minPoseTrackingConfidence)
                    .setMinPosePresenceConfidence(minPosePresenceConfidence)
                    .setRunningMode(runningMode)

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                    .setResultListener(this::returnLivestreamResult)
                    .setErrorListener(this::returnLivestreamError)
            }

            val options = optionsBuilder.build()
            poseLandmarker =
                PoseLandmarker.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            poseLandmarkerHelperListener?.onError(
                "Pose Landmarker failed to initialize. See error logs for " +
                        "details"
            )
            Log.e(
                TAG, "MediaPipe failed to load the task with error: " + e
                    .message
            )
        } catch (e: RuntimeException) {
            // This occurs if the model being used does not support GPU
            poseLandmarkerHelperListener?.onError(
                "Pose Landmarker failed to initialize. See error logs for " +
                        "details", GPU_ERROR
            )
            Log.e(
                TAG,
                "Image classifier failed to load model with error: " + e.message
            )
        }
    }

    // Convert the ImageProxy to MP Image and feed it to PoselandmakerHelper.
    fun detectLiveStream(
        imageProxy: ImageProxy,
        isFrontCamera: Boolean
    ) {
        if (runningMode != RunningMode.LIVE_STREAM) {
            throw IllegalArgumentException(
                "Attempting to call detectLiveStream" +
                        " while not using RunningMode.LIVE_STREAM"
            )
        }
        val frameTime = SystemClock.uptimeMillis()

        // Copy out RGB bits from the frame to a bitmap buffer
        val bitmapBuffer =
            Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )

        imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
        imageProxy.close()

        val matrix = Matrix().apply {
            // Rotate the frame received from the camera to be in the same direction as it'll be shown
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

            // flip image if user use front camera
            if (isFrontCamera) {
                postScale(
                    -1f,
                    1f,
                    imageProxy.width.toFloat(),
                    imageProxy.height.toFloat()
                )
            }
        }
        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
            matrix, true
        )

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(rotatedBitmap).build()

        detectAsync(mpImage, frameTime)
    }

    // Run pose landmark using MediaPipe Pose Landmarker API
    @VisibleForTesting
    fun detectAsync(mpImage: MPImage, frameTime: Long) {
        poseLandmarker?.detectAsync(mpImage, frameTime)
        // As we're using running mode LIVE_STREAM, the landmark result will
        // be returned in returnLivestreamResult function
    }

    fun resetPoseData() {

        synchronized(poseDataLock) {

            poseData.clear()

            poseData.append(csvHeader)

            poseDataRowCount = 0
        }
    }


    // Accepts the URI for a video file loaded from the user's gallery and attempts to run
    // pose landmarker inference on the video. This process will evaluate every
    // frame in the video and attach the results to a bundle that will be
    // returned.
    fun detectVideoFile(
        videoUri: Uri,
        inferenceIntervalMs: Long
    ): ResultBundle? {
        if (runningMode != RunningMode.VIDEO) {
            throw IllegalArgumentException(
                "Attempting to call detectVideoFile" +
                        " while not using RunningMode.VIDEO"
            )
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        val startTime = SystemClock.uptimeMillis()

        var didErrorOccurred = false

        // Load frames from the video and run the pose landmarker.
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(context, videoUri)
        val videoLengthMs =
            retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLong()

        // Note: We need to read width/height from frame instead of getting the width/height
        // of the video directly because MediaRetriever returns frames that are smaller than the
        // actual dimension of the video file.
        val firstFrame = retriever.getFrameAtTime(0)
        val width = firstFrame?.width
        val height = firstFrame?.height

        // If the video is invalid, returns a null detection result
        if ((videoLengthMs == null) || (width == null) || (height == null)) return null

        // Next, we'll get one frame every frameInterval ms, then run detection on these frames.
        val resultList = mutableListOf<PoseLandmarkerResult>()
        val numberOfFrameToRead = videoLengthMs.div(inferenceIntervalMs)

        for (i in 0..numberOfFrameToRead) {
            val timestampMs = i * inferenceIntervalMs // ms

            retriever
                .getFrameAtTime(
                    timestampMs * 1000, // convert from ms to micro-s
                    MediaMetadataRetriever.OPTION_CLOSEST
                )
                ?.let { frame ->
                    // Convert the video frame to ARGB_8888 which is required by the MediaPipe
                    val argb8888Frame =
                        if (frame.config == Bitmap.Config.ARGB_8888) frame
                        else frame.copy(Bitmap.Config.ARGB_8888, false)

                    // Convert the input Bitmap object to an MPImage object to run inference
                    val mpImage = BitmapImageBuilder(argb8888Frame).build()

                    // Run pose landmarker using MediaPipe Pose Landmarker API
                    poseLandmarker?.detectForVideo(mpImage, timestampMs)
                        ?.let { detectionResult ->

                            // 保存视频每一帧关键点
                            saveLandmarks(
                                result = detectionResult,
                                frameTimestampMs = timestampMs
                            )

                            resultList.add(detectionResult)
                        } ?: {
                        didErrorOccurred = true
                        poseLandmarkerHelperListener?.onError(
                            "ResultBundle could not be returned" +
                                    " in detectVideoFile"
                        )
                    }
                }
                ?: run {
                    didErrorOccurred = true
                    poseLandmarkerHelperListener?.onError(
                        "Frame at specified time could not be" +
                                " retrieved when detecting in video."
                    )
                }
        }

        retriever.release()

        val inferenceTimePerFrameMs =
            (SystemClock.uptimeMillis() - startTime).div(numberOfFrameToRead)

        return if (didErrorOccurred) {
            null
        } else {
            ResultBundle(resultList, inferenceTimePerFrameMs, height, width)
        }
    }

    // Accepted a Bitmap and runs pose landmarker inference on it to return
    // results back to the caller
    fun detectImage(image: Bitmap): ResultBundle? {
        if (runningMode != RunningMode.IMAGE) {
            throw IllegalArgumentException(
                "Attempting to call detectImage" +
                        " while not using RunningMode.IMAGE"
            )
        }


        // Inference time is the difference between the system time at the
        // start and finish of the process
        val startTime = SystemClock.uptimeMillis()

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(image).build()

        // Run pose landmarker using MediaPipe Pose Landmarker API
        poseLandmarker?.detect(mpImage)?.also { landmarkResult ->

            // 保存图片检测得到的关键点
            saveLandmarks(
                result = landmarkResult,
                frameTimestampMs = 0L
            )

            val inferenceTimeMs = SystemClock.uptimeMillis() - startTime
            return ResultBundle(
                listOf(landmarkResult),
                inferenceTimeMs,
                image.height,
                image.width
            )
        }

        // If poseLandmarker?.detect() returns null, this is likely an error. Returning null
        // to indicate this.
        poseLandmarkerHelperListener?.onError(
            "Pose Landmarker failed to detect."
        )
        return null
    }

    /**
     * 将一次检测结果追加到CSV缓存。
     *
     * @param result MediaPipe姿态检测结果
     * @param frameTimestampMs 当前输入帧时间戳
     */
    private fun saveLandmarks(
        result: PoseLandmarkerResult,
        frameTimestampMs: Long
    ) {

        // 当前真实系统时间，Unix Epoch毫秒时间戳
        val systemTimestampMs = System.currentTimeMillis()

        // 归一化图像坐标
        val normalizedPoses = result.landmarks()

        // 世界坐标，单位为米
        val worldPoses = result.worldLandmarks()

        // 没有检测到任何人体时，直接返回，不增加数据行
        if (normalizedPoses.isEmpty()) {
            Log.d(
                TAG,
                "No pose detected at frameTimestampMs=$frameTimestampMs"
            )
            return
        }

        synchronized(poseDataLock) {

            normalizedPoses.forEachIndexed {
                    personId,
                    normalizedLandmarks ->

                // 获取同一个人的世界坐标
                val worldLandmarks =
                    worldPoses.getOrNull(personId)

                normalizedLandmarks.forEachIndexed {
                        landmarkId,
                        normalizedLandmark ->

                    // 获取同一个关键点的世界坐标
                    val worldLandmark =
                        worldLandmarks?.getOrNull(landmarkId)

                    val visibility =
                        optionalFloatToCsv(
                            normalizedLandmark.visibility()
                        )

                    val presence =
                        optionalFloatToCsv(
                            normalizedLandmark.presence()
                        )

                    /*
                     * 理论上Pose Landmarker应当同时输出世界坐标。
                     * 使用空字符串处理极少数列表不完整的情况，
                     * 避免数组越界或NullPointerException。
                     */
                    val worldX =
                        worldLandmark?.x()?.toString() ?: ""

                    val worldY =
                        worldLandmark?.y()?.toString() ?: ""

                    val worldZ =
                        worldLandmark?.z()?.toString() ?: ""

                    poseData
                        .append(systemTimestampMs)
                        .append(",")

                        .append(frameTimestampMs)
                        .append(",")

                        .append(personId)
                        .append(",")

                        .append(landmarkId)
                        .append(",")

                        .append(normalizedLandmark.x())
                        .append(",")

                        .append(normalizedLandmark.y())
                        .append(",")

                        .append(normalizedLandmark.z())
                        .append(",")

                        .append(worldX)
                        .append(",")

                        .append(worldY)
                        .append(",")

                        .append(worldZ)
                        .append(",")

                        .append(visibility)
                        .append(",")

                        .append(presence)
                        .append("\n")

                    poseDataRowCount++
                }
            }
        }
    }

    // Return the landmark result to this PoseLandmarkerHelper's caller
    private fun returnLivestreamResult(
        result: PoseLandmarkerResult,
        input: MPImage
    ) {
        val finishTimeMs = SystemClock.uptimeMillis()
        val inferenceTime = finishTimeMs - result.timestampMs()

        // 保存关键点坐标
        saveLandmarks(
            result = result,
            frameTimestampMs = result.timestampMs()
        )

        poseLandmarkerHelperListener?.onResults(
            ResultBundle(
                listOf(result),
                inferenceTime,
                input.height,
                input.width
            )
        )
    }

    // Return errors thrown during detection to this PoseLandmarkerHelper's
    // caller
    private fun returnLivestreamError(error: RuntimeException) {
        poseLandmarkerHelperListener?.onError(
            error.message ?: "An unknown error has occurred"
        )
    }

    //公用常量
    companion object {
        const val TAG = "PoseLandmarkerHelper"

        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DEFAULT_POSE_DETECTION_CONFIDENCE = 0.5F
        const val DEFAULT_POSE_TRACKING_CONFIDENCE = 0.5F
        const val DEFAULT_POSE_PRESENCE_CONFIDENCE = 0.5F

        //修改识别的人数
        const val DEFAULT_NUM_POSES = 2
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
        const val MODEL_POSE_LANDMARKER_FULL = 0
        const val MODEL_POSE_LANDMARKER_LITE = 1
        const val MODEL_POSE_LANDMARKER_HEAVY = 2
    }

    // 一次检测结果的数据包
    data class ResultBundle(
        val results: List<PoseLandmarkerResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
    )

    // 结果回调接口
    interface LandmarkerListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)
    }
}