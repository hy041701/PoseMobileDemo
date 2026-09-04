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
package com.google.mediapipe.examples.poselandmarker.fragment

import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.res.Configuration
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.core.AspectRatio
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import com.google.mediapipe.examples.poselandmarker.PoseLandmarkerHelper
import com.google.mediapipe.examples.poselandmarker.MainViewModel
import com.google.mediapipe.examples.poselandmarker.R
import com.google.mediapipe.examples.poselandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.io.File
import com.google.mediapipe.examples.poselandmarker.transport.PoseFrameConverter
import com.google.mediapipe.examples.poselandmarker.transport.protocol.PoseJsonEncoder
import com.google.mediapipe.examples.poselandmarker.transport.network.PoseConnectionManager

import android.app.Activity
import android.content.Intent
import androidx.activity.result.contract.ActivityResultContracts
import com.google.mediapipe.examples.poselandmarker.pairing.QrScanActivity
import com.google.mediapipe.examples.poselandmarker.transport.OneEuroPoseFilter

class CameraFragment : Fragment(), PoseLandmarkerHelper.LandmarkerListener,
    PoseConnectionManager.StateListener {
    companion object {
        private const val TAG = "Pose Landmarker"
        private const val FULL_BODY_CALIBRATION_FRAMES = 3
        private const val FULL_BODY_MIN_CONFIDENCE = 0.60f
        private const val FULL_BODY_EDGE_MARGIN = 0.03f
        private const val CALIBRATION_HINT_DURATION_MS = 5_000L
    }
    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!
    private lateinit var poseLandmarkerHelper: PoseLandmarkerHelper
    private lateinit var saveDirectory: File
    private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_BACK
    private var poseFrameId: Long = 0L

    // 发送给OTT前使用的姿态坐标滤波器。
    private val poseFrameFilter = OneEuroPoseFilter(initialIntervalSeconds = 1f / 30f, minCutoff = 1.0f, beta = 0.01f, derivativeCutoff = 30.0f, resetGapMs = 400L)

    private val uiHandler = Handler(Looper.getMainLooper())
    @Volatile private var isReadyCheckActive = false
    @Volatile private var isFullBodyCalibrationHintActive = false
    @Volatile private var isFullBodyCalibrationFinishedForConnection = false
    private var completeBodyCalibrationFrameCount = 0
    private val calibrationHintTimeoutTask = Runnable { finishFullBodyCalibrationHint() }
    private var readyPoseStartedAtMs: Long? = null
    private var connectionSuccessDialog: AlertDialog? = null
    private var originalCameraTopMargin: Int? = null
    private var isImmersiveDetectionMode = false
    private var isOpeningQrAfterDetectionExit = false
    @Volatile private var isPoseModelLoading = true

    //临时性能诊断：统计CameraX送帧、MediaPipe结果回调和姿态发送速度。测试结束后删除。
    private var posePerfWindowStartedAtMs = SystemClock.elapsedRealtime()
    private var posePerfCameraFrameCount = 0L
    private var posePerfCameraWidth = 0
    private var posePerfCameraHeight = 0
    private var posePerfInferenceCount = 0L
    private var posePerfValidPersonCount = 0L
    private var posePerfSendAttemptCount = 0L
    private var posePerfSendSuccessCount = 0L
    private var posePerfInferenceTotalMs = 0L
    private var posePerfInferenceMaxMs = 0L

    //QrScanActivity扫描完成以后，host和port会从这里返回。
    private val qrScanLauncher = registerForActivityResult(
            ActivityResultContracts.StartActivityForResult()
        ) { result ->
            // 1. 判断扫码Activity是否成功返回
            if (result.resultCode != Activity.RESULT_OK) {
                Log.w(TAG, "QR scan cancelled")
                return@registerForActivityResult
            }

            // 2. 取得OTT IP
            val host = result.data?.getStringExtra("ott_host")
            if (host.isNullOrBlank()) {
                Log.e(TAG, "QR result has no host")
                return@registerForActivityResult
            }

            // 3. 取得OTT端口
            val port = result.data?.getIntExtra("ott_port", -1) ?: -1
            if (port !in 1..65535) {
                Log.e(TAG, "QR result invalid port=$port")
                return@registerForActivityResult
            }

            //扫码页结束后立即切换为全屏前置相机，避免短暂露出普通Camera组件。
            showFullScreenCameraWhileConnecting()

            // 5. 使用二维码中的动态IP连接OTT
            PoseConnectionManager.connect(host = host, port = port)
        }

    //Blocking ML operations are performed using this executor
    private lateinit var backgroundExecutor: ExecutorService

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(R.id.action_camera_to_permissions)
        }

        //正常切换扫码页或其他App时模型不会再在onPause中释放；这里只保留异常关闭后的兜底恢复。
        backgroundExecutor.execute {
            if(this::poseLandmarkerHelper.isInitialized) {
                if (poseLandmarkerHelper.isClose() && !isPoseModelLoading) {
                    isPoseModelLoading = true
                    poseLandmarkerHelper.setupPoseLandmarker()
                    isPoseModelLoading = false
                }
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if(this::poseLandmarkerHelper.isInitialized) {
            viewModel.setMinPoseDetectionConfidence(poseLandmarkerHelper.minPoseDetectionConfidence)
            viewModel.setMinPoseTrackingConfidence(poseLandmarkerHelper.minPoseTrackingConfidence)
            viewModel.setMinPosePresenceConfidence(poseLandmarkerHelper.minPosePresenceConfidence)
            viewModel.setDelegate(poseLandmarkerHelper.currentDelegate)

            //短暂离开页面（例如打开扫码页或切换App）不销毁模型，返回后可立即继续识别。
        }
    }

    override fun onDestroyView() {
        PoseConnectionManager.clearStateListener(this)
        uiHandler.removeCallbacksAndMessages(null)
        connectionSuccessDialog?.dismiss()
        connectionSuccessDialog = null

        //页面真正销毁时才在GPU所属线程释放MediaPipe模型。
        if (::backgroundExecutor.isInitialized) {
            if (this::poseLandmarkerHelper.isInitialized) {
                backgroundExecutor.execute {
                    poseLandmarkerHelper.clearPoseLandmarker(saveCsv = false)
                }
            }
            backgroundExecutor.shutdown()
            try {
                backgroundExecutor.awaitTermination(2, TimeUnit.SECONDS)
            } catch (e: InterruptedException) {
                Log.e(TAG, "Executor shutdown interrupted", e)
            }
        }

        // 最后再清理ViewBinding
        _fragmentCameraBinding = null
        super.onDestroyView()
    }
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)
        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(
        view: View,
        savedInstanceState: Bundle?
    ) {
        super.onViewCreated(view, savedInstanceState)

        backgroundExecutor = Executors.newSingleThreadExecutor()

        // 设置Camera保存路径
        saveDirectory = File(
            requireContext().getExternalFilesDir(null),
            "PoseDataset/camera"
        )

        PoseConnectionManager.setStateListener(this)

        //相机预览不依赖MediaPipe模型，进入页面后立即启动，避免等待模型时出现黑屏。
        fragmentCameraBinding.viewFinder.post { setUpCamera() }

        //GPU必须在创建它的同一线程中使用；Helper初始化与CameraX分析共用该单线程。
        backgroundExecutor.execute {
            poseLandmarkerHelper = PoseLandmarkerHelper(
                context = requireContext(),
                saveDirectory = saveDirectory,
                runningMode = RunningMode.LIVE_STREAM,
                minPoseDetectionConfidence = viewModel.currentMinPoseDetectionConfidence,
                minPoseTrackingConfidence = viewModel.currentMinPoseTrackingConfidence,
                minPosePresenceConfidence = viewModel.currentMinPosePresenceConfidence,
                currentModel = viewModel.currentModel,
                currentDelegate = viewModel.currentDelegate,
                poseLandmarkerHelperListener = this
            )
            isPoseModelLoading = false

            activity?.runOnUiThread {
                if (_fragmentCameraBinding != null) {
                    initBottomSheetControls()
                }
            }
        }

        //点击“扫描二维码”
        fragmentCameraBinding.btnScanOtt.setOnClickListener {
                openQrScanner()
            }
    }

    override fun onPairingSucceeded() {
        activity?.runOnUiThread {
            if (_fragmentCameraBinding == null) return@runOnUiThread
            resetFullBodyCalibrationSession()
            switchToFrontCameraForExercise()
            showConnectionSucceededDialog()
        }
    }

    //扫码仍使用后置摄像头；配对成功后将人体检测相机切换为前置摄像头。
    private fun switchToFrontCameraForExercise() {
        if (cameraFacing == CameraSelector.LENS_FACING_FRONT) {
            return
        }
        // 前后相机画面坐标和分辨率可能不同，不能沿用旧滤波状态。
        poseFrameFilter.reset()
        cameraFacing = CameraSelector.LENS_FACING_FRONT
        if (cameraProvider != null && _fragmentCameraBinding != null) { bindCameraUseCases() }
    }

    //扫码完成后立即显示全屏前置相机，连接成功前暂不执行准备动作判断。
    private fun showFullScreenCameraWhileConnecting() {
        if (_fragmentCameraBinding == null) return
        isReadyCheckActive = false
        readyPoseStartedAtMs = null
        switchToFrontCameraForExercise()
        enterImmersiveCameraMode()
        fragmentCameraBinding.overlay.visibility = View.GONE
        fragmentCameraBinding.readyCheckOverlay.visibility = View.VISIBLE
        fragmentCameraBinding.tvReadyCheckHint.text = "正在连接电视..."
    }

    override fun onDisconnected() {
        poseFrameFilter.reset()   //重置滤波器
        activity?.runOnUiThread {
            if (_fragmentCameraBinding == null) return@runOnUiThread
            resetFullBodyCalibrationSession()
            leaveImmersiveCameraMode()
            if (!isOpeningQrAfterDetectionExit) {
                Toast.makeText(requireContext(), "电视连接已断开", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onConnectionError(error: String) {
        poseFrameFilter.reset()
        activity?.runOnUiThread {
            if (_fragmentCameraBinding == null) return@runOnUiThread
            resetFullBodyCalibrationSession()
            leaveImmersiveCameraMode()
            Toast.makeText(requireContext(), "连接失败，请重新扫码", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onPracticeStarted(workoutId: String) {
        Log.i(TAG, "OTT允许开始训练，workoutId=$workoutId")
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) { startFullBodyCalibrationHint() }
        }
    }

    override fun onPracticeStopped() {
        Log.i(TAG, "OTT停止训练")
    }

    override fun onRemoteExitDetection() {
        poseFrameFilter.reset()
        activity?.runOnUiThread {
            if (_fragmentCameraBinding == null) return@runOnUiThread
            resetFullBodyCalibrationSession()
            isOpeningQrAfterDetectionExit = true
            leaveImmersiveCameraMode()
            PoseConnectionManager.finishRemoteExitDetection()
            openQrScanner()
        }
    }

    //检测状态下按返回键：通知OTT并回到扫码页面。
    fun handleBackPressed(): Boolean {
        if (!isImmersiveDetectionMode) return false
        resetFullBodyCalibrationSession()
        isOpeningQrAfterDetectionExit = true
        PoseConnectionManager.exitDetectionFromPhone()
        leaveImmersiveCameraMode()
        openQrScanner()
        return true
    }

    private fun openQrScanner() {
        Log.i(TAG, "Open QR scanner")
        val intent = Intent(requireContext(), QrScanActivity::class.java)
        qrScanLauncher.launch(intent)
    }

    private fun showConnectionSucceededDialog() {
        connectionSuccessDialog?.dismiss()
        connectionSuccessDialog = AlertDialog.Builder(requireContext())
            .setTitle("连接成功")
            .setMessage("即将进行动作检测")
            .setCancelable(false)
            .create()
        connectionSuccessDialog?.show()
        uiHandler.postDelayed({
            connectionSuccessDialog?.dismiss()
            connectionSuccessDialog = null
            enterReadyCheckMode()
        }, 1200L)
    }

    private fun enterReadyCheckMode() {
        if (_fragmentCameraBinding == null) return
        isReadyCheckActive = true
        readyPoseStartedAtMs = null
        enterImmersiveCameraMode()
        fragmentCameraBinding.overlay.visibility = View.GONE
        fragmentCameraBinding.readyCheckOverlay.visibility = View.VISIBLE
        fragmentCameraBinding.readyPoseGuide.visibility = View.VISIBLE
        fragmentCameraBinding.tvReadyCheckHint.text = "请面对手机，举起右手"
    }

    private fun enterImmersiveCameraMode() {
        isImmersiveDetectionMode = true
        isOpeningQrAfterDetectionExit = false
        val fragmentContainer = requireActivity().findViewById<View>(R.id.fragment_container)
        val containerParams = fragmentContainer.layoutParams as ViewGroup.MarginLayoutParams
        if (originalCameraTopMargin == null) {
            originalCameraTopMargin = containerParams.topMargin
        }
        containerParams.topMargin = 0
        fragmentContainer.layoutParams = containerParams
        requireActivity().findViewById<View>(R.id.toolbar)?.visibility = View.GONE
        requireActivity().findViewById<View>(R.id.navigation)?.visibility = View.GONE
        requireActivity().findViewById<View>(R.id.view)?.visibility = View.GONE
        fragmentCameraBinding.bottomSheetLayout.root.visibility = View.GONE
        fragmentCameraBinding.btnScanOtt.visibility = View.GONE
        WindowInsetsControllerCompat(
            requireActivity().window,
            requireActivity().window.decorView
        ).hide(WindowInsetsCompat.Type.systemBars())
    }

    private fun leaveImmersiveCameraMode() {
        isImmersiveDetectionMode = false
        isReadyCheckActive = false
        readyPoseStartedAtMs = null
        fragmentCameraBinding.readyCheckOverlay.visibility = View.GONE
        fragmentCameraBinding.overlay.visibility = View.VISIBLE
        fragmentCameraBinding.bottomSheetLayout.root.visibility = View.VISIBLE
        fragmentCameraBinding.btnScanOtt.visibility = View.VISIBLE
        val fragmentContainer = requireActivity().findViewById<View>(R.id.fragment_container)
        val containerParams = fragmentContainer.layoutParams as ViewGroup.MarginLayoutParams
        originalCameraTopMargin?.let { containerParams.topMargin = it }
        fragmentContainer.layoutParams = containerParams
        requireActivity().findViewById<View>(R.id.toolbar)?.visibility = View.VISIBLE
        requireActivity().findViewById<View>(R.id.navigation)?.visibility = View.VISIBLE
        requireActivity().findViewById<View>(R.id.view)?.visibility = View.VISIBLE
        WindowInsetsControllerCompat(
            requireActivity().window,
            requireActivity().window.decorView
        ).show(WindowInsetsCompat.Type.systemBars())
    }

    private fun updateReadyPoseDetection(result: PoseLandmarkerResult) {
        if (!isReadyCheckActive) return
        if (!isRaiseRightHandPose(result)) {
            readyPoseStartedAtMs = null
            activity?.runOnUiThread {
                if (_fragmentCameraBinding != null) {
                    fragmentCameraBinding.tvReadyCheckHint.text = "请面对手机，举起右手"
                }
            }
            return
        }

        val now = SystemClock.elapsedRealtime()
        val startedAt = readyPoseStartedAtMs ?: now.also { readyPoseStartedAtMs = it }
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {
                fragmentCameraBinding.tvReadyCheckHint.text = "保持动作..."
            }
        }
        if (now - startedAt < 600L) return

        isReadyCheckActive = false
        val sent = PoseConnectionManager.markReadyCheckPassed()
        activity?.runOnUiThread {
            if (_fragmentCameraBinding == null) return@runOnUiThread
            fragmentCameraBinding.tvReadyCheckHint.text =
                if (sent) "动作检测成功" else "连接已断开，请重新扫码"
            uiHandler.postDelayed({
                if (_fragmentCameraBinding != null && sent) {
                    fragmentCameraBinding.readyCheckOverlay.visibility = View.GONE
                }
            }, 800L)
        }
    }

    //OTT开始倒计时后提示用户完整入镜；同一次连接只执行一次。
    private fun startFullBodyCalibrationHint() {
        if (isFullBodyCalibrationFinishedForConnection || isFullBodyCalibrationHintActive) { return }
        isFullBodyCalibrationHintActive = true
        completeBodyCalibrationFrameCount = 0
        fragmentCameraBinding.readyCheckOverlay.visibility = View.VISIBLE
        fragmentCameraBinding.readyCheckOverlay.bringToFront()
        //标定阶段只保留文字提示，不再显示动作检测阶段的举右手示意图。
        fragmentCameraBinding.readyPoseGuide.visibility = View.GONE
        fragmentCameraBinding.tvReadyCheckHint.text = "请确保完整人物进入画面"
        uiHandler.removeCallbacks(calibrationHintTimeoutTask)
        uiHandler.postDelayed(calibrationHintTimeoutTask, CALIBRATION_HINT_DURATION_MS)
    }

    //手机端使用与OTT一致的完整人体规则，连续3帧满足后提前取消提示。
    private fun updateFullBodyCalibrationHint(result: PoseLandmarkerResult) {
        if (!isFullBodyCalibrationHintActive) { return }
        val landmarks = result.landmarks().firstOrNull()
        if (landmarks == null || landmarks.size < 29) {
            completeBodyCalibrationFrameCount = 0
            return
        }

        val requiredIds = intArrayOf(0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
        val isCompleteBody = requiredIds.all { id ->
            val point = landmarks[id]
            val x = point.x()
            val y = point.y()
            x.isFinite() && y.isFinite() &&
                    point.visibility().orElse(0f) >= FULL_BODY_MIN_CONFIDENCE &&
                    point.presence().orElse(0f) >= FULL_BODY_MIN_CONFIDENCE &&
                    x in FULL_BODY_EDGE_MARGIN..(1f - FULL_BODY_EDGE_MARGIN) &&
                    y in FULL_BODY_EDGE_MARGIN..(1f - FULL_BODY_EDGE_MARGIN)
        }

        if (!isCompleteBody) {
            completeBodyCalibrationFrameCount = 0
            return
        }
        completeBodyCalibrationFrameCount++
        if (completeBodyCalibrationFrameCount >= FULL_BODY_CALIBRATION_FRAMES) {
            finishFullBodyCalibrationHint()
        }
    }

    //识别成功或5秒超时都结束提示；超时后OTT会自动使用原始映射。
    private fun finishFullBodyCalibrationHint() {
        if (!isFullBodyCalibrationHintActive) { return }
        isFullBodyCalibrationHintActive = false
        isFullBodyCalibrationFinishedForConnection = true
        completeBodyCalibrationFrameCount = 0
        uiHandler.removeCallbacks(calibrationHintTimeoutTask)
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {
                fragmentCameraBinding.readyCheckOverlay.visibility = View.GONE
            }
        }
    }

    //新连接开始或连接结束时允许下一次重新提示和标定。
    private fun resetFullBodyCalibrationSession() {
        uiHandler.removeCallbacks(calibrationHintTimeoutTask)
        isFullBodyCalibrationHintActive = false
        isFullBodyCalibrationFinishedForConnection = false
        completeBodyCalibrationFrameCount = 0
        if (_fragmentCameraBinding != null) {
            fragmentCameraBinding.readyPoseGuide.visibility = View.VISIBLE
        }
        if (_fragmentCameraBinding != null && !isReadyCheckActive) {
            fragmentCameraBinding.readyCheckOverlay.visibility = View.GONE
        }
    }

    private fun isRaiseRightHandPose(result: PoseLandmarkerResult): Boolean {
        val landmarks = result.landmarks().firstOrNull() ?: return false
        if (landmarks.size < 25) return false

        val nose = landmarks[0]
        //前置相机图像会在推理前镜像，因此真实右侧对应MediaPipe的左侧编号。
        val physicalRightShoulder = landmarks[11]
        val physicalRightElbow = landmarks[13]
        val physicalRightWrist = landmarks[15]

        //准备动作只要求真实右侧上半身可靠，不要求另一只手或髋部入镜。
        val required = listOf(
            nose,
            physicalRightShoulder,
            physicalRightElbow,
            physicalRightWrist
        )
        if (required.any {
                it.visibility().orElse(0f) < 0.65f ||
                    it.presence().orElse(0f) < 0.65f
            }) return false

        return physicalRightWrist.y() < physicalRightShoulder.y() - 0.08f &&
            physicalRightWrist.y() < nose.y() + 0.08f &&
            physicalRightElbow.y() < physicalRightShoulder.y() + 0.15f
    }

    private fun initBottomSheetControls() {
        // init bottom sheet settings
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPoseDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPoseTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinPosePresenceConfidence
            )

        // When clicked, lower pose detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseDetectionConfidence >= 0.2) {
                poseLandmarkerHelper.minPoseDetectionConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseDetectionConfidence <= 0.8) {
                poseLandmarkerHelper.minPoseDetectionConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower pose tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseTrackingConfidence >= 0.2) {
                poseLandmarkerHelper.minPoseTrackingConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose tracking score threshold floor
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPoseTrackingConfidence <= 0.8) {
                poseLandmarkerHelper.minPoseTrackingConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, lower pose presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdMinus.setOnClickListener {
            if (poseLandmarkerHelper.minPosePresenceConfidence >= 0.2) {
                poseLandmarkerHelper.minPosePresenceConfidence -= 0.1f
                updateControlsUi()
            }
        }

        // When clicked, raise pose presence score threshold floor
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdPlus.setOnClickListener {
            if (poseLandmarkerHelper.minPosePresenceConfidence <= 0.8) {
                poseLandmarkerHelper.minPosePresenceConfidence += 0.1f
                updateControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference.
        // Current options are CPU and GPU
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            viewModel.currentDelegate, false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {
                    try {
                        //Spinner首次绑定也会回调；选项没有变化时禁止重复重建模型。
                        if (poseLandmarkerHelper.currentDelegate == p2) return
                        poseLandmarkerHelper.currentDelegate = p2
                        updateControlsUi()
                    } catch(e: UninitializedPropertyAccessException) {
                        Log.e(TAG, "PoseLandmarkerHelper has not been initialized yet.")
                    }
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                }
            }

        // When clicked, change the underlying model used for object detection
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.setSelection(
            viewModel.currentModel,
            false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerModel.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?,
                    p1: View?,
                    p2: Int,
                    p3: Long
                ) {
                    //Spinner首次绑定也会回调；选项没有变化时禁止重复重建模型。
                    if (poseLandmarkerHelper.currentModel == p2) return
                    poseLandmarkerHelper.currentModel = p2
                    updateControlsUi()
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                }
            }
    }

    // Update the values displayed in the bottom sheet. Reset Poselandmarker
    // helper.
    private fun updateControlsUi() {
        if(this::poseLandmarkerHelper.isInitialized) {
            fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPoseDetectionConfidence
                )
            fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPoseTrackingConfidence
                )
            fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
                String.format(
                    Locale.US,
                    "%.2f",
                    poseLandmarkerHelper.minPosePresenceConfidence
                )

            if (isPoseModelLoading) { return }
            isPoseModelLoading = true
            backgroundExecutor.execute {

                //调整模型、阈值或CPU/GPU时，只重新创建MediaPipe实例，不生成CSV。
                poseLandmarkerHelper.clearPoseLandmarker(
                    saveCsv = false
                )
                poseLandmarkerHelper.setupPoseLandmarker()
                isPoseModelLoading = false
            }
            fragmentCameraBinding.overlay.clear()
        }
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            }, ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {

        // CameraProvider
        val cameraProvider = cameraProvider
            ?: throw IllegalStateException("Camera initialization failed.")

        val cameraSelector = CameraSelector.Builder().requireLensFacing(cameraFacing).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer = ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                // The analyzer can then be assigned to the instance
                .also {
                    it.setAnalyzer(backgroundExecutor) { image ->
                        posePerfCameraFrameCount++
                        posePerfCameraWidth = image.width
                        posePerfCameraHeight = image.height
                        detectPose(image)
                    }
                }

        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            // A variable number of use-cases can be passed here -
            // camera provides access to CameraControl & CameraInfo
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )

            // Attach the viewfinder's surface provider to preview use case
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun detectPose(imageProxy: ImageProxy) {
        //模型后台加载期间CameraX仍负责预览；分析帧必须及时关闭，不能堵塞相机管线。
        if (!this::poseLandmarkerHelper.isInitialized ||
            isPoseModelLoading ||
            poseLandmarkerHelper.isClose()
        ) {
            imageProxy.close()
            return
        }
        poseLandmarkerHelper.detectLiveStream(
            imageProxy = imageProxy,
            isFrontCamera = cameraFacing == CameraSelector.LENS_FACING_FRONT
        )
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation =
            fragmentCameraBinding.viewFinder.display.rotation
    }

    // Update UI after pose have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
    override fun onResults(
        resultBundle: PoseLandmarkerHelper.ResultBundle
    ) {
        val result = resultBundle.results.firstOrNull()

        posePerfInferenceCount++
        posePerfInferenceTotalMs += resultBundle.inferenceTime
        posePerfInferenceMaxMs = maxOf(posePerfInferenceMaxMs, resultBundle.inferenceTime)
        if (result != null && result.landmarks().isNotEmpty()) {
            posePerfValidPersonCount++
        }

        if (result != null) {
            updateReadyPoseDetection(result)
            updateFullBodyCalibrationHint(result)

            //只有在OTT选择了健身视频后开始发送pose
            if (PoseConnectionManager.poseStreamingEnabled) {
                // 1. MediaPipe -> rawPoseFrame
                val rawPoseFrame = PoseFrameConverter.convert(result = result, frameId = poseFrameId++, imageWidth = resultBundle.inputImageWidth, imageHeight = resultBundle.inputImageHeight)

                // 发送给OTT前进行One Euro滤波。
                val filteredPoseFrame = poseFrameFilter.filter(rawPoseFrame)

                // OTT收到的是滤波后的坐标。
                val json = PoseJsonEncoder.encode(filteredPoseFrame)

                if (PoseConnectionManager.isConnected()
                ) {
                    posePerfSendAttemptCount++
                    val success = PoseConnectionManager.sendPose(json)
                    if (success) {
                        posePerfSendSuccessCount++
                    }
                }
            } else {//这一等级对应if (poseStreamingEnabled)的否定状态
                //当前只是connected，mediapipe运行，但是不进行参数解析传输等操作
            }
        }

        // 手机自己的界面姿态绘制代码，不要删除
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {
                fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                    String.format("%d ms", resultBundle.inferenceTime)

                // Pass necessary information to OverlayView for drawing on the canvas
                fragmentCameraBinding.overlay.setResults(
                    resultBundle.results.first(),
                    resultBundle.inputImageHeight,
                    resultBundle.inputImageWidth,
                    RunningMode.LIVE_STREAM
                )
                // Force a redraw
                fragmentCameraBinding.overlay.invalidate()
            }
        }

        reportPhonePosePerformance()
    }

    //每两秒输出一次手机端识别与发送性能，统一使用POSE_PERF_PHONE标签。
    private fun reportPhonePosePerformance() {
        val nowMs = SystemClock.elapsedRealtime()
        val elapsedMs = nowMs - posePerfWindowStartedAtMs
        if (elapsedMs < 2_000L) { return }

        val seconds = elapsedMs / 1_000f
        val averageInferenceMs = if (posePerfInferenceCount == 0L) 0f
        else posePerfInferenceTotalMs.toFloat() / posePerfInferenceCount

        Log.i(
            "POSE_PERF_PHONE",
            "PHONE camera=${"%.1f".format(posePerfCameraFrameCount / seconds)}fps, " +
                    "input=${posePerfCameraWidth}x${posePerfCameraHeight}, " +
                    "inference=${"%.1f".format(posePerfInferenceCount / seconds)}fps, " +
                    "valid=${"%.1f".format(posePerfValidPersonCount / seconds)}fps, " +
                    "sendAttempt=${"%.1f".format(posePerfSendAttemptCount / seconds)}fps, " +
                    "sendSuccess=${"%.1f".format(posePerfSendSuccessCount / seconds)}fps, " +
                    "inferAvg=${"%.1f".format(averageInferenceMs)}ms, " +
                    "inferMax=${posePerfInferenceMaxMs}ms, " +
                    "streaming=${PoseConnectionManager.poseStreamingEnabled}, " +
                    "delegate=${if (poseLandmarkerHelper.currentDelegate == PoseLandmarkerHelper.DELEGATE_GPU) "GPU" else "CPU"}, " +
                    "model=${when (poseLandmarkerHelper.currentModel) {
                        PoseLandmarkerHelper.MODEL_POSE_LANDMARKER_FULL -> "FULL"
                        PoseLandmarkerHelper.MODEL_POSE_LANDMARKER_LITE -> "LITE"
                        PoseLandmarkerHelper.MODEL_POSE_LANDMARKER_HEAVY -> "HEAVY"
                        else -> "UNKNOWN"
                    }}"
        )

        posePerfCameraFrameCount = 0L
        posePerfInferenceCount = 0L
        posePerfValidPersonCount = 0L
        posePerfSendAttemptCount = 0L
        posePerfSendSuccessCount = 0L
        posePerfInferenceTotalMs = 0L
        posePerfInferenceMaxMs = 0L
        posePerfWindowStartedAtMs = nowMs
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
            if (errorCode == PoseLandmarkerHelper.GPU_ERROR) {
                fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                    PoseLandmarkerHelper.DELEGATE_CPU, false
                )
            }
        }
    }
}
