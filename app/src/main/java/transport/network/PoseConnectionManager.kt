package com.google.mediapipe.examples.poselandmarker.transport.network

import android.util.Log
import org.json.JSONObject

object PoseConnectionManager :
    PoseWebSocketClient.ConnectionListener {
    private const val TAG = "PoseConnection"

    //WebSocket Client现在属于整个App进程， 不再属于CameraFragment。
    private val client = PoseWebSocketClient(this)

    interface StateListener {
        fun onPairingSucceeded()
        fun onDisconnected()
        fun onConnectionError(error: String)
        fun onPracticeStarted(workoutId: String)
        fun onPracticeStopped()
        fun onRemoteExitDetection()
    }

    @Volatile
    private var stateListener: StateListener? = null

    @Volatile
    private var practiceStartReceived = false

    @Volatile
    private var readyCheckPassed = false
    //OTT有没有要求手机上传Pose。
    @Volatile
    var poseStreamingEnabled = false
        private set

    fun setStateListener(listener: StateListener) {
        stateListener = listener
    }

    fun clearStateListener(listener: StateListener) {
        if (stateListener === listener) {
            stateListener = null
        }
    }

    fun connect(
        host: String,
        port: Int
    ) {
        Log.i(TAG, "Connect requested: " + "$host:$port")
        // 每次新连接默认不上传Pose
        poseStreamingEnabled = false
        practiceStartReceived = false
        readyCheckPassed = false
        client.connect(host = host, port = port)
    }

    fun markReadyCheckPassed(): Boolean {
        if (!client.isConnected()) return false
        readyCheckPassed = true
        poseStreamingEnabled = practiceStartReceived
        return client.send("""{"type":"ready_check_passed"}""")
    }


    fun isConnected():
            Boolean {
        return client.isConnected()
    }

    fun sendPose(
        message: String
    ): Boolean {
        if (!poseStreamingEnabled) {
            return false
        }
        if (!client.isConnected()) {
            return false
        }
        return client.send(message)
    }

    //手机主动退出检测：通知OTT后关闭当前连接。
    fun exitDetectionFromPhone() {
        if (client.isConnected()) {
            client.send("""{"type":"phone_exit_detection","version":1}""")
        }
        resetPracticeState()
        client.disconnect()
    }

    //OTT主动退出检测后结束本次连接。
    fun finishRemoteExitDetection() {
        resetPracticeState()
        client.disconnect()
    }

    private fun resetPracticeState() {
        poseStreamingEnabled = false
        practiceStartReceived = false
        readyCheckPassed = false
    }


    override fun onConnected() {
        Log.i(TAG, "OTT connected")

        //WebSocket连接成功≠已开始训练。
        poseStreamingEnabled = false

        val hello =
            """
            {
              "type":"hello",
              "device":"android_phone",
              "version":1
            }
            """.trimIndent()
        val success = client.send(hello)
        Log.i(TAG, "Send HELLO = $success")
    }

    override fun onDisconnected() {
        poseStreamingEnabled = false
        practiceStartReceived = false
        readyCheckPassed = false
        Log.i(TAG, "OTT disconnected")
        stateListener?.onDisconnected()
    }

    override fun onMessage(
        message: String
    ) {
        Log.i(TAG, "OTT message: $message")
        try {
            val json = JSONObject(message)
            when (json.optString("type")
            ) {
                "hello_ack" -> {
                    Log.i(TAG, "Received HELLO_ACK")
                    poseStreamingEnabled = false
                    stateListener?.onPairingSucceeded()
                }
                "practice_start" -> {
                    val workoutId = json.optString("workout_id", "")
                    practiceStartReceived = true
                    poseStreamingEnabled = readyCheckPassed
                    Log.i(TAG, "Received PRACTICE_START, " + "workoutId=$workoutId")
                    stateListener?.onPracticeStarted(workoutId)
                }
                "practice_stop" -> {
                    practiceStartReceived = false
                    poseStreamingEnabled = false
                    Log.i(TAG, "Received PRACTICE_STOP")
                    stateListener?.onPracticeStopped()
                }
                "exit_detection" -> {
                    Log.i(TAG, "Received EXIT_DETECTION")
                    resetPracticeState()
                    stateListener?.onRemoteExitDetection()
                }
                else -> {
                    Log.w(TAG, "Unknown OTT message")
                }
            }
        } catch (
            e: Exception
        ) {
            Log.e(TAG, "Parse OTT message failed", e)
        }
    }


    override fun onError(
        error: String
    ) {
        poseStreamingEnabled = false
        Log.e(TAG, "WebSocket error: $error")
        stateListener?.onConnectionError(error)
    }
}
