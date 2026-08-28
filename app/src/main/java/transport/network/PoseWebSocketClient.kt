package com.google.mediapipe.examples.poselandmarker.transport.network

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener

class PoseWebSocketClient(
    private val listener: ConnectionListener
) {
    companion object {
        private const val TAG = "PoseWsClient"
    }

    interface ConnectionListener {
        fun onConnected()
        fun onDisconnected()
        fun onMessage(message: String)
        fun onError(error: String)
    }

    //先不要加pingInterval，先把最基本的长连接稳定性跑通。
    private val client = OkHttpClient.Builder().build()
    private var webSocket: WebSocket? = null
    @Volatile
    private var connected = false

    fun connect(
        host: String,
        port: Int
    ) {
        val url = "ws://$host:$port"
        val request = Request.Builder().url(url).build()

        webSocket = client.newWebSocket(request,
                object : WebSocketListener() {

                    override fun onOpen(
                        webSocket: WebSocket,
                        response: Response
                    ) {
                        connected = true
                        listener.onConnected()
                    }
                    override fun onMessage(
                        webSocket: WebSocket,
                        text: String
                    ) {
                        listener.onMessage(text)
                    }

                    override fun onClosing(
                        webSocket: WebSocket,
                        code: Int,
                        reason: String
                    ) {
                        Log.w(TAG, "===== WebSocket CLOSING =====")
                        Log.w(TAG, "code=$code, reason=$reason")
                    }

                    override fun onClosed(
                        webSocket: WebSocket,
                        code: Int,
                        reason: String
                    ) {
                        connected = false
                        Log.w(TAG, "===== WebSocket CLOSED =====")
                        Log.w(TAG, "code=$code, reason=$reason")
                        listener.onDisconnected()
                    }

                    override fun onFailure(
                        webSocket: WebSocket,
                        t: Throwable,
                        response: Response?
                    ) {
                        connected = false
                        Log.e(TAG, "===== WebSocket FAILURE =====")
                        Log.e(TAG, "Exception type = ${t.javaClass.name}")
                        Log.e(TAG, "Exception message = ${t.message}")
                        if (response != null) {
                            Log.e(TAG, "HTTP response code = ${response.code}")
                        }
                        Log.e(TAG, "WebSocket stack trace", t)
                        listener.onError(t.message ?: "Unknown WebSocket error"
                        )
                    }
                }
            )
    }

    fun send(
        message: String
    ): Boolean {
        if (!connected) {
            Log.w(TAG, "send() cancelled because WebSocket is disconnected")
            return false
        }
        val result = webSocket?.send(message) ?: false
        return result
    }

    fun isConnected():
            Boolean {
        return connected
    }

    fun disconnect() {
        connected = false
        webSocket?.close(1000, "Phone disconnect")
        webSocket = null
    }
}
