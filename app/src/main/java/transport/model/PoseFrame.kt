package com.google.mediapipe.examples.poselandmarker.transport.model

data class PoseFrame(

    // 第几帧
    val frameId: Long,

    // 手机系统时间
    val timestampMs: Long,

    // MediaPipe输入尺寸
    val imageWidth: Int,
    val imageHeight: Int,

    // 当前帧中检测到的人
    val persons:
    List<PosePersonData>
)