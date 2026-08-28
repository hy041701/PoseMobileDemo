package com.google.mediapipe.examples.poselandmarker.transport.model

data class PoseLandmarkData(

    // MediaPipe关键点编号：0~32
    val id: Int,

    // 图像归一化坐标
    val x: Float,
    val y: Float,
    val z: Float,

    // 世界坐标
    val worldX: Float,
    val worldY: Float,
    val worldZ: Float,

    // 可见性
    val visibility: Float,

    // 关键点存在概率
    val presence: Float
)