package com.google.mediapipe.examples.poselandmarker.transport.model

data class PosePersonData(

    val personId: Int,

    val landmarks:
    List<PoseLandmarkData>
)