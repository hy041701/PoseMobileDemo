package com.google.mediapipe.examples.poselandmarker.transport

import com.google.mediapipe.examples.poselandmarker.transport.model.PoseFrame
import com.google.mediapipe.examples.poselandmarker.transport.model.PoseLandmarkData
import com.google.mediapipe.examples.poselandmarker.transport.model.PosePersonData
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

object PoseFrameConverter {
    fun convert(
        result: PoseLandmarkerResult,
        frameId: Long,
        imageWidth: Int,
        imageHeight: Int
    ): PoseFrame {
        val persons =
            mutableListOf<PosePersonData>()

        // MediaPipe的归一化关键点
        val normalizedPoses =
            result.landmarks()

        // MediaPipe的世界坐标关键点
        val worldPoses =
            result.worldLandmarks()

        normalizedPoses.forEachIndexed {   //遍历每一个人，personIndex人的下标
                personIndex,
                normalizedLandmarks ->
            val worldLandmarks =
                worldPoses.getOrNull(     //根据当前的personIndex，去worldPoses中找同一个人的world landmarks
                    personIndex
                )

            val landmarkList =
                normalizedLandmarks.mapIndexed {      //遍历这个人的33个关键点
                        landmarkIndex,
                        landmark ->
                    val worldLandmark =
                        worldLandmarks
                            ?.getOrNull(
                                landmarkIndex
                            )

                    PoseLandmarkData(
                        id =
                            landmarkIndex,
                        x =
                            landmark.x(),
                        y =
                            landmark.y(),
                        z =
                            landmark.z(),
                        worldX =
                            worldLandmark
                                ?.x()
                                ?: 0f,
                        worldY =
                            worldLandmark
                                ?.y()
                                ?: 0f,
                        worldZ =
                            worldLandmark
                                ?.z()
                                ?: 0f,
                        visibility =
                            landmark
                                .visibility()
                                .orElse(0f),
                        presence =
                            landmark
                                .presence()
                                .orElse(0f)
                    )
                }

            persons.add(
                PosePersonData(
                    personId =
                        personIndex,
                    landmarks =
                        landmarkList
                )
            )
        }

        return PoseFrame(
            frameId =
                frameId,
            timestampMs =
                System.currentTimeMillis(),
            imageWidth =
                imageWidth,
            imageHeight =
                imageHeight,
            persons =
                persons
        )
    }
}