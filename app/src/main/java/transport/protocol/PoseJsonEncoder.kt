package com.google.mediapipe.examples.poselandmarker.transport.protocol

import com.google.mediapipe.examples.poselandmarker.transport.model.PoseFrame
import org.json.JSONArray
import org.json.JSONObject

object PoseJsonEncoder {
    fun encode(
        frame: PoseFrame
    ): String {
        val root = JSONObject()
        root.put("type", "pose")
        root.put("version", 1)
        root.put("frame_id", frame.frameId)
        root.put("timestamp_ms", frame.timestampMs)
        root.put("image_width", frame.imageWidth)
        root.put("image_height", frame.imageHeight)

        val personsArray = JSONArray()

        frame.persons.forEach {
                person ->
            val personJson = JSONObject()
            personJson.put("person_id", person.personId)
            val landmarksArray = JSONArray()

            person.landmarks.forEach {
                    landmark ->
                val point = JSONObject()
                point.put("id", landmark.id)
                point.put("x", landmark.x)
                point.put("y", landmark.y)
                point.put("z", landmark.z)
                point.put("world_x", landmark.worldX)
                point.put("world_y", landmark.worldY)
                point.put("world_z", landmark.worldZ)
                point.put("visibility", landmark.visibility)
                point.put("presence", landmark.presence)

                landmarksArray.put(point)
            }

            personJson.put("landmarks", landmarksArray)
            personsArray.put(personJson)
        }
        root.put("persons", personsArray)
        return root.toString()
    }
}