package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.util.AttributeSet
import android.view.View

class ReadyPoseGuideView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val dimPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(118, 18, 16, 24)
    }
    private val clearPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.TRANSPARENT
        style = Paint.Style.FILL
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    }
    private val outlinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(215, 255, 255, 255)
        style = Paint.Style.STROKE
        strokeWidth = 4f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    init { setLayerType(LAYER_TYPE_SOFTWARE, null) }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val layer = canvas.saveLayer(0f, 0f, width.toFloat(), height.toFloat(), null)
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), dimPaint)

        //参考人物采用较宽的肩膀、躯干和四肢，避免显示成细长火柴人。
        val guideHeight = height * 0.60f
        val guideWidth = guideHeight * 0.70f
        val left = (width - guideWidth) / 2f
        val top = height * 0.14f
        val sx = guideWidth / 240f
        val sy = guideHeight / 360f
        val strokeScale = (sx + sy) / 2f

        fun x(value: Float) = left + value * sx
        fun y(value: Float) = top + value * sy

        //左臂自然下垂，大臂仅比前臂略粗。
        val leftArm = Path().apply {
            moveTo(x(91f), y(91f))
            cubicTo(x(80f), y(91f), x(73f), y(99f), x(70f), y(111f))
            lineTo(x(61f), y(151f))
            lineTo(x(55f), y(184f))
            cubicTo(x(53f), y(194f), x(58f), y(202f), x(66f), y(204f))
            cubicTo(x(74f), y(206f), x(81f), y(200f), x(82f), y(191f))
            lineTo(x(86f), y(160f))
            lineTo(x(98f), y(115f))
            cubicTo(x(101f), y(103f), x(99f), y(94f), x(91f), y(91f))
            close()
        }

        //右臂从圆润肩线向右上方伸展，大臂仅比前臂略粗。
        val rightArm = Path().apply {
            moveTo(x(143f), y(91f))
            cubicTo(x(135f), y(96f), x(134f), y(107f), x(140f), y(115f))
            lineTo(x(149f), y(125f))
            cubicTo(x(155f), y(132f), x(165f), y(132f), x(171f), y(124f))
            lineTo(x(185f), y(104f))
            lineTo(x(211f), y(55f))
            cubicTo(x(216f), y(46f), x(213f), y(37f), x(205f), y(33f))
            cubicTo(x(197f), y(29f), x(189f), y(33f), x(184f), y(42f))
            lineTo(x(159f), y(84f))
            lineTo(x(151f), y(94f))
            cubicTo(x(149f), y(90f), x(146f), y(89f), x(143f), y(91f))
            close()
        }

        //躯干短而圆润，肩部平滑，腰部只轻微收窄。
        val torso = Path().apply {
            moveTo(x(100f), y(82f))
            cubicTo(x(89f), y(83f), x(82f), y(89f), x(80f), y(100f))
            cubicTo(x(78f), y(116f), x(82f), y(138f), x(86f), y(157f))
            lineTo(x(90f), y(178f))
            cubicTo(x(92f), y(188f), x(100f), y(194f), x(110f), y(195f))
            lineTo(x(130f), y(195f))
            cubicTo(x(140f), y(194f), x(148f), y(188f), x(150f), y(178f))
            lineTo(x(154f), y(157f))
            cubicTo(x(158f), y(138f), x(162f), y(116f), x(160f), y(100f))
            cubicTo(x(158f), y(89f), x(151f), y(83f), x(140f), y(82f))
            cubicTo(x(128f), y(80f), x(112f), y(80f), x(100f), y(82f))
            close()
        }

        //腿长大于躯干长度，并由大腿向脚踝轻微收细。
        val leftLeg = Path().apply {
            moveTo(x(96f), y(181f))
            cubicTo(x(89f), y(188f), x(87f), y(199f), x(87f), y(214f))
            lineTo(x(84f), y(326f))
            cubicTo(x(84f), y(339f), x(91f), y(347f), x(101f), y(347f))
            cubicTo(x(111f), y(347f), x(117f), y(339f), x(117f), y(326f))
            lineTo(x(120f), y(190f))
            cubicTo(x(113f), y(184f), x(104f), y(180f), x(96f), y(181f))
            close()
        }
        val rightLeg = Path().apply {
            moveTo(x(144f), y(181f))
            cubicTo(x(151f), y(188f), x(153f), y(199f), x(153f), y(214f))
            lineTo(x(156f), y(326f))
            cubicTo(x(156f), y(339f), x(149f), y(347f), x(139f), y(347f))
            cubicTo(x(129f), y(347f), x(123f), y(339f), x(123f), y(326f))
            lineTo(x(120f), y(190f))
            cubicTo(x(127f), y(184f), x(136f), y(180f), x(144f), y(181f))
            close()
        }

        //合并为一个轮廓，避免肩膀和髋部出现内部连接线。
        val silhouette = Path(leftLeg).apply {
            op(rightLeg, Path.Op.UNION)
            op(leftArm, Path.Op.UNION)
            op(rightArm, Path.Op.UNION)
            op(torso, Path.Op.UNION)
            addCircle(x(120f), y(44f), 27f * strokeScale, Path.Direction.CW)
        }

        outlinePaint.strokeWidth = 3.2f * strokeScale
        canvas.drawPath(silhouette, clearPaint)
        canvas.drawPath(silhouette, outlinePaint)
        canvas.restoreToCount(layer)
    }
}
