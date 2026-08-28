package com.google.mediapipe.examples.poselandmarker.pairing

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log

import androidx.appcompat.app.AppCompatActivity

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView

import androidx.core.content.ContextCompat

import com.google.mediapipe.examples.poselandmarker.R

import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage

import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class QrScanActivity : AppCompatActivity() {
    companion object {
        private const val TAG = "QrScan"
    }

    // XML中的摄像头预览区域
    private lateinit var previewView: PreviewView

    // 专门处理二维码图像分析的线程
    private lateinit var cameraExecutor: ExecutorService

    //防止二维码连续识别几十次。一旦成功识别出OTT二维码，就变成true。
    private var scanFinished = false

    //创建ML Kit二维码扫描器。我们明确告诉它：只识别QR_CODE，不需要识别条形码等其他格式。
    private val barcodeScanner by lazy {
        val options = BarcodeScannerOptions
                .Builder()
                .setBarcodeFormats(
                    Barcode.FORMAT_QR_CODE  //表示只识别二维码
                )
                .build()
        BarcodeScanning.getClient(options)
    }

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {
        super.onCreate(savedInstanceState)

        // 加载刚才创建的activity_qr_scan.xml
        setContentView(R.layout.activity_qr_scan)

        // 找到XML中的PreviewView
        previewView = findViewById(R.id.qrPreviewView)

        // 创建一个后台线程
        cameraExecutor = Executors.newSingleThreadExecutor()

        // 开始打开摄像头
        startCamera()
    }

    private fun startCamera() {
        //获取CameraX的CameraProvider。
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()

                // 1. 创建摄像头Preview
                val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider
                            )
                        }

                // 2. 创建ImageAnalysis
                val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(
                            ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                        )
                        .build()

                //CameraX拿到每一帧以后，调用analyzeQrImage()
                imageAnalysis.setAnalyzer(cameraExecutor)
                { imageProxy ->
                        analyzeQrImage(imageProxy)
                    }

                // 3. 使用手机后置摄像头
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                try {
                    // 将：Preview， ImageAnalysis，绑定到当前QrScanActivity。
                    cameraProvider.bindToLifecycle(
                            this,
                            cameraSelector,
                            preview,
                            imageAnalysis
                        )
                } catch (
                    exception: Exception
                ) {
                    Log.e(
                        TAG,
                        "QR camera bind failed",
                        exception
                    )
                }
            },
                ContextCompat.getMainExecutor(
                        this
                    )
            )
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun analyzeQrImage(imageProxy: ImageProxy)
    {
        //已经扫描成功， 就不要继续分析后面的帧。
        if (scanFinished) {
            imageProxy.close()
            return
        }

        //CameraX的ImageProxy中取得原始camera image。
        val mediaImage = imageProxy.image
        if (mediaImage == null) { imageProxy.close()
            return
        }

        //转成ML Kit需要的InputImage。rotationDegrees非常重要，它告诉ML Kit手机当前画面方向。
        val inputImage = InputImage.fromMediaImage(
                    mediaImage,
                    imageProxy
                        .imageInfo
                        .rotationDegrees
                )

        //交给ML Kit识别二维码。
        barcodeScanner.process(
                inputImage
            ).addOnSuccessListener {
                    barcodes ->
                // 如果这一帧识别到了二维码，取第一个二维码的原始字符串。
                val rawValue = barcodes.firstOrNull()?.rawValue

                if (rawValue != null) {
                    handleQrValue(rawValue)
                }
            }
            .addOnFailureListener {
                    exception ->
                Log.e(
                    TAG,
                    "QR scan failed",
                    exception
                )
            }
            .addOnCompleteListener {
                //非常重要：无论成功还是失败，这一帧处理完成后都必须close。
                imageProxy.close()
            }
    }

    private fun handleQrValue(
        qrValue: String
    ) {
        // 已经成功过了，直接返回。
        if (scanFinished) {
            return
        }

        Log.i(
            TAG,
            "Raw QR = $qrValue"
        )

        // 1. 把二维码字符串解析成Uri
        val uri = Uri.parse(qrValue)

        // 2. 判断是不是我们OTT生成的二维码
        if (uri.scheme != "poseott" || uri.host != "pair") {
            Log.w(
                TAG,
                "Not PoseOTT QR"
            )
            return
        }

        // 3. 获取OTT IP
        val host = uri.getQueryParameter("host") ?: return

        // 4. 获取OTT WebSocket端口
        val port = uri.getQueryParameter("port")?.toIntOrNull() ?: return

        // 端口合法性检查
        if (port !in 1..65535
        ) {
            Log.e(
                TAG,
                "Invalid port=$port"
            )
            return
        }

        // 5. 标记扫描完成
        scanFinished = true
        Log.i(
            TAG,
            "OTT host=$host, port=$port"
        )

        // 6. 准备返回给CameraFragment的数据
        val resultIntent = Intent().apply {
                putExtra(
                    "ott_host",
                    host
                )
                putExtra(
                    "ott_port",
                    port
                )
            }

        // 7. 返回成功
        setResult(
            Activity.RESULT_OK,
            resultIntent
        )

        // 8. 关闭扫码页面
        finish()
    }

    override fun onDestroy() {
        //释放ML Kit二维码扫描器。
        barcodeScanner.close()

        //关闭二维码分析线程。
        if (::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
        }
        super.onDestroy()
    }
}