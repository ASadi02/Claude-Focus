package com.s23ultra.focusstack

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.provider.MediaStore
import android.util.Size
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.hypot
import kotlin.math.pow

class MainActivity : Activity() {

    companion object {
        /** Spec: "sonsuz" ucu 0.15 diyoptri. */
        const val FAR_DIOPTER = 0.15f
        /** <1: yakın uçta (yüksek diyoptri) adımlar sıklaşır, uzakta seyrekleşir. */
        const val GAMMA = 0.85f
        /** Odak komutundan sonra VCM titreşiminin durması için bekleme (120-150 ms). */
        const val SETTLE_MS = 150L
        const val OUT_DIR = "Pictures/S23Ultra_FocusStack"
    }

    private class Lens(
        val label: String,
        val openId: String,
        val physicalId: String?,
        val chars: CameraCharacteristics,
        val minFocus: Float
    )

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private lateinit var cm: CameraManager
    private lateinit var textureView: TextureView
    private lateinit var statusView: TextView
    private lateinit var countView: TextView
    private lateinit var btn06: Button
    private lateinit var btn1: Button
    private lateinit var shutter: Button
    private lateinit var bgThread: HandlerThread
    private lateinit var bg: Handler

    private val lenses = LinkedHashMap<String, Lens>()
    private var selected = "1x"
    private var frameCount = 12

    private var camera: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var previewSurface: Surface? = null
    private var camJob: Job? = null
    private var busy = false

    @Volatile private var lastResult: TotalCaptureResult? = null
    @Volatile private var pending: CompletableDeferred<File>? = null

    // ------------------------------------------------------------ yaşam döngüsü

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        cm = getSystemService(CAMERA_SERVICE) as CameraManager
        bgThread = HandlerThread("cam-bg").also { it.start() }
        bg = Handler(bgThread.looper)
        buildUi()
        discoverLenses()
        updateLensUi()
    }

    override fun onResume() {
        super.onResume()
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            if (textureView.isAvailable) startCamera()
        } else {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
        }
    }

    override fun onPause() {
        camJob?.cancel()
        closeCamera()
        super.onPause()
    }

    override fun onDestroy() {
        scope.cancel()
        bgThread.quitSafely()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            if (textureView.isAvailable) startCamera()
        } else {
            status("Kamera izni verilmedi.")
        }
    }

    // ------------------------------------------------------------ arayüz

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun buildUi() {
        val sw = resources.displayMetrics.widthPixels
        val root = FrameLayout(this)
        root.setBackgroundColor(Color.BLACK)

        textureView = TextureView(this)
        root.addView(
            textureView,
            FrameLayout.LayoutParams(sw, sw * 4 / 3, Gravity.TOP or Gravity.CENTER_HORIZONTAL)
        )
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(st: SurfaceTexture, w: Int, h: Int) = startCamera()
            override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, w: Int, h: Int) {}
            override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean = true
            override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}
        }

        statusView = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 13f
            setPadding(dp(12), dp(8), dp(12), dp(8))
            setBackgroundColor(0x66000000)
        }
        root.addView(
            statusView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP
            )
        )

        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), dp(24))
            setBackgroundColor(0xAA000000.toInt())
        }
        countView = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 15f
            text = "Kare sayısı: $frameCount"
        }
        val seek = SeekBar(this).apply {
            max = 20 // 5..25
            progress = frameCount - 5
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    frameCount = p + 5
                    countView.text = "Kare sayısı: $frameCount"
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        btn06 = Button(this).apply { text = "0.6x"; setOnClickListener { selectLens("0.6x") } }
        btn1 = Button(this).apply { text = "1x"; setOnClickListener { selectLens("1x") } }
        shutter = Button(this).apply { text = "ÇEK"; setOnClickListener { onShutter() } }
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        val lp = { LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f) }
        row.addView(btn06, lp())
        row.addView(shutter, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.4f))
        row.addView(btn1, lp())

        panel.addView(countView)
        panel.addView(seek)
        panel.addView(row)
        root.addView(
            panel,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM
            )
        )
        setContentView(root)
    }

    private fun status(msg: String) {
        runOnUiThread { statusView.text = msg }
    }

    private fun toast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
    }

    private fun updateLensUi() {
        btn06.isEnabled = lenses.containsKey("0.6x")
        btn1.isEnabled = lenses.containsKey("1x")
        btn06.alpha = if (selected == "0.6x") 1f else 0.55f
        btn1.alpha = if (selected == "1x") 1f else 0.55f
    }

    private fun selectLens(key: String) {
        if (busy || !lenses.containsKey(key)) return
        selected = key
        updateLensUi()
        startCamera()
    }

    // ------------------------------------------------------------ lens keşfi

    private fun discoverLenses() {
        val found = LinkedHashMap<String, Lens>()

        fun score(l: Lens) = (if (l.minFocus > 0f) 1000 else 0) + (if (l.physicalId == null) 1 else 0)

        fun consider(openId: String, physId: String?, ch: CameraCharacteristics) {
            if (ch.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_BACK) return
            val f = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: return
            val sz = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return
            val eq = f * 43.27f / hypot(sz.width, sz.height) // 35mm eşdeğer odak uzaklığı
            val key = when {
                eq in 11f..17f -> "0.6x"
                eq in 20f..29f -> "1x"
                else -> return
            }
            val minF = ch.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f
            val cand = Lens(key, openId, physId, ch, minF)
            val old = found[key]
            if (old == null || score(cand) > score(old)) found[key] = cand
        }

        var firstBackLogical: Pair<String, CameraCharacteristics>? = null
        for (id in cm.cameraIdList) {
            val ch = try { cm.getCameraCharacteristics(id) } catch (e: Exception) { continue }
            if (ch.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_BACK) continue
            val phys = ch.physicalCameraIds
            if (phys.isEmpty()) {
                consider(id, null, ch)           // doğrudan açılabilen kamera
            } else {
                if (firstBackLogical == null) firstBackLogical = Pair(id, ch)
                for (p in phys) {                 // mantıksal kameranın fiziksel lensleri
                    try { consider(id, p, cm.getCameraCharacteristics(p)) } catch (e: Exception) { }
                }
            }
        }
        if (!found.containsKey("1x") && firstBackLogical != null) {
            val (id, ch) = firstBackLogical
            found["1x"] = Lens("1x", id, null, ch, ch.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE) ?: 0f)
        }
        lenses.clear()
        lenses.putAll(found)
        if (!lenses.containsKey(selected) && lenses.isNotEmpty()) selected = lenses.keys.first()
    }

    // ------------------------------------------------------------ kamera

    private fun closeCamera() {
        try { session?.close() } catch (e: Exception) { }
        try { camera?.close() } catch (e: Exception) { }
        try { reader?.close() } catch (e: Exception) { }
        session = null; camera = null; reader = null; previewSurface = null
    }

    private fun startCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        if (!textureView.isAvailable) return
        camJob?.cancel()
        camJob = scope.launch {
            try {
                closeCamera()
                val lens = lenses[selected]
                if (lens == null) {
                    status("0.6x / 1x lens bulunamadı.")
                    return@launch
                }
                val map = lens.chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
                val prev = map.getOutputSizes(SurfaceTexture::class.java)
                    .filter { it.width * 3 == it.height * 4 && it.width <= 1600 }
                    .maxByOrNull { it.width * it.height } ?: Size(1280, 960)
                val jpgAll = map.getOutputSizes(ImageFormat.JPEG)
                val jpg = jpgAll.filter { it.width * 3 == it.height * 4 && it.width * it.height <= 12_600_000 }
                    .maxByOrNull { it.width * it.height }
                    ?: jpgAll.maxByOrNull { it.width * it.height }!!

                val dev = openDevice(lens.openId)
                camera = dev
                val rd = ImageReader.newInstance(jpg.width, jpg.height, ImageFormat.JPEG, 3)
                rd.setOnImageAvailableListener(imageListener, bg)
                reader = rd
                val st = textureView.surfaceTexture!!
                st.setDefaultBufferSize(prev.width, prev.height)
                val ps = Surface(st)
                previewSurface = ps

                val outs = listOf(ps, rd.surface).map { s ->
                    OutputConfiguration(s).also { oc -> lens.physicalId?.let { oc.setPhysicalCameraId(it) } }
                }
                session = createSession(dev, outs)
                session!!.setRepeatingRequest(previewBuilder().build(), repeatCb, bg)

                val cm10 = if (lens.minFocus > 0f) " (≈${(100f / lens.minFocus).toInt()} cm)" else ""
                status(
                    "${lens.label} hazır • min odak: ${"%.1f".format(lens.minFocus)} dpt$cm10 • " +
                            "JPEG ${jpg.width}x${jpg.height}"
                )
                if (lens.minFocus <= 0f) {
                    status("${lens.label}: LENS_INFO_MINIMUM_FOCUS_DISTANCE = 0 (sabit odak). Odak basamaklama yapılamaz.")
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                status("Kamera hatası: ${e.message}")
            }
        }
    }

    @SuppressLint("MissingPermission")
    private suspend fun openDevice(id: String): CameraDevice = suspendCancellableCoroutine { c ->
        cm.openCamera(id, mainExecutor, object : CameraDevice.StateCallback() {
            override fun onOpened(d: CameraDevice) {
                if (c.isActive) c.resume(d) else d.close()
            }
            override fun onDisconnected(d: CameraDevice) {
                d.close()
                if (c.isActive) c.resumeWithException(IllegalStateException("kamera bağlantısı koptu"))
            }
            override fun onError(d: CameraDevice, error: Int) {
                d.close()
                if (c.isActive) c.resumeWithException(IllegalStateException("kamera hatası $error"))
            }
        })
    }

    private suspend fun createSession(
        dev: CameraDevice, outs: List<OutputConfiguration>
    ): CameraCaptureSession = suspendCancellableCoroutine { c ->
        val cfg = SessionConfiguration(
            SessionConfiguration.SESSION_REGULAR, outs, mainExecutor,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(s: CameraCaptureSession) {
                    if (c.isActive) c.resume(s) else s.close()
                }
                override fun onConfigureFailed(s: CameraCaptureSession) {
                    if (c.isActive) c.resumeWithException(IllegalStateException("oturum yapılandırılamadı"))
                }
            })
        dev.createCaptureSession(cfg)
    }

    private fun previewBuilder(): CaptureRequest.Builder {
        val b = camera!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        b.addTarget(previewSurface!!)
        b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
        return b
    }

    private val repeatCb = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            s: CameraCaptureSession, r: CaptureRequest, result: TotalCaptureResult
        ) {
            lastResult = result
        }
    }

    private val imageListener = ImageReader.OnImageAvailableListener { r ->
        val img = r.acquireNextImage() ?: return@OnImageAvailableListener
        try {
            val buf = img.planes[0].buffer
            val bytes = ByteArray(buf.remaining())
            buf.get(bytes)
            val f = File(cacheDir, "fs_${System.nanoTime()}.jpg")
            f.writeBytes(bytes)
            pending?.complete(f)
        } catch (e: Exception) {
            pending?.completeExceptionally(e)
        } finally {
            img.close()
        }
    }

    // ------------------------------------------------------------ seri çekim

    private fun applyBurst(b: CaptureRequest.Builder, diopter: Float) {
        b.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
        b.set(CaptureRequest.LENS_FOCUS_DISTANCE, diopter)
        b.set(CaptureRequest.CONTROL_AE_LOCK, true)
        b.set(CaptureRequest.CONTROL_AWB_LOCK, true)
        b.set(
            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
            CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF
        )
        b.set(
            CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
            CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF
        )
    }

    private fun focusSteps(n: Int, near: Float): FloatArray = FloatArray(n) { i ->
        val t = i / (n - 1f)
        FAR_DIOPTER + (near - FAR_DIOPTER) * (1f - t).pow(GAMMA) // i=0 en yakın
    }

    private fun onShutter() {
        if (busy) return
        val lens = lenses[selected] ?: return
        if (lens.minFocus <= 0f) {
            toast("Seçili lens sabit odaklı; odak basamaklama yapılamaz.")
            return
        }
        busy = true
        shutter.isEnabled = false
        btn06.isEnabled = false
        btn1.isEnabled = false
        scope.launch {
            val files = ArrayList<File>()
            try {
                runBurst(lens, files)
                // Preview'ı normale döndür
                try { session?.setRepeatingRequest(previewBuilder().build(), repeatCb, bg) } catch (e: Exception) { }

                status("Birleştiriliyor (hizalama + piramit)…")
                val rotation = lens.chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
                val bmp = FocusStackProcessor.stack(files, rotation) { d, t ->
                    status("Birleştiriliyor: kare $d / $t")
                }
                status("Galeriye kaydediliyor…")
                val name = withContext(Dispatchers.IO) { saveToGallery(bmp) }
                bmp.recycle()
                status("Kaydedildi: $OUT_DIR/$name")
                toast("Kaydedildi: $OUT_DIR/$name")
            } catch (e: CancellationException) {
                throw e
            } catch (e: Throwable) {
                status("Hata: ${e.message}")
                toast("Hata: ${e.message}")
            } finally {
                files.forEach { it.delete() }
                busy = false
                shutter.isEnabled = true
                updateLensUi()
            }
        }
    }

    private suspend fun runBurst(lens: Lens, files: MutableList<File>) {
        val sess = session ?: throw IllegalStateException("oturum yok")
        val dev = camera ?: throw IllegalStateException("kamera yok")
        val rd = reader ?: throw IllegalStateException("reader yok")
        val n = frameCount
        val steps = focusSteps(n, lens.minFocus)

        // 1) Pozlama / beyaz dengesi yakınsasın, sonra kilitle
        status("Pozlama kilitleniyor…")
        var waited = 0
        while (waited < 2500) {
            val st = lastResult?.get(CaptureResult.CONTROL_AE_STATE)
            if (st == CaptureResult.CONTROL_AE_STATE_CONVERGED || st == CaptureResult.CONTROL_AE_STATE_LOCKED) break
            delay(100); waited += 100
        }
        val lockB = previewBuilder()
        applyBurst(lockB, steps[0])
        sess.setRepeatingRequest(lockB.build(), repeatCb, bg)
        delay(400)

        // 2) Odak basamakları
        for (i in 0 until n) {
            status("Çekiliyor ${i + 1}/$n  (odak ${"%.2f".format(steps[i])} dpt)")
            val rb = previewBuilder()
            applyBurst(rb, steps[i])
            sess.setRepeatingRequest(rb.build(), repeatCb, bg)

            delay(SETTLE_MS)
            var w = 0
            while (w < 400 && lastResult?.get(CaptureResult.LENS_STATE) == CaptureResult.LENS_STATE_MOVING) {
                delay(30); w += 30
            }

            val cb = dev.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
            cb.addTarget(rd.surface)
            applyBurst(cb, steps[i])
            cb.set(CaptureRequest.JPEG_QUALITY, 100.toByte())
            cb.set(CaptureRequest.JPEG_ORIENTATION, 0)

            val d = CompletableDeferred<File>()
            pending = d
            sess.capture(cb.build(), null, bg)
            files.add(withTimeout(10_000) { d.await() })
        }
        pending = null
    }

    // ------------------------------------------------------------ kayıt

    private fun saveToGallery(bmp: Bitmap): String {
        val name = "FocusStack_" + SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date()) + ".jpg"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, OUT_DIR)
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw IllegalStateException("MediaStore kaydı oluşturulamadı")
        contentResolver.openOutputStream(uri)!!.use { bmp.compress(Bitmap.CompressFormat.JPEG, 97, it) }
        values.clear()
        values.put(MediaStore.Images.Media.IS_PENDING, 0)
        contentResolver.update(uri, values, null, null)
        return name
    }
}
