package com.s23ultra.focusstack

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.util.stream.IntStream
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Focus stacking motoru (saf Kotlin, NDK/OpenCV yok).
 *
 * 1) Her kare referans (orta) kareye göre scale + X/Y kayma için sub-piksel hizalanır
 *    (kaba->ince Lucas-Kanade / Gauss-Newton, 3 parametre: s, tx, ty).
 * 2) Hafif Gaussian yumuşatma + Modified-Laplacian enerji haritası çıkarılır.
 * 3) Kareler tek tek (streaming) 6 seviyeli (5 Laplacian + alçak geçiren artık) piramitte
 *    enerji ağırlıklarıyla birleştirilir; böylece bellek kullanımı kare sayısından bağımsızdır.
 */
object FocusStackProcessor {

    /** Birleştirme çözünürlüğünün uzun kenarı. OOM olmuyorsa 3072'ye çıkarabilirsin. */
    const val MAX_WORK_SIDE = 2304

    private const val PYRAMID_LEVELS = 6
    private const val EST_MAX_SIDE = 1280
    private val BLUR3 = floatArrayOf(0.25f, 0.5f, 0.25f)
    private val BOX7 = FloatArray(7) { 1f / 7f }

    private class Img(val w: Int, val h: Int, val d: FloatArray)
    private class EstPyr(val k0: Int, val levels: List<Img>)

    private inline fun parFor(n: Int, crossinline body: (Int) -> Unit) {
        IntStream.range(0, n).parallel().forEach { i -> body(i) }
    }

    // ---------------------------------------------------------------- ana giriş

    suspend fun stack(
        files: List<File>,
        rotationDeg: Int,
        onProgress: (done: Int, total: Int) -> Unit
    ): Bitmap = withContext(Dispatchers.Default) {
        require(files.size >= 2) { "En az 2 kare gerekli" }

        val bo = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(files[0].path, bo)
        val sc = min(1f, MAX_WORK_SIDE.toFloat() / max(bo.outWidth, bo.outHeight))
        val W = max(64, (bo.outWidth * sc).roundToInt())
        val H = max(64, (bo.outHeight * sc).roundToInt())
        val n = files.size
        val refIdx = n / 2

        val refPyr = buildEstPyr(decode(files[refIdx], W, H), W, H)

        val sizes = ArrayList<IntArray>()
        var lw = W
        var lh = H
        sizes.add(intArrayOf(lw, lh))
        while (sizes.size < PYRAMID_LEVELS && min(lw, lh) >= 16) {
            lw = (lw + 1) / 2
            lh = (lh + 1) / 2
            sizes.add(intArrayOf(lw, lh))
        }
        val lv = sizes.size
        val num = Array(3) { Array(lv) { k -> FloatArray(sizes[k][0] * sizes[k][1]) } }
        val den = Array(lv) { k -> FloatArray(sizes[k][0] * sizes[k][1]) }

        val dr = FloatArray(W * H)
        val dg = FloatArray(W * H)
        val db = FloatArray(W * H)
        val valid = FloatArray(W * H)
        val yBuf = FloatArray(W * H)
        val chans = arrayOf(dr, dg, db)

        for (i in 0 until n) {
            ensureActive()
            val px = decode(files[i], W, H)

            var s = 1f
            var tx = 0f
            var ty = 0f
            if (i != refIdx) {
                val p = estimate(refPyr, buildEstPyr(px, W, H), W, H)
                s = p[0].toFloat(); tx = p[1].toFloat(); ty = p[2].toFloat()
            }
            warp(px, W, H, s, tx, ty, dr, dg, db, valid)

            parFor(W * H) { j -> yBuf[j] = 0.299f * dr[j] + 0.587f * dg[j] + 0.114f * db[j] }
            val wgt = sharpness(yBuf, W, H)
            parFor(W * H) { j ->
                val e = wgt[j]
                wgt[j] = (e * e * e + 1e-6f) * valid[j]
            }

            // ağırlık Gauss piramidi
            val wp = ArrayList<Img>()
            var wc = Img(W, H, wgt)
            wp.add(wc)
            for (k in 1 until lv) {
                wc = reduce(wc)
                wp.add(wc)
            }
            for (k in 0 until lv) {
                val dk = den[k]
                val wk = wp[k].d
                parFor(dk.size) { j -> dk[j] += wk[j] }
            }

            // kanal başına Laplacian piramit biriktirme
            for (c in 0 until 3) {
                var g = Img(W, H, chans[c])
                for (k in 0 until lv - 1) {
                    val gn = reduce(g)
                    val up = expandTo(gn, g.w, g.h)
                    val nk = num[c][k]
                    val wk = wp[k].d
                    val gd = g.d
                    parFor(nk.size) { j -> nk[j] += wk[j] * (gd[j] - up[j]) }
                    g = gn
                }
                val nk = num[c][lv - 1]
                val wk = wp[lv - 1].d
                val gd = g.d
                parFor(nk.size) { j -> nk[j] += wk[j] * gd[j] }
            }
            onProgress(i + 1, n)
        }

        // normalize + piramidi çöz
        val out = Array(3) { c ->
            for (k in 0 until lv) {
                val a = num[c][k]
                val d = den[k]
                parFor(a.size) { j -> a[j] /= d[j] }
            }
            var cur = Img(sizes[lv - 1][0], sizes[lv - 1][1], num[c][lv - 1])
            for (k in lv - 2 downTo 0) {
                val up = expandTo(cur, sizes[k][0], sizes[k][1])
                val a = num[c][k]
                parFor(a.size) { j -> up[j] += a[j] }
                cur = Img(sizes[k][0], sizes[k][1], up)
            }
            cur.d
        }

        val argb = IntArray(W * H)
        parFor(W * H) { i ->
            val r = out[0][i].roundToInt().coerceIn(0, 255)
            val g = out[1][i].roundToInt().coerceIn(0, 255)
            val b = out[2][i].roundToInt().coerceIn(0, 255)
            argb[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        val bmp = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
        bmp.setPixels(argb, 0, W, 0, 0, W, H)
        if (rotationDeg % 360 != 0) {
            val m = Matrix().apply { postRotate(rotationDeg.toFloat()) }
            val rotated = Bitmap.createBitmap(bmp, 0, 0, W, H, m, true)
            bmp.recycle()
            rotated
        } else {
            bmp
        }
    }

    // ---------------------------------------------------------------- yardımcılar

    private fun decode(file: File, outW: Int, outH: Int): IntArray {
        val bo = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.path, bo)
        var sample = 1
        while (bo.outWidth / (sample * 2) >= outW && bo.outHeight / (sample * 2) >= outH) sample *= 2
        val o = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bm = BitmapFactory.decodeFile(file.path, o)
            ?: throw IllegalStateException("JPEG çözülemedi: ${file.name}")
        val scaled = if (bm.width == outW && bm.height == outH) bm
        else Bitmap.createScaledBitmap(bm, outW, outH, true)
        if (scaled !== bm) bm.recycle()
        val px = IntArray(outW * outH)
        scaled.getPixels(px, 0, outW, 0, 0, outW, outH)
        scaled.recycle()
        return px
    }

    private fun convSep(src: FloatArray, w: Int, h: Int, k: FloatArray): FloatArray {
        val r = k.size / 2
        val tmp = FloatArray(w * h)
        val out = FloatArray(w * h)
        parFor(h) { y ->
            val o = y * w
            for (x in 0 until w) {
                var s = 0f
                for (i in -r..r) {
                    val xx = min(max(x + i, 0), w - 1)
                    s += k[i + r] * src[o + xx]
                }
                tmp[o + x] = s
            }
        }
        parFor(h) { y ->
            for (x in 0 until w) {
                var s = 0f
                for (i in -r..r) {
                    val yy = min(max(y + i, 0), h - 1)
                    s += k[i + r] * tmp[yy * w + x]
                }
                out[y * w + x] = s
            }
        }
        return out
    }

    /** 5 tap Gauss + 2'ye alt örnekleme. */
    private fun reduce(img: Img): Img {
        val w = img.w
        val h = img.h
        val src = img.d
        val tmp = FloatArray(w * h)
        parFor(h) { y ->
            val o = y * w
            for (x in 0 until w) {
                val a = src[o + max(x - 2, 0)]
                val b = src[o + max(x - 1, 0)]
                val c = src[o + x]
                val d = src[o + min(x + 1, w - 1)]
                val e = src[o + min(x + 2, w - 1)]
                tmp[o + x] = (a + 4f * b + 6f * c + 4f * d + e) / 16f
            }
        }
        val nw = (w + 1) / 2
        val nh = (h + 1) / 2
        val out = FloatArray(nw * nh)
        parFor(nh) { j ->
            val y = j * 2
            val y0 = max(y - 2, 0) * w
            val y1 = max(y - 1, 0) * w
            val y2 = y * w
            val y3 = min(y + 1, h - 1) * w
            val y4 = min(y + 2, h - 1) * w
            for (i in 0 until nw) {
                val x = i * 2
                out[j * nw + i] =
                    (tmp[y0 + x] + 4f * tmp[y1 + x] + 6f * tmp[y2 + x] + 4f * tmp[y3 + x] + tmp[y4 + x]) / 16f
            }
        }
        return Img(nw, nh, out)
    }

    /** Bilinear 2x büyütme (reduce ile aynı piksel hizası: dst x -> src x/2). */
    private fun expandTo(s: Img, dw: Int, dh: Int): FloatArray {
        val out = FloatArray(dw * dh)
        parFor(dh) { y ->
            val fy = y * 0.5f
            val y0 = min(fy.toInt(), s.h - 1)
            val y1 = min(y0 + 1, s.h - 1)
            val ty = fy - y0
            for (x in 0 until dw) {
                val fx = x * 0.5f
                val x0 = min(fx.toInt(), s.w - 1)
                val x1 = min(x0 + 1, s.w - 1)
                val tx = fx - x0
                val a = s.d[y0 * s.w + x0]
                val b = s.d[y0 * s.w + x1]
                val c = s.d[y1 * s.w + x0]
                val d = s.d[y1 * s.w + x1]
                out[y * dw + x] = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
            }
        }
        return out
    }

    /** ISO kumlanmasını azaltan hafif Gauss + Modified Laplacian + yerel ortalama. */
    private fun sharpness(y: FloatArray, w: Int, h: Int): FloatArray {
        val s = convSep(y, w, h, BLUR3)
        val e = FloatArray(w * h)
        parFor(h) { yy ->
            for (x in 0 until w) {
                val c = s[yy * w + x]
                val l = s[yy * w + max(x - 1, 0)]
                val r = s[yy * w + min(x + 1, w - 1)]
                val u = s[max(yy - 1, 0) * w + x]
                val d = s[min(yy + 1, h - 1) * w + x]
                e[yy * w + x] = abs(2f * c - l - r) + abs(2f * c - u - d)
            }
        }
        return convSep(e, w, h, BOX7)
    }

    // ---------------------------------------------------------------- hizalama

    private fun buildEstPyr(px: IntArray, W: Int, H: Int): EstPyr {
        val luma = FloatArray(W * H)
        parFor(H) { y ->
            for (x in 0 until W) {
                val p = px[y * W + x]
                luma[y * W + x] = 0.299f * ((p shr 16) and 255) +
                        0.587f * ((p shr 8) and 255) + 0.114f * (p and 255)
            }
        }
        var cur = Img(W, H, luma)
        var k = 0
        while (max(cur.w, cur.h) > EST_MAX_SIDE) {
            cur = reduce(cur)
            k++
        }
        val levels = ArrayList<Img>()
        levels.add(Img(cur.w, cur.h, convSep(cur.d, cur.w, cur.h, BLUR3)))
        while (max(cur.w, cur.h) > 160) {
            cur = reduce(cur)
            levels.add(Img(cur.w, cur.h, convSep(cur.d, cur.w, cur.h, BLUR3)))
        }
        return EstPyr(k, levels)
    }

    private fun bilin(a: FloatArray, i0: Int, w: Int, fx: Float, fy: Float): Float =
        (a[i0] * (1 - fx) + a[i0 + 1] * fx) * (1 - fy) +
                (a[i0 + w] * (1 - fx) + a[i0 + w + 1] * fx) * fy

    private fun solve3(h: DoubleArray, b: DoubleArray): DoubleArray? {
        val a = arrayOf(
            doubleArrayOf(h[0], h[1], h[2], b[0]),
            doubleArrayOf(h[1], h[3], h[4], b[1]),
            doubleArrayOf(h[2], h[4], h[5], b[2])
        )
        for (i in 0 until 3) {
            var p = i
            for (r in i + 1 until 3) if (abs(a[r][i]) > abs(a[p][i])) p = r
            if (abs(a[p][i]) < 1e-9) return null
            val t = a[i]; a[i] = a[p]; a[p] = t
            for (r in i + 1 until 3) {
                val f = a[r][i] / a[i][i]
                for (c in i until 4) a[r][c] -= f * a[i][c]
            }
        }
        val x = DoubleArray(3)
        for (i in 2 downTo 0) {
            var s = a[i][3]
            for (c in i + 1 until 3) s -= a[i][c] * x[c]
            x[i] = s / a[i][i]
        }
        return x
    }

    /**
     * Referans koordinatı p -> kare koordinatı q = c + s*(p-c) + t modelini
     * kaba->ince Gauss-Newton ile tahmin eder. Dönüş: [s, tx, ty] (çalışma çözünürlüğünde piksel).
     */
    private fun estimate(ref: EstPyr, frm: EstPyr, W: Int, H: Int): DoubleArray {
        var s = 1.0
        var tx = 0.0
        var ty = 0.0
        val n = min(ref.levels.size, frm.levels.size)
        for (j in n - 1 downTo 0) {
            if (j < n - 1) { tx *= 2.0; ty *= 2.0 }
            val k = ref.k0 + j
            val f = (1 shl k).toDouble()
            val cx = (W - 1) / (2.0 * f)
            val cy = (H - 1) / (2.0 * f)
            val R = ref.levels[j]
            val F = frm.levels[j]
            val w = R.w
            val h = R.h
            val gx = FloatArray(w * h)
            val gy = FloatArray(w * h)
            for (y in 1 until h - 1) {
                for (x in 1 until w - 1) {
                    val i = y * w + x
                    gx[i] = (F.d[i + 1] - F.d[i - 1]) * 0.5f
                    gy[i] = (F.d[i + w] - F.d[i - w]) * 0.5f
                }
            }
            val iters = if (j == 0) 8 else 15
            val st = if (w * h > 500_000) 2 else 1
            for (iter in 0 until iters) {
                val hh = DoubleArray(6)
                val bb = DoubleArray(3)
                var cnt = 0
                for (y in 2 until h - 2 step st) {
                    val uy = y - cy
                    for (x in 2 until w - 2 step st) {
                        val ux = x - cx
                        val qx = cx + s * ux + tx
                        val qy = cy + s * uy + ty
                        if (qx < 2.0 || qy < 2.0 || qx > w - 3.0 || qy > h - 3.0) continue
                        val x0 = qx.toInt()
                        val y0 = qy.toInt()
                        val fx = (qx - x0).toFloat()
                        val fy = (qy - y0).toFloat()
                        val i0 = y0 * w + x0
                        val fv = bilin(F.d, i0, w, fx, fy)
                        val gxv = bilin(gx, i0, w, fx, fy).toDouble()
                        val gyv = bilin(gy, i0, w, fx, fy).toDouble()
                        val r = (fv - R.d[y * w + x]).toDouble()
                        val j0 = gxv * ux + gyv * uy
                        hh[0] += j0 * j0; hh[1] += j0 * gxv; hh[2] += j0 * gyv
                        hh[3] += gxv * gxv; hh[4] += gxv * gyv; hh[5] += gyv * gyv
                        bb[0] -= j0 * r; bb[1] -= gxv * r; bb[2] -= gyv * r
                        cnt++
                    }
                }
                if (cnt < 200) break
                hh[0] *= 1.0005; hh[3] *= 1.0005; hh[5] *= 1.0005
                val d = solve3(hh, bb) ?: break
                s += d[0]; tx += d[1]; ty += d[2]
                s = s.coerceIn(0.85, 1.15)
                tx = tx.coerceIn(-0.2 * w, 0.2 * w)
                ty = ty.coerceIn(-0.2 * h, 0.2 * h)
                if (abs(d[0]) < 2e-6 && abs(d[1]) < 2e-3 && abs(d[2]) < 2e-3) break
            }
        }
        val scale = (1 shl ref.k0).toDouble()
        return doubleArrayOf(s, tx * scale, ty * scale)
    }

    /** Kareyi referans koordinatına bilinear örnekler; kare dışı kalan pikseller valid=0. */
    private fun warp(
        src: IntArray, W: Int, H: Int, s: Float, tx: Float, ty: Float,
        dr: FloatArray, dg: FloatArray, db: FloatArray, valid: FloatArray
    ) {
        val cx = (W - 1) / 2f
        val cy = (H - 1) / 2f
        parFor(H) { y ->
            for (x in 0 until W) {
                var qx = cx + s * (x - cx) + tx
                var qy = cy + s * (y - cy) + ty
                var v = 1f
                if (qx < 0f || qy < 0f || qx > W - 1f || qy > H - 1f) {
                    v = 0f
                    qx = qx.coerceIn(0f, W - 1f)
                    qy = qy.coerceIn(0f, H - 1f)
                }
                val x0 = min(qx.toInt(), W - 2)
                val y0 = min(qy.toInt(), H - 2)
                val fx = qx - x0
                val fy = qy - y0
                val p00 = src[y0 * W + x0]
                val p01 = src[y0 * W + x0 + 1]
                val p10 = src[(y0 + 1) * W + x0]
                val p11 = src[(y0 + 1) * W + x0 + 1]
                val w00 = (1 - fx) * (1 - fy)
                val w01 = fx * (1 - fy)
                val w10 = (1 - fx) * fy
                val w11 = fx * fy
                val i = y * W + x
                dr[i] = w00 * ((p00 shr 16) and 255) + w01 * ((p01 shr 16) and 255) +
                        w10 * ((p10 shr 16) and 255) + w11 * ((p11 shr 16) and 255)
                dg[i] = w00 * ((p00 shr 8) and 255) + w01 * ((p01 shr 8) and 255) +
                        w10 * ((p10 shr 8) and 255) + w11 * ((p11 shr 8) and 255)
                db[i] = w00 * (p00 and 255) + w01 * (p01 and 255) +
                        w10 * (p10 and 255) + w11 * (p11 and 255)
                valid[i] = v
            }
        }
    }
}
