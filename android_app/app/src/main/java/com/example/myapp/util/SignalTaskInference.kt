package com.example.myapp.util

import android.content.Context
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

data class WindowPrediction(
    val segmentIndex: Int,
    val startSec: Double,
    val endSec: Double,
    val label: String,
    val confidence: Float
)

data class FolderInferenceResult(
    val success: Boolean,
    val message: String,
    val predictionsByFile: Map<String, List<WindowPrediction>> = emptyMap()
)

enum class InferenceTask(
    val windowSec: Double,
    val inputChannels: Int,
    val targetPoints: Int,
    val labels: List<String>,
    val adaptiveBoundaries: Boolean,
    val boundarySearchRadius: Double
) {
    Digits(
        windowSec = 30.0,
        inputChannels = 1,
        targetPoints = 300,
        labels = (0..9).map { it.toString() },
        adaptiveBoundaries = true,
        boundarySearchRadius = 2.0
    ),
    Letters(
        windowSec = 30.0,
        inputChannels = 2,
        targetPoints = 300,
        labels = ('A'..'Z').map { it.toString() },
        adaptiveBoundaries = true,
        boundarySearchRadius = 2.0
    ),
    Health(
        windowSec = 60.0,
        inputChannels = 2,
        targetPoints = 300,
        labels = listOf("正常", "咳嗽", "锻炼", "鼻塞", "鼻子呼吸"),
        adaptiveBoundaries = false,
        boundarySearchRadius = 2.0
    )
}

private data class RawRow(
    val timeSec: Double,
    val ch1: Double,
    val ch2: Double
)

private data class DualSignalFrame(
    val timeSec: DoubleArray,
    val ch1: DoubleArray,
    val ch2: DoubleArray
)

private data class SegmentTensor(
    val segmentIndex: Int,
    val startSec: Double,
    val endSec: Double,
    val inputCHW: FloatArray
)

private data class HeaderColumns(
    val timeCol: Int,
    val ch1DeltaCol: Int,
    val ch2DeltaCol: Int
)

fun inferFolderWithPtl(
    context: Context,
    folder: String,
    modelPath: String,
    task: InferenceTask
): FolderInferenceResult {
    return try {
        val files = context.assets
            .list(folder)
            ?.filter { it.lowercase().endsWith(".csv") }
            ?.sorted()
            .orEmpty()
        if (files.isEmpty()) {
            return FolderInferenceResult(false, "未在 $folder 找到可推理的 CSV 文件。")
        }

        val predictions = linkedMapOf<String, List<WindowPrediction>>()
        for (fileName in files) {
            val path = "$folder/$fileName"
            val content = context.assets.open(path).bufferedReader().use { it.readText() }
            val frame = parseDualSignalCsv(content)
            if (frame == null) {
                predictions[fileName] = emptyList()
                continue
            }
            val segments = buildSegments(frame, task)
            val perFilePred = ArrayList<WindowPrediction>(segments.size)
            for (seg in segments) {
                val logits = runPtlInference(
                    context = context,
                    modelPath = modelPath,
                    inputCHW = seg.inputCHW,
                    channels = task.inputChannels,
                    targetPoints = task.targetPoints
                )
                val probs = softmax(logits)
                val bestIdx = argmax(probs)
                val label = task.labels.getOrElse(bestIdx) { "未知($bestIdx)" }
                val confidence = probs.getOrElse(bestIdx) { 0f }
                perFilePred.add(
                    WindowPrediction(
                        segmentIndex = seg.segmentIndex,
                        startSec = seg.startSec,
                        endSec = seg.endSec,
                        label = label,
                        confidence = confidence
                    )
                )
            }
            predictions[fileName] = perFilePred
        }

        val totalSegments = predictions.values.sumOf { it.size }
        if (totalSegments == 0) {
            return FolderInferenceResult(
                success = false,
                message = "未生成任何有效分段，请检查输入数据或模型配置。"
            )
        }
        FolderInferenceResult(
            success = true,
            message = "推理完成: 文件 ${predictions.size} 个，窗口 $totalSegments 段。",
            predictionsByFile = predictions
        )
    } catch (e: Exception) {
        FolderInferenceResult(
            success = false,
            message = "推理失败: ${e.message ?: "未知错误"}"
        )
    }
}

private fun parseDualSignalCsv(content: String): DualSignalFrame? {
    val lines = content.lineSequence()
        .map { it.trim() }
        .filter { it.isNotEmpty() }
        .toList()
    if (lines.isEmpty()) return null

    val rows = parseByHeader(lines) ?: parseByFallback(lines)
    if (rows.isEmpty()) return null

    val grouped = linkedMapOf<Double, MutableList<RawRow>>()
    rows.forEach { row ->
        grouped.getOrPut(row.timeSec) { mutableListOf() }.add(row)
    }
    if (grouped.isEmpty()) return null

    val sortedTimes = grouped.keys.sorted()
    val baseTime = sortedTimes.first()
    val time = DoubleArray(sortedTimes.size)
    val ch1 = DoubleArray(sortedTimes.size)
    val ch2 = DoubleArray(sortedTimes.size)
    sortedTimes.forEachIndexed { idx, t ->
        val bucket = grouped[t].orEmpty()
        time[idx] = t - baseTime
        ch1[idx] = meanIgnoreNaN(bucket.map { it.ch1 })
        ch2[idx] = meanIgnoreNaN(bucket.map { it.ch2 })
    }
    return DualSignalFrame(timeSec = time, ch1 = ch1, ch2 = ch2)
}

private fun parseByHeader(lines: List<String>): List<RawRow>? {
    if (lines.size < 3) return null
    val header1 = tokenize(lines[0])
    val header2 = tokenize(lines[1])
    if (header1.isEmpty() || header2.isEmpty()) return null

    val cols = detectHeaderColumns(header1, header2) ?: return null
    val out = mutableListOf<RawRow>()
    for (line in lines.drop(2)) {
        val tokens = tokenize(line)
        val time = parseTokenAsDouble(tokens.getOrNull(cols.timeCol))
        if (time == null) continue
        val ch1 = parseTokenAsDouble(tokens.getOrNull(cols.ch1DeltaCol)) ?: Double.NaN
        val ch2 = parseTokenAsDouble(tokens.getOrNull(cols.ch2DeltaCol)) ?: Double.NaN
        out.add(RawRow(timeSec = time, ch1 = ch1, ch2 = ch2))
    }
    return out
}

private fun parseByFallback(lines: List<String>): List<RawRow> {
    val out = mutableListOf<RawRow>()
    var fallbackTime = 0.0
    for (line in lines) {
        val nums = tokenize(line).mapNotNull { parseTokenAsDouble(it) }
        if (nums.isEmpty()) continue
        val time = if (nums.size >= 2) nums[0] else fallbackTime
        val ch1 = if (nums.size >= 2) nums[1] else nums.last()
        val ch2 = if (nums.size >= 3) nums[2] else ch1
        out.add(RawRow(timeSec = time, ch1 = ch1, ch2 = ch2))
        fallbackTime += 0.06
    }
    return out
}

private fun detectHeaderColumns(header1: List<String>, header2: List<String>): HeaderColumns? {
    val maxSize = max(header1.size, header2.size)
    var timeCol: Int? = null
    var ch1DeltaCol: Int? = null
    var ch2DeltaCol: Int? = null

    fun inferChannel(col: Int): String {
        val direct = normalizeHeader(header1.getOrElse(col) { "" })
        if (direct.startsWith("ch")) return direct.uppercase()
        if (col > 0) {
            val left = normalizeHeader(header1.getOrElse(col - 1) { "" })
            if (left.startsWith("ch")) return left.uppercase()
        }
        return ""
    }

    for (idx in 0 until maxSize) {
        val h2 = normalizeHeader(header2.getOrElse(idx) { "" })
        if (h2.contains("time") && h2.contains("(s)")) {
            timeCol = idx
            continue
        }
        if (h2.contains("r/r0") && h2.contains("%")) {
            when (inferChannel(idx)) {
                "CH1" -> ch1DeltaCol = idx
                "CH2" -> ch2DeltaCol = idx
            }
        }
    }

    return if (timeCol != null && ch1DeltaCol != null && ch2DeltaCol != null) {
        HeaderColumns(timeCol = timeCol!!, ch1DeltaCol = ch1DeltaCol!!, ch2DeltaCol = ch2DeltaCol!!)
    } else {
        null
    }
}

private fun buildSegments(frame: DualSignalFrame, task: InferenceTask): List<SegmentTensor> {
    if (frame.timeSec.isEmpty()) return emptyList()
    val edges = buildSegmentEdges(
        frame = frame,
        windowSec = task.windowSec,
        adaptiveBoundaries = task.adaptiveBoundaries,
        boundarySearchRadius = task.boundarySearchRadius
    )
    if (edges.isEmpty()) return emptyList()

    val out = mutableListOf<SegmentTensor>()
    for (segIdx in edges.indices) {
        val edge = edges[segIdx]
        val startSec = edge.first
        val endSec = edge.second
        val indices = mutableListOf<Int>()
        for (i in frame.timeSec.indices) {
            val t = frame.timeSec[i]
            if (t >= startSec && t < endSec) {
                indices.add(i)
            }
        }
        if (indices.size < 2) continue

        val segTime = DoubleArray(indices.size)
        val segCh1 = DoubleArray(indices.size)
        val segCh2 = DoubleArray(indices.size)
        var nonNa = 0
        indices.forEachIndexed { idx, sourceIdx ->
            segTime[idx] = frame.timeSec[sourceIdx]
            segCh1[idx] = frame.ch1[sourceIdx]
            segCh2[idx] = frame.ch2[sourceIdx]
            if (!segCh1[idx].isNaN()) nonNa += 1
            if (!segCh2[idx].isNaN()) nonNa += 1
        }
        val validRatio = nonNa.toDouble() / (segTime.size * 2.0)
        if (validRatio < 0.6) continue

        val ch1Processed = robustChannelPreprocess(segTime, segCh1) ?: continue
        val ch2Processed = robustChannelPreprocess(segTime, segCh2) ?: continue

        val targetT = DoubleArray(task.targetPoints) { i ->
            startSec + (endSec - startSec) * (i.toDouble() / task.targetPoints.toDouble())
        }
        val ch1Resampled = interpolate1d(targetT, segTime, ch1Processed)
        val ch2Resampled = interpolate1d(targetT, segTime, ch2Processed)

        val selectedChannels = if (task.inputChannels == 1) {
            arrayOf(robustNormalize(ch1Resampled))
        } else {
            arrayOf(robustNormalize(ch1Resampled), robustNormalize(ch2Resampled))
        }

        val input = FloatArray(task.inputChannels * task.targetPoints)
        var cursor = 0
        selectedChannels.forEach { channel ->
            channel.forEach { v ->
                input[cursor] = v.toFloat()
                cursor += 1
            }
        }
        out.add(
            SegmentTensor(
                segmentIndex = segIdx,
                startSec = startSec,
                endSec = endSec,
                inputCHW = input
            )
        )
    }
    return out
}

private fun buildSegmentEdges(
    frame: DualSignalFrame,
    windowSec: Double,
    adaptiveBoundaries: Boolean,
    boundarySearchRadius: Double
): List<Pair<Double, Double>> {
    if (frame.timeSec.isEmpty()) return emptyList()
    val maxTime = frame.timeSec.maxOrNull() ?: return emptyList()
    val offsetSec = 0.0
    if (offsetSec >= maxTime) return emptyList()
    val segmentCount = floor((maxTime - offsetSec) / windowSec).toInt()
    if (segmentCount <= 0) return emptyList()

    return if (adaptiveBoundaries) {
        makeAdaptiveEdges(
            frame = frame,
            offsetSec = offsetSec,
            segmentCount = segmentCount,
            windowSec = windowSec,
            boundarySearchRadius = boundarySearchRadius
        )
    } else {
        makeFixedEdges(offsetSec = offsetSec, segmentCount = segmentCount, windowSec = windowSec)
    }
}

private fun makeFixedEdges(
    offsetSec: Double,
    segmentCount: Int,
    windowSec: Double
): List<Pair<Double, Double>> {
    val out = mutableListOf<Pair<Double, Double>>()
    for (i in 0 until segmentCount) {
        out.add(Pair(offsetSec + i * windowSec, offsetSec + (i + 1) * windowSec))
    }
    return out
}

private fun makeAdaptiveEdges(
    frame: DualSignalFrame,
    offsetSec: Double,
    segmentCount: Int,
    windowSec: Double,
    boundarySearchRadius: Double
): List<Pair<Double, Double>> {
    if (segmentCount <= 1) return makeFixedEdges(offsetSec, segmentCount, windowSec)

    val ch1Processed = robustChannelPreprocess(frame.timeSec, frame.ch1) ?: return makeFixedEdges(offsetSec, segmentCount, windowSec)
    val ch2Processed = robustChannelPreprocess(frame.timeSec, frame.ch2) ?: return makeFixedEdges(offsetSec, segmentCount, windowSec)

    val endTime = offsetSec + segmentCount * windowSec
    val uniformT = mutableListOf<Double>()
    var t = offsetSec
    while (t <= endTime + 1e-6) {
        uniformT.add(t)
        t += 0.1
    }
    if (uniformT.size < 10) return makeFixedEdges(offsetSec, segmentCount, windowSec)

    val ut = uniformT.toDoubleArray()
    val s1 = interpolate1d(ut, frame.timeSec, ch1Processed)
    val s2 = interpolate1d(ut, frame.timeSec, ch2Processed)
    val energy = DoubleArray(ut.size) { idx -> abs(s1[idx]) + abs(s2[idx]) }
    val grad = gradientAbs(energy)

    val boundaries = mutableListOf<Double>()
    boundaries.add(offsetSec)
    val minSeg = 0.65 * windowSec
    for (i in 1 until segmentCount) {
        val nominal = offsetSec + i * windowSec
        val lower = max(nominal - boundarySearchRadius, boundaries.last() + minSeg)
        val remain = (segmentCount - i) * minSeg
        val upper = min(nominal + boundarySearchRadius, endTime - remain)

        var candidate = nominal
        if (upper > lower) {
            var bestLocalIdx = -1
            var bestGrad = Double.POSITIVE_INFINITY
            for (j in ut.indices) {
                val tj = ut[j]
                if (tj >= lower && tj <= upper) {
                    if (grad[j] < bestGrad) {
                        bestGrad = grad[j]
                        bestLocalIdx = j
                    }
                }
            }
            if (bestLocalIdx >= 0) {
                candidate = ut[bestLocalIdx]
            }
        }

        if (candidate <= boundaries.last()) {
            candidate = boundaries.last() + 0.05
        }
        boundaries.add(candidate)
    }
    boundaries.add(endTime)

    val out = mutableListOf<Pair<Double, Double>>()
    for (i in 0 until segmentCount) {
        val startSec = boundaries[i]
        val endSec = boundaries[i + 1]
        if (endSec > startSec) {
            out.add(Pair(startSec, endSec))
        }
    }
    return out
}

private fun robustChannelPreprocess(time: DoubleArray, values: DoubleArray): DoubleArray? {
    val filled = fillChannel(time, values) ?: return null
    val low = percentile(filled, 0.5)
    val high = percentile(filled, 99.5)
    val clipped = if (high > low) {
        DoubleArray(filled.size) { idx -> filled[idx].coerceIn(low, high) }
    } else {
        filled
    }
    val medianFiltered = medianFilter(clipped, window = 5)
    val smooth = movingAverage(medianFiltered, window = 9)
    return detrend(time, smooth)
}

private fun fillChannel(time: DoubleArray, values: DoubleArray): DoubleArray? {
    val validX = mutableListOf<Double>()
    val validY = mutableListOf<Double>()
    for (i in values.indices) {
        val v = values[i]
        if (!v.isNaN()) {
            validX.add(time[i])
            validY.add(v)
        }
    }
    if (validY.isEmpty()) return null
    if (validY.size == 1) {
        return DoubleArray(values.size) { validY.first() }
    }
    val x = validX.toDoubleArray()
    val y = validY.toDoubleArray()
    return interpolate1d(time, x, y)
}

private fun medianFilter(signal: DoubleArray, window: Int): DoubleArray {
    val w = max(1, window)
    val half = w / 2
    val out = DoubleArray(signal.size)
    for (i in signal.indices) {
        val start = max(0, i - half)
        val end = min(signal.lastIndex, i + half)
        val slice = DoubleArray(end - start + 1) { k -> signal[start + k] }
        out[i] = percentile(slice, 50.0)
    }
    return out
}

private fun movingAverage(signal: DoubleArray, window: Int): DoubleArray {
    val w = max(1, window)
    val half = w / 2
    val out = DoubleArray(signal.size)
    for (i in signal.indices) {
        val start = max(0, i - half)
        val end = min(signal.lastIndex, i + half)
        var sum = 0.0
        var count = 0
        for (j in start..end) {
            sum += signal[j]
            count += 1
        }
        out[i] = if (count == 0) 0.0 else sum / count.toDouble()
    }
    return out
}

private fun detrend(time: DoubleArray, signal: DoubleArray): DoubleArray {
    if (signal.size < 3) return signal.copyOf()
    val meanTime = time.average()
    val centeredT = DoubleArray(time.size) { idx -> time[idx] - meanTime }
    val varT = centeredT.map { it * it }.average()
    if (varT < 1e-12) return signal.copyOf()

    val meanSignal = signal.average()
    var cov = 0.0
    for (i in signal.indices) {
        cov += centeredT[i] * (signal[i] - meanSignal)
    }
    cov /= signal.size.toDouble()

    val slope = cov / varT
    val intercept = meanSignal - slope * centeredT.average()
    return DoubleArray(signal.size) { i ->
        signal[i] - (slope * centeredT[i] + intercept)
    }
}

private fun robustNormalize(values: DoubleArray): DoubleArray {
    val median = percentile(values, 50.0)
    val absDev = DoubleArray(values.size) { idx -> abs(values[idx] - median) }
    val mad = percentile(absDev, 50.0)
    val scale = 1.4826 * mad
    if (scale < 1e-6) {
        return DoubleArray(values.size) { 0.0 }
    }
    return DoubleArray(values.size) { idx -> (values[idx] - median) / scale }
}

private fun gradientAbs(values: DoubleArray): DoubleArray {
    if (values.isEmpty()) return DoubleArray(0)
    if (values.size == 1) return doubleArrayOf(0.0)
    val out = DoubleArray(values.size)
    for (i in values.indices) {
        val g = when (i) {
            0 -> values[1] - values[0]
            values.lastIndex -> values[values.lastIndex] - values[values.lastIndex - 1]
            else -> 0.5 * (values[i + 1] - values[i - 1])
        }
        out[i] = abs(g)
    }
    return out
}

private fun interpolate1d(target: DoubleArray, x: DoubleArray, y: DoubleArray): DoubleArray {
    if (x.isEmpty() || y.isEmpty()) return DoubleArray(target.size)
    if (x.size == 1 || y.size == 1) return DoubleArray(target.size) { y.first() }

    val out = DoubleArray(target.size)
    var j = 0
    for (i in target.indices) {
        val t = target[i]
        if (t <= x.first()) {
            out[i] = y.first()
            continue
        }
        if (t >= x.last()) {
            out[i] = y.last()
            continue
        }
        while (j < x.lastIndex - 1 && x[j + 1] < t) {
            j += 1
        }
        val x0 = x[j]
        val x1 = x[j + 1]
        val y0 = y[j]
        val y1 = y[j + 1]
        val ratio = if (x1 == x0) 0.0 else (t - x0) / (x1 - x0)
        out[i] = y0 + ratio * (y1 - y0)
    }
    return out
}

private fun meanIgnoreNaN(values: List<Double>): Double {
    val valid = values.filter { !it.isNaN() }
    return if (valid.isEmpty()) Double.NaN else valid.average()
}

private fun percentile(values: DoubleArray, pct: Double): Double {
    if (values.isEmpty()) return 0.0
    val sorted = values.filter { !it.isNaN() }.sorted()
    if (sorted.isEmpty()) return 0.0
    if (sorted.size == 1) return sorted.first()

    val p = pct.coerceIn(0.0, 100.0) / 100.0
    val rank = p * (sorted.size - 1).toDouble()
    val low = floor(rank).toInt()
    val high = ceil(rank).toInt()
    if (low == high) return sorted[low]
    val weight = rank - low.toDouble()
    return sorted[low] * (1.0 - weight) + sorted[high] * weight
}

private fun tokenize(line: String): List<String> {
    return line.split(',', ';', '\t').map { it.trim() }
}

private fun parseTokenAsDouble(token: String?): Double? {
    if (token == null) return null
    if (token.isBlank()) return null
    if (token.equals("na", ignoreCase = true)) return null
    return token.toDoubleOrNull()
}

private fun normalizeHeader(v: String): String = v.trim().lowercase()

private fun softmax(logits: FloatArray): FloatArray {
    if (logits.isEmpty()) return FloatArray(0)
    var maxLogit = logits[0]
    for (v in logits) {
        if (v > maxLogit) maxLogit = v
    }
    val exps = FloatArray(logits.size)
    var sum = 0.0f
    for (i in logits.indices) {
        val e = kotlin.math.exp(logits[i] - maxLogit)
        exps[i] = e
        sum += e
    }
    if (sum <= 0f) return FloatArray(logits.size)
    return FloatArray(logits.size) { idx -> exps[idx] / sum }
}

private fun argmax(values: FloatArray): Int {
    if (values.isEmpty()) return 0
    var bestIdx = 0
    var bestVal = values[0]
    for (i in 1 until values.size) {
        if (values[i] > bestVal) {
            bestVal = values[i]
            bestIdx = i
        }
    }
    return bestIdx
}
