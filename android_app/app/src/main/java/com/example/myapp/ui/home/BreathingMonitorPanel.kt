package com.example.myapp.ui.home

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.myapp.model.MonitorMode
import com.example.myapp.util.loadDemoSignalFiles
import kotlinx.coroutines.delay
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

private data class WaveSample(val timeSec: Double, val value: Float)

private const val HEALTH_DEMO_FILE_WINDOW_SEC = 60.0

@Composable
fun BreathingMonitorPanel(
    mode: MonitorMode,
    signalFolder: String,
    playbackVersion: Int,
    exerciseAnalysisEnabled: Boolean,
    isRecognitionRunning: Boolean,
    modelLabel: String,
    onRateUpdate: (Double) -> Unit,
    onSignalTimeUpdate: (Double) -> Unit,
    onPlaybackCursorUpdate: (String, Double) -> Unit
) {
    val context = LocalContext.current
    val demoFiles = remember(context, signalFolder) {
        val specific = loadDemoSignalFiles(context, signalFolder)
        if (specific.isNotEmpty()) specific else loadDemoSignalFiles(context, "demo_signals")
    }

    val samples = remember { mutableStateListOf<WaveSample>() }
    val peakTimes = remember { mutableStateListOf<Double>() }
    var timeSec by remember { mutableStateOf(0.0) }
    var respiratoryRateBpm by remember { mutableStateOf(0.0) }

    val isHealthDemoPlayback = remember(signalFolder, isRecognitionRunning) {
        signalFolder == "demo_signals/yundong" && isRecognitionRunning
    }

    LaunchedEffect(demoFiles, playbackVersion) {
        samples.clear()
        peakTimes.clear()
        timeSec = 0.0
        respiratoryRateBpm = 0.0

        var fileIndex = 0
        var sampleIndex = 0
        var lastCsvTimeInFile = 0.0

        while (true) {
            val value: Float
            val delayMs: Long
            var currentFileName = "内置模拟信号"
            var currentFileElapsedSec = timeSec

            if (demoFiles.isNotEmpty()) {
                var file = demoFiles[fileIndex]
                if (file.points.isEmpty()) {
                    fileIndex = (fileIndex + 1) % demoFiles.size
                    sampleIndex = 0
                    lastCsvTimeInFile = 0.0
                    continue
                }

                if (isHealthDemoPlayback) {
                    while (true) {
                        file = demoFiles[fileIndex]
                        val firstPointTime = file.points.firstOrNull()?.timeSec ?: 0.0
                        val pointTime = file.points.getOrNull(sampleIndex)?.timeSec ?: firstPointTime
                        val elapsed = (pointTime - firstPointTime).coerceAtLeast(0.0)
                        if (elapsed < HEALTH_DEMO_FILE_WINDOW_SEC) break
                        fileIndex = (fileIndex + 1) % demoFiles.size
                        sampleIndex = 0
                        lastCsvTimeInFile = 0.0
                    }
                    file = demoFiles[fileIndex]
                }

                val point = file.points[sampleIndex]
                currentFileName = file.name
                val firstPointTime = file.points.firstOrNull()?.timeSec ?: point.timeSec
                currentFileElapsedSec = (point.timeSec - firstPointTime).coerceAtLeast(0.0)

                value = point.value
                val deltaSec = if (sampleIndex == 0) 0.06 else (point.timeSec - lastCsvTimeInFile).coerceAtLeast(0.001)
                delayMs = (deltaSec * 1000.0).toLong().coerceIn(1L, 5_000L)
                lastCsvTimeInFile = point.timeSec
                timeSec += deltaSec

                sampleIndex += 1
                val shouldAdvanceFile = sampleIndex >= file.points.size ||
                    (isHealthDemoPlayback && sampleIndex < file.points.size &&
                        ((file.points[sampleIndex].timeSec - firstPointTime).coerceAtLeast(0.0) >= HEALTH_DEMO_FILE_WINDOW_SEC))
                if (shouldAdvanceFile) {
                    sampleIndex = 0
                    lastCsvTimeInFile = 0.0
                    fileIndex = (fileIndex + 1) % demoFiles.size
                }
            } else {
                val deltaSec = 0.06
                delayMs = 60L
                timeSec += deltaSec
                val frequencyHz = 0.25 + 0.04 * sin(timeSec / 10.0)
                value = sin(2 * PI * frequencyHz * timeSec).toFloat()
                currentFileElapsedSec = timeSec
            }

            onPlaybackCursorUpdate(currentFileName, currentFileElapsedSec)
            onSignalTimeUpdate(timeSec)

            samples.add(WaveSample(timeSec, value))
            if (samples.size > 420) samples.removeAt(0)

            if (samples.size >= 3) {
                val a = samples[samples.size - 3]
                val b = samples[samples.size - 2]
                val c = samples[samples.size - 1]
                val isPeak = b.value > a.value && b.value > c.value && b.value > 0.85f
                if (isPeak) {
                    val lastPeak = peakTimes.lastOrNull()
                    if (lastPeak == null || b.timeSec - lastPeak > 1.2) {
                        peakTimes.add(b.timeSec)
                        if (peakTimes.size > 8) peakTimes.removeAt(0)
                        if (peakTimes.size >= 2) {
                            val periods = peakTimes.zipWithNext { p1, p2 -> p2 - p1 }
                            val averagePeriod = periods.average()
                            if (averagePeriod > 0) {
                                val latestRate = 60.0 / averagePeriod
                                if (abs(latestRate - respiratoryRateBpm) > 0.05) {
                                    respiratoryRateBpm = latestRate
                                    onRateUpdate(latestRate)
                                } else {
                                    respiratoryRateBpm = latestRate
                                }
                            }
                        }
                    }
                }
            }

            delay(delayMs)
        }
    }

    if (mode == MonitorMode.Numeric) {
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("数值模式", style = MaterialTheme.typography.titleMedium)
                Text("呼吸频率: ${"%.1f".format(respiratoryRateBpm)} 次/分", style = MaterialTheme.typography.headlineSmall)
                Text("根据波峰周期实时计算。")
                if (exerciseAnalysisEnabled) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("模型识别标签:")
                        RecognitionLabelBadge(labelText = modelLabel)
                    }
                }
            }
        }
    } else {
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("波形模式", style = MaterialTheme.typography.titleMedium)
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(220.dp)
                        .background(Color(0xFFEEF3FF), RoundedCornerShape(8.dp))
                ) {
                    Canvas(modifier = Modifier.fillMaxSize().padding(8.dp)) {
                        if (samples.size < 2) return@Canvas
                        val path = Path()
                        val minTime = samples.first().timeSec
                        val maxTime = samples.last().timeSec
                        val timeSpan = (maxTime - minTime).coerceAtLeast(0.1)

                        samples.forEachIndexed { index, sample ->
                            val x = ((sample.timeSec - minTime) / timeSpan).toFloat() * size.width
                            val y = size.height / 2f - sample.value * (size.height * 0.4f)
                            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
                        }

                        drawLine(
                            color = Color.Gray,
                            start = Offset(0f, size.height / 2f),
                            end = Offset(size.width, size.height / 2f),
                            strokeWidth = 1f
                        )
                        drawPath(path = path, color = Color(0xFF1E64FF), style = androidx.compose.ui.graphics.drawscope.Stroke(width = 3f))
                    }
                }
                if (exerciseAnalysisEnabled) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("模型识别标签:")
                        RecognitionLabelBadge(labelText = modelLabel)
                    }
                }
            }
        }
    }
}
