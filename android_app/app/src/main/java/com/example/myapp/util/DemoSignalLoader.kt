package com.example.myapp.util

import android.content.Context

data class DemoSignalPoint(
    val timeSec: Double,
    val value: Float
)

data class DemoSignalFile(
    val name: String,
    val points: List<DemoSignalPoint>
)

fun loadDemoSignalFiles(context: Context, folder: String = "demo_signals"): List<DemoSignalFile> {
    val fileNames = context.assets.list(folder)?.filter { it.lowercase().endsWith(".csv") }?.sorted().orEmpty()
    return fileNames.mapNotNull { fileName ->
        val path = "$folder/$fileName"
        val content = runCatching { context.assets.open(path).bufferedReader().use { it.readText() } }.getOrNull()
            ?: return@mapNotNull null
        val points = parseCsvPoints(content)
        if (points.isEmpty()) return@mapNotNull null
        DemoSignalFile(name = fileName, points = normalizePoints(points))
    }
}

private fun parseCsvPoints(content: String): List<DemoSignalPoint> {
    val raw = mutableListOf<DemoSignalPoint>()
    var fallbackTime = 0.0

    content.lineSequence().forEach { line ->
        if (line.isBlank()) return@forEach
        val tokens = line.split(',', ';', '\t').map { it.trim() }
        val numbers = tokens.mapNotNull { it.toDoubleOrNull() }
        if (numbers.isEmpty()) return@forEach

        val value = numbers.last().toFloat()
        val timeSec = if (numbers.size >= 2) numbers.first() else fallbackTime
        raw.add(DemoSignalPoint(timeSec = timeSec, value = value))
        fallbackTime += 0.06
    }

    if (raw.isEmpty()) return emptyList()

    // Ensure monotonic time axis even when CSV contains non-increasing timestamps.
    var lastTime = raw.first().timeSec
    return raw.mapIndexed { index, p ->
        val t = when {
            index == 0 -> p.timeSec.coerceAtLeast(0.0)
            p.timeSec > lastTime -> p.timeSec
            else -> lastTime + 0.06
        }
        lastTime = t
        DemoSignalPoint(timeSec = t, value = p.value)
    }
}

private fun normalizePoints(input: List<DemoSignalPoint>): List<DemoSignalPoint> {
    val min = input.minOfOrNull { it.value } ?: return emptyList()
    val max = input.maxOfOrNull { it.value } ?: return emptyList()
    val span = max - min
    if (span < 1e-6f) {
        return input.map { it.copy(value = 0f) }
    }
    return input.map { p ->
        val normalized = ((p.value - min) / span) * 2f - 1f
        p.copy(value = normalized)
    }
}
