package com.example.myapp.util

import android.content.Context
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.util.concurrent.ConcurrentHashMap

private const val assetPrefix = "asset:///"
const val DEFAULT_PTL_MODEL_PATH = "asset:///models/health_status_recognition_best.ptl"

data class ModelLoadResult(
    val success: Boolean,
    val message: String
)

private val moduleCache = ConcurrentHashMap<String, Module>()
private val resolvedPathCache = ConcurrentHashMap<String, String>()

fun tryLoadPtlFromAssets(context: Context, modelPath: String): ModelLoadResult {
    return try {
        val key = normalizeModelPath(modelPath)
        reloadModule(context, key)
        ModelLoadResult(true, "模型读取成功: $key")
    } catch (e: Exception) {
        ModelLoadResult(false, "模型读取失败: ${e.message ?: "未知错误"}")
    }
}

fun runPtlInference(
    context: Context,
    modelPath: String,
    inputCHW: FloatArray,
    channels: Int,
    targetPoints: Int = 300
): FloatArray {
    require(channels > 0) { "channels 必须大于 0" }
    require(targetPoints > 0) { "targetPoints 必须大于 0" }
    require(inputCHW.size == channels * targetPoints) {
        "输入长度不匹配，期待=${channels * targetPoints}，实际=${inputCHW.size}"
    }
    val module = loadModule(context, normalizeModelPath(modelPath))
    val input = Tensor.fromBlob(inputCHW, longArrayOf(1, channels.toLong(), targetPoints.toLong()))
    val output = module.forward(IValue.from(input)).toTensor()
    return output.dataAsFloatArray
}

private fun normalizeModelPath(modelPath: String): String {
    val normalized = modelPath.trim()
    require(normalized.isNotBlank()) { "未设置模型路径。" }
    require(normalized.lowercase().endsWith(".ptl")) { "路径不是 .ptl 文件。" }
    return normalized
}

private fun loadModule(context: Context, normalizedPath: String): Module {
    return moduleCache.getOrPut(normalizedPath) {
        val resolvedPath = resolvedPathCache.getOrPut(normalizedPath) {
            resolveModelPath(context, normalizedPath)
        }
        LiteModuleLoader.load(resolvedPath)
    }
}

private fun reloadModule(context: Context, normalizedPath: String): Module {
    moduleCache.remove(normalizedPath)
    resolvedPathCache.remove(normalizedPath)
    return loadModule(context, normalizedPath)
}

private fun resolveModelPath(context: Context, normalizedPath: String): String {
    val file = File(normalizedPath)
    if (file.exists() && file.isFile) {
        return file.absolutePath
    }
    val assetPath = if (normalizedPath.startsWith(assetPrefix)) {
        normalizedPath.removePrefix(assetPrefix)
    } else {
        normalizedPath
    }
    return copyAssetToInternalPath(context, assetPath)
}

private fun copyAssetToInternalPath(context: Context, assetPath: String): String {
    val cacheDir = File(context.filesDir, "ptl_models")
    if (!cacheDir.exists()) {
        cacheDir.mkdirs()
    }
    val outName = assetPath.replace("/", "_")
    val outFile = File(cacheDir, outName)
    context.assets.open(assetPath).use { input ->
        outFile.outputStream().use { output ->
            input.copyTo(output)
        }
    }
    return outFile.absolutePath
}
