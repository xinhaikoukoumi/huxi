package com.example.morse

import android.content.Context
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File

class MorseModelManager(private val context: Context) {
    private val letters = loadModule("models/letters/morse_char_model_android.ptl")
    private val digits = loadModule("models/digits/morse_char_model_android.ptl")
    private val health52 = loadModule("models/health_seed52/morse_char_model_android.ptl")
    private val health62 = loadModule("models/health_seed62/morse_char_model_android.ptl")

    fun inferLetters(inputCHW: FloatArray, channels: Long): FloatArray {
        return inferSingle(letters, inputCHW, channels)
    }

    fun inferDigits(inputCHW: FloatArray, channels: Long): FloatArray {
        return inferSingle(digits, inputCHW, channels)
    }

    fun inferHealthEnsemble(inputCHW: FloatArray): FloatArray {
        val l52 = inferSingle(health52, inputCHW, 2)
        val l62 = inferSingle(health62, inputCHW, 2)
        val out = FloatArray(l52.size)
        for (i in out.indices) out[i] = 0.5f * (l52[i] + l62[i])
        return out
    }

    private fun inferSingle(module: Module, inputCHW: FloatArray, channels: Long): FloatArray {
        val input = Tensor.fromBlob(inputCHW, longArrayOf(1, channels, 300))
        val output = module.forward(IValue.from(input)).toTensor()
        return output.dataAsFloatArray
    }

    private fun loadModule(assetPath: String): Module {
        val file = assetFilePath(assetPath)
        return LiteModuleLoader.load(file)
    }

    private fun assetFilePath(assetPath: String): String {
        val file = File(context.filesDir, assetPath.replace("/", "_"))
        if (!file.exists()) {
            context.assets.open(assetPath).use { input ->
                file.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return file.absolutePath
    }
}
