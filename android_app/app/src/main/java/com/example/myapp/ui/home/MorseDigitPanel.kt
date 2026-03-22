package com.example.myapp.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.myapp.model.MorseRecognitionMode
import java.util.Locale

@Composable
fun MorseDigitPanel(
    mode: MorseRecognitionMode,
    modelPath: String,
    onModelPathChange: (String) -> Unit,
    onTryLoadModel: () -> Unit,
    loadStatus: String,
    modelReady: Boolean,
    isRecognizing: Boolean,
    onStartRecognition: () -> Unit,
    onStopRecognition: () -> Unit,
    currentLabel: String,
    currentWindowBest: String,
    sequenceRaw: String,
    sequenceDisplay: String
) {
    val validModelPath = modelPath.isBlank() || modelPath.lowercase(Locale.getDefault()).endsWith(".ptl")
    var modelConfigExpanded by rememberSaveable { mutableStateOf(true) }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("摩斯/数字编码", style = MaterialTheme.typography.titleMedium)
            Text("当前模式: ${mode.title}")

            TextButton(onClick = { modelConfigExpanded = !modelConfigExpanded }) {
                Text(if (modelConfigExpanded) "收起模型配置" else "展开模型配置")
            }

            if (modelConfigExpanded) {
                OutlinedTextField(
                    value = modelPath,
                    onValueChange = onModelPathChange,
                    label = { Text("${mode.title}模型 .ptl 路径") },
                    modifier = Modifier.fillMaxWidth()
                )
                if (!validModelPath) {
                    Text("模型路径需为 .ptl 文件", color = MaterialTheme.colorScheme.error)
                }
                Button(onClick = onTryLoadModel, modifier = Modifier.fillMaxWidth()) { Text("尝试读取模型") }
            }
            Text(loadStatus)

            Text(
                text = if (modelReady) "模型状态: 已就绪" else "模型状态: 需要先导入模型",
                color = if (modelReady) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error
            )

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = onStartRecognition, enabled = modelReady && !isRecognizing) { Text("开始识别") }
                Button(onClick = onStopRecognition, enabled = isRecognizing) { Text("停止识别") }
            }
            Text(if (isRecognizing) "识别运行状态: 识别中" else "识别运行状态: 未开始")
            Text("当前识别标签: $currentLabel")
            Text("当前30秒窗口最可能结果: ${if (currentWindowBest.isBlank()) "-" else currentWindowBest}")
            Text("序列(原始): ${if (sequenceRaw.isBlank()) "-" else sequenceRaw}")
            Text("序列(解析): ${if (sequenceDisplay.isBlank()) "-" else sequenceDisplay}")
            Text("数字标签: 0-9；字母标签: A-Z")
        }
    }
}
