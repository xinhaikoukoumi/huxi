package com.example.myapp.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
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
import java.util.Locale

@Composable
fun ExerciseAnalysisPanel(
    ptlModelPath: String,
    onPathChange: (String) -> Unit,
    onTryLoad: () -> Unit,
    enabled: Boolean,
    isRecognizing: Boolean,
    onStartRecognition: () -> Unit,
    onStopRecognition: () -> Unit,
    modelLabel: String,
    loadStatus: String
) {
    val isValidPtlPath = ptlModelPath.isBlank() || ptlModelPath.lowercase(Locale.getDefault()).endsWith(".ptl")
    var modelConfigExpanded by rememberSaveable { mutableStateOf(true) }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("健康运动分析", style = MaterialTheme.typography.titleMedium)
            TextButton(onClick = { modelConfigExpanded = !modelConfigExpanded }) {
                Text(if (modelConfigExpanded) "收起模型配置" else "展开模型配置")
            }

            if (modelConfigExpanded) {
                Text("模型权重文件读取位置")
                OutlinedTextField(
                    value = ptlModelPath,
                    onValueChange = onPathChange,
                    label = { Text(".ptl 文件路径") },
                    placeholder = { Text("asset:///models/health_status_recognition_best.ptl") },
                    modifier = Modifier.fillMaxWidth()
                )
                if (!isValidPtlPath) {
                    Text("请填写 .ptl 结尾的权重文件路径。", color = MaterialTheme.colorScheme.error)
                }
                Button(onClick = onTryLoad, modifier = Modifier.fillMaxWidth()) {
                    Text("尝试读取模型")
                }
            }
            Text(loadStatus)
            Text(if (enabled) "当前状态: 已开启" else "当前状态: 未开启")
            if (enabled) {
                Text(if (isRecognizing) "识别运行状态: 识别中" else "识别运行状态: 未开始")
                Button(
                    onClick = onStartRecognition,
                    enabled = !isRecognizing && loadStatus.startsWith("模型读取成功"),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("开始识别")
                }
                Button(
                    onClick = onStopRecognition,
                    enabled = isRecognizing,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("停止识别")
                }
            }
            if (enabled) {
                Text("模型识别标签: $modelLabel")
            }
            Text("标签映射: 0 -> 正常, 1 -> 咳嗽, 2 -> 锻炼, 3 -> 鼻塞, 4 -> 鼻子呼吸")
        }
    }
}
