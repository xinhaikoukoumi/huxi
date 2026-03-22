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
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun DictionaryPanel(
    dictionary: Map<Int, String>,
    dictionaryEnabled: Boolean,
    onToggleDictionaryEnabled: (Boolean) -> Unit,
    onSaveDictionary: (Map<Int, String>) -> Unit
) {
    val editable = remember(dictionary) { mutableStateMapOf<Int, String>().apply { putAll(dictionary) } }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("词库设置", style = MaterialTheme.typography.titleMedium)
            Text("为数字识别标签 0-9 设置对应英文单词")
            Button(
                onClick = { onToggleDictionaryEnabled(!dictionaryEnabled) },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (dictionaryEnabled) "已启用词库（点击关闭）" else "启用词库设置")
            }

            (0..9).forEach { index ->
                OutlinedTextField(
                    value = editable[index].orEmpty(),
                    onValueChange = { editable[index] = it.trim() },
                    label = { Text("$index 对应单词") },
                    modifier = Modifier.fillMaxWidth()
                )
            }

            Button(
                onClick = { onSaveDictionary((0..9).associateWith { editable[it].orEmpty() }) },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("保存词库")
            }
        }
    }
}
