package com.example.myapp.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.example.myapp.model.BreathRateRecord
import com.example.myapp.model.UserProfile
import com.example.myapp.util.formatTimestamp
import com.example.myapp.util.validateProfileInput

@Composable
fun DataManagementPanel(
    currentUser: UserProfile?,
    currentBpm: Double,
    rateHistory: List<BreathRateRecord>,
    onSaveProfile: (UserProfile) -> Unit,
    onAddRateHistory: (Double) -> Unit,
    onClearRateHistory: () -> Unit,
    onClearAccount: () -> Unit,
    onLogout: () -> Unit
) {
    var name by remember(currentUser?.name) { mutableStateOf(currentUser?.name.orEmpty()) }
    var ageInput by remember(currentUser?.age) { mutableStateOf(currentUser?.age?.toString().orEmpty()) }
    var gender by remember(currentUser?.gender) { mutableStateOf(currentUser?.gender.orEmpty()) }
    var history by remember(currentUser?.history) { mutableStateOf(currentUser?.history.orEmpty()) }
    var password by remember(currentUser?.password) { mutableStateOf(currentUser?.password.orEmpty()) }
    var statusMessage by remember { mutableStateOf("") }
    var showClearHistoryDialog by remember { mutableStateOf(false) }
    var showClearAccountDialog by remember { mutableStateOf(false) }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("数据管理", style = MaterialTheme.typography.titleMedium)

            if (currentUser == null) {
                Text("当前没有可展示的登录用户信息。")
            } else {
                Text("编辑当前登录人信息")
                OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("姓名") }, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(
                    value = ageInput,
                    onValueChange = { ageInput = it.filter(Char::isDigit) },
                    label = { Text("年龄") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth()
                )
                OutlinedTextField(value = gender, onValueChange = { gender = it }, label = { Text("性别") }, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(value = history, onValueChange = { history = it }, label = { Text("病史") }, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    label = { Text("密码") },
                    visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth()
                )
                Button(
                    onClick = {
                        val validationError = validateProfileInput(name, ageInput, gender, password)
                        if (validationError != null) {
                            statusMessage = "保存失败，$validationError"
                        } else {
                            onSaveProfile(
                                UserProfile(
                                    name = name.trim(),
                                    age = ageInput.toInt(),
                                    gender = gender.trim(),
                                    history = history.trim(),
                                    password = password
                                )
                            )
                            statusMessage = "用户信息已保存到本地。"
                        }
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("保存用户信息")
                }
            }

            Button(onClick = onLogout, modifier = Modifier.fillMaxWidth()) { Text("退出登录") }
            Button(onClick = { showClearAccountDialog = true }, modifier = Modifier.fillMaxWidth()) {
                Text("清空本地账号数据（注销）")
            }

            HorizontalDivider()
            Text("呼吸频率历史记录（本地存储）")
            Text("当前频率: ${"%.1f".format(currentBpm)} 次/分")

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = {
                    if (currentBpm > 0.0) {
                        onAddRateHistory(currentBpm)
                        statusMessage = "已保存当前呼吸频率到历史记录。"
                    } else {
                        statusMessage = "当前频率无效，稍后再试。"
                    }
                }) { Text("保存当前频率") }

                Button(onClick = { showClearHistoryDialog = true }) { Text("清空历史记录") }
            }

            if (rateHistory.isEmpty()) {
                Text("暂无历史记录。")
            } else {
                rateHistory.asReversed().take(10).forEachIndexed { index, record ->
                    Text("${index + 1}. ${formatTimestamp(record.timestamp)} - ${"%.1f".format(record.bpm)} 次/分")
                }
            }

            if (statusMessage.isNotBlank()) {
                Text(statusMessage)
            }
        }
    }

    if (showClearHistoryDialog) {
        AlertDialog(
            onDismissRequest = { showClearHistoryDialog = false },
            title = { Text("确认清空") },
            text = { Text("是否清空所有呼吸频率历史记录？") },
            confirmButton = {
                TextButton(onClick = {
                    onClearRateHistory()
                    statusMessage = "历史记录已清空。"
                    showClearHistoryDialog = false
                }) { Text("确认") }
            },
            dismissButton = {
                TextButton(onClick = { showClearHistoryDialog = false }) { Text("取消") }
            }
        )
    }

    if (showClearAccountDialog) {
        AlertDialog(
            onDismissRequest = { showClearAccountDialog = false },
            title = { Text("确认注销") },
            text = { Text("此操作会清空本地账号与历史记录，是否继续？") },
            confirmButton = {
                TextButton(onClick = {
                    showClearAccountDialog = false
                    onClearAccount()
                }) { Text("确认") }
            },
            dismissButton = {
                TextButton(onClick = { showClearAccountDialog = false }) { Text("取消") }
            }
        )
    }
}

