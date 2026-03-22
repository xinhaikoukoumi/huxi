package com.example.myapp.ui.auth

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.example.myapp.model.UserProfile
import com.example.myapp.util.validateProfileInput

@Composable
fun AuthScreen(
    registeredUser: UserProfile?,
    onRegister: (UserProfile) -> Unit,
    onLogin: (String, String) -> Boolean
) {
    var registerMode by remember { mutableStateOf(registeredUser == null) }
    var statusMessage by remember { mutableStateOf("") }

    var name by remember { mutableStateOf("") }
    var ageInput by remember { mutableStateOf("") }
    var gender by remember { mutableStateOf("") }
    var history by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }

    LaunchedEffect(registeredUser) {
        registerMode = registeredUser == null
        statusMessage = ""
        name = ""
        ageInput = ""
        gender = ""
        history = ""
        password = ""
    }

    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        Column(
            modifier = Modifier
                .padding(innerPadding)
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text("实时呼吸监测系统", style = MaterialTheme.typography.headlineSmall)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                TextButton(onClick = { registerMode = true }) { Text("注册") }
                TextButton(onClick = { registerMode = false }) { Text("登录") }
            }
            HorizontalDivider()

            if (registerMode) {
                Text("本地注册（先注册后登录）")
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
                            statusMessage = validationError
                        } else {
                            onRegister(
                                UserProfile(
                                    name = name.trim(),
                                    age = ageInput.toInt(),
                                    gender = gender.trim(),
                                    history = history.trim(),
                                    password = password
                                )
                            )
                            statusMessage = "注册成功，正在进入主页..."
                        }
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("注册并进入主页")
                }
            } else {
                Text("使用本地已注册账号登录")
                if (registeredUser == null) {
                    Text("当前没有本地账号，请先注册。", color = MaterialTheme.colorScheme.error)
                }
                OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("姓名") }, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    label = { Text("密码") },
                    visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth()
                )
                Button(
                    onClick = {
                        statusMessage = if (onLogin(name, password)) {
                            "登录成功。"
                        } else {
                            "登录失败，请检查本地账号和密码。"
                        }
                    },
                    enabled = registeredUser != null,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("登录")
                }
            }

            if (statusMessage.isNotBlank()) {
                Text(statusMessage)
            }
        }
    }
}

