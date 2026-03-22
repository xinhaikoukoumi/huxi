package com.example.myapp.util

fun validateProfileInput(
    name: String,
    ageInput: String,
    gender: String,
    password: String
): String? {
    val age = ageInput.toIntOrNull() ?: return "年龄必须为数字。"
    if (name.isBlank() || gender.isBlank() || password.isBlank()) {
        return "请完整填写姓名、年龄、性别和密码。"
    }
    if (age !in 1..120) return "年龄范围应在 1 到 120 之间。"
    return null
}

