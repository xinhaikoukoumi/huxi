package com.example.myapp.model

data class UserProfile(
    val name: String,
    val age: Int,
    val gender: String,
    val history: String,
    val password: String
)

enum class MonitorMode {
    Numeric,
    Waveform
}

enum class MenuModule(val title: String) {
    ExerciseAnalysis("健康运动分析"),
    MorseDigit("摩斯/数字编码"),
    Dictionary("词库设置"),
    DataManagement("数据管理")
}

enum class MorseRecognitionMode(val title: String) {
    Digits("数字识别"),
    Letters("字母识别")
}

data class BreathRateRecord(val timestamp: Long, val bpm: Double)
