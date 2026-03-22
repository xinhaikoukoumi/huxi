package com.example.myapp.data

import android.content.Context
import androidx.core.content.edit
import com.example.myapp.model.BreathRateRecord
import com.example.myapp.model.UserProfile
import com.example.myapp.util.DEFAULT_PTL_MODEL_PATH
import java.util.Locale

class UserRepository(context: Context) {
    private val prefs = context.getSharedPreferences("local_user_store", Context.MODE_PRIVATE)
    private val keyRateHistory = "rate_history"
    private val keyExerciseModelPath = "exercise_model_path"
    private val keyDigitDictionary = "digit_dictionary"
    private val keyDictionaryEnabled = "dictionary_enabled"

    fun getRegisteredUser(): UserProfile? {
        val name = prefs.getString("name", null) ?: return null
        val age = prefs.getInt("age", -1)
        val gender = prefs.getString("gender", "") ?: ""
        val history = prefs.getString("history", "") ?: ""
        val password = prefs.getString("password", "") ?: ""
        if (age < 0 || password.isBlank()) return null
        return UserProfile(name, age, gender, history, password)
    }

    fun registerUser(user: UserProfile) {
        prefs.edit {
            putString("name", user.name)
            putInt("age", user.age)
            putString("gender", user.gender)
            putString("history", user.history)
            putString("password", user.password)
        }
    }

    fun updateUserProfile(user: UserProfile) {
        registerUser(user)
    }

    fun clearAccountData() {
        prefs.edit { clear() }
    }

    fun validateLogin(name: String, password: String): Boolean {
        val user = getRegisteredUser() ?: return false
        return user.name == name.trim() && user.password == password
    }

    fun getRateHistory(): List<BreathRateRecord> {
        val raw = prefs.getString(keyRateHistory, "").orEmpty()
        if (raw.isBlank()) return emptyList()
        return raw.split("|").mapNotNull { item ->
            val parts = item.split(",")
            if (parts.size != 2) return@mapNotNull null
            val timestamp = parts[0].toLongOrNull() ?: return@mapNotNull null
            val bpm = parts[1].toDoubleOrNull() ?: return@mapNotNull null
            BreathRateRecord(timestamp = timestamp, bpm = bpm)
        }
    }

    fun appendRateRecord(record: BreathRateRecord, maxItems: Int = 100) {
        val updated = getRateHistory().toMutableList().apply { add(record) }
        val trimmed = if (updated.size > maxItems) updated.takeLast(maxItems) else updated
        val encoded = trimmed.joinToString("|") {
            "${it.timestamp},${String.format(Locale.US, "%.2f", it.bpm)}"
        }
        prefs.edit { putString(keyRateHistory, encoded) }
    }

    fun clearRateHistory() {
        prefs.edit { remove(keyRateHistory) }
    }

    fun getExerciseModelPath(): String {
        return prefs.getString(keyExerciseModelPath, null)?.takeIf { it.isNotBlank() }
            ?: DEFAULT_PTL_MODEL_PATH
    }

    fun saveExerciseModelPath(path: String) {
        prefs.edit { putString(keyExerciseModelPath, path.trim()) }
    }

    fun getDigitDictionary(): Map<Int, String> {
        val defaults = defaultDigitDictionary()
        val raw = prefs.getString(keyDigitDictionary, null) ?: return defaults
        val parsed = raw.split("|").mapNotNull { entry ->
            val parts = entry.split("=")
            if (parts.size != 2) return@mapNotNull null
            val key = parts[0].toIntOrNull() ?: return@mapNotNull null
            key to parts[1]
        }.toMap()
        return defaults + parsed
    }

    fun saveDigitDictionary(mapping: Map<Int, String>) {
        val encoded = (0..9).joinToString("|") { index ->
            "$index=${mapping[index].orEmpty()}"
        }
        prefs.edit { putString(keyDigitDictionary, encoded) }
    }

    fun isDictionaryEnabled(): Boolean {
        return prefs.getBoolean(keyDictionaryEnabled, false)
    }

    fun saveDictionaryEnabled(enabled: Boolean) {
        prefs.edit { putBoolean(keyDictionaryEnabled, enabled) }
    }

    private fun defaultDigitDictionary(): Map<Int, String> {
        return mapOf(
            0 to "zero",
            1 to "one",
            2 to "two",
            3 to "three",
            4 to "four",
            5 to "five",
            6 to "six",
            7 to "seven",
            8 to "eight",
            9 to "nine"
        )
    }
}
