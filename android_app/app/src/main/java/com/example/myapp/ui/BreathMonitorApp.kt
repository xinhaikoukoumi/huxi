package com.example.myapp.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import com.example.myapp.data.UserRepository
import com.example.myapp.model.BreathRateRecord
import com.example.myapp.model.UserProfile
import com.example.myapp.ui.auth.AuthScreen
import com.example.myapp.ui.home.HomeScreen

@Composable
fun BreathMonitorApp() {
    val context = LocalContext.current
    val repository = remember(context) { UserRepository(context) }

    var registeredUser by remember { mutableStateOf(repository.getRegisteredUser()) }
    var currentUser by remember { mutableStateOf<UserProfile?>(null) }
    var rateHistory by remember { mutableStateOf(repository.getRateHistory()) }
    var exerciseModelPath by remember { mutableStateOf(repository.getExerciseModelPath()) }
    var digitDictionary by remember { mutableStateOf(repository.getDigitDictionary()) }
    var dictionaryEnabled by remember { mutableStateOf(repository.isDictionaryEnabled()) }

    if (currentUser == null) {
        AuthScreen(
            registeredUser = registeredUser,
            onRegister = { user ->
                repository.registerUser(user)
                registeredUser = user
                currentUser = user
                rateHistory = repository.getRateHistory()
            },
            onLogin = { name, password ->
                val success = repository.validateLogin(name, password)
                if (success) {
                    currentUser = repository.getRegisteredUser()
                    rateHistory = repository.getRateHistory()
                }
                success
            }
        )
    } else {
        HomeScreen(
            currentUser = currentUser,
            rateHistory = rateHistory,
            initialModelPath = exerciseModelPath,
            digitDictionary = digitDictionary,
            dictionaryEnabled = dictionaryEnabled,
            onSaveModelPath = { path ->
                exerciseModelPath = path
                repository.saveExerciseModelPath(path)
            },
            onSaveDigitDictionary = { mapping ->
                digitDictionary = mapping
                repository.saveDigitDictionary(mapping)
            },
            onToggleDictionaryEnabled = { enabled ->
                dictionaryEnabled = enabled
                repository.saveDictionaryEnabled(enabled)
            },
            onSaveProfile = { updatedUser ->
                repository.updateUserProfile(updatedUser)
                registeredUser = updatedUser
                currentUser = updatedUser
            },
            onAddRateHistory = { bpm ->
                repository.appendRateRecord(BreathRateRecord(System.currentTimeMillis(), bpm))
                rateHistory = repository.getRateHistory()
            },
            onClearRateHistory = {
                repository.clearRateHistory()
                rateHistory = emptyList()
            },
            onClearAccount = {
                repository.clearAccountData()
                registeredUser = null
                currentUser = null
                rateHistory = emptyList()
            },
            onLogout = {
                currentUser = null
            }
        )
    }
}
