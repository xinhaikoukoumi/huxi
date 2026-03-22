package com.example.myapp.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.myapp.model.BreathRateRecord
import com.example.myapp.model.MenuModule
import com.example.myapp.model.MorseRecognitionMode
import com.example.myapp.model.MonitorMode
import com.example.myapp.model.UserProfile
import com.example.myapp.util.DEFAULT_PTL_MODEL_PATH
import com.example.myapp.util.InferenceTask
import com.example.myapp.util.WindowPrediction
import com.example.myapp.util.inferFolderWithPtl
import com.example.myapp.util.tryLoadPtlFromAssets
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    currentUser: UserProfile?,
    rateHistory: List<BreathRateRecord>,
    initialModelPath: String,
    digitDictionary: Map<Int, String>,
    dictionaryEnabled: Boolean,
    onSaveModelPath: (String) -> Unit,
    onSaveDigitDictionary: (Map<Int, String>) -> Unit,
    onToggleDictionaryEnabled: (Boolean) -> Unit,
    onSaveProfile: (UserProfile) -> Unit,
    onAddRateHistory: (Double) -> Unit,
    onClearRateHistory: () -> Unit,
    onClearAccount: () -> Unit,
    onLogout: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var selectedMode by remember { mutableStateOf(MonitorMode.Numeric) }
    var expandedMenu by remember { mutableStateOf(false) }
    var selectedModule by remember { mutableStateOf<MenuModule?>(null) }
    var currentBpm by remember { mutableStateOf(0.0) }

    var exerciseAnalysisEnabled by remember { mutableStateOf(false) }
    var showExerciseConfirmDialog by remember { mutableStateOf(false) }
    var ptlModelPath by remember(initialModelPath) { mutableStateOf(initialModelPath) }
    var modelLabel by remember { mutableStateOf("待识别") }
    var modelLoadStatus by remember { mutableStateOf("尚未尝试读取模型。") }
    var modelImported by remember { mutableStateOf(false) }
    var isRecognizing by remember { mutableStateOf(false) }
    var healthPredictions by remember { mutableStateOf<Map<String, List<WindowPrediction>>>(emptyMap()) }

    var showImportResultDialog by remember { mutableStateOf(false) }
    var importResultMessage by remember { mutableStateOf("") }

    var showMorseModeDialog by remember { mutableStateOf(false) }
    var morseMode by remember { mutableStateOf(MorseRecognitionMode.Digits) }
    var morseRecognizing by remember { mutableStateOf(false) }
    var morseLabel by remember { mutableStateOf("-") }

    var digitModelPath by remember { mutableStateOf("asset:///models/digits_recognition_best.ptl") }
    var letterModelPath by remember { mutableStateOf("asset:///models/letters_recognition_best.ptl") }
    var digitLoadStatus by remember { mutableStateOf("未读取") }
    var letterLoadStatus by remember { mutableStateOf("未读取") }
    var digitModelLoaded by remember { mutableStateOf(false) }
    var letterModelLoaded by remember { mutableStateOf(false) }
    var morsePredictions by remember { mutableStateOf<Map<String, List<WindowPrediction>>>(emptyMap()) }

    var playbackVersion by remember { mutableStateOf(0) }
    var morseCurrentSegmentKey by remember { mutableStateOf<String?>(null) }
    var morseWindowVotes by remember { mutableStateOf<Map<String, Int>>(emptyMap()) }
    var morseWindowBest by remember { mutableStateOf("") }
    var morseSequence by remember { mutableStateOf<List<String>>(emptyList()) }

    val currentMorseModelReady = if (morseMode == MorseRecognitionMode.Digits) {
        digitModelLoaded
    } else {
        letterModelLoaded
    }

    val activeSignalFolder = when {
        morseRecognizing || selectedModule == MenuModule.MorseDigit -> {
            if (morseMode == MorseRecognitionMode.Digits) "demo_signals/num" else "demo_signals/zimu"
        }
        exerciseAnalysisEnabled || selectedModule == MenuModule.ExerciseAnalysis -> "demo_signals/yundong"
        else -> "demo_signals/yundong"
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("呼吸监测主页") },
                actions = {
                    IconButton(onClick = { expandedMenu = true }) {
                        Icon(imageVector = Icons.Filled.MoreVert, contentDescription = "更多功能")
                    }
                    DropdownMenu(expanded = expandedMenu, onDismissRequest = { expandedMenu = false }) {
                        MenuModule.entries.forEach { module ->
                            DropdownMenuItem(
                                text = { Text(module.title) },
                                onClick = {
                                    expandedMenu = false
                                    when (module) {
                                        MenuModule.ExerciseAnalysis -> {
                                            if (ptlModelPath.isBlank()) {
                                                ptlModelPath = DEFAULT_PTL_MODEL_PATH
                                                onSaveModelPath(DEFAULT_PTL_MODEL_PATH)
                                            }
                                            selectedModule = module
                                            showExerciseConfirmDialog = true
                                        }
                                        MenuModule.MorseDigit -> {
                                            selectedModule = module
                                            showMorseModeDialog = true
                                        }
                                        MenuModule.Dictionary,
                                        MenuModule.DataManagement -> {
                                            selectedModule = module
                                        }
                                    }
                                }
                            )
                        }
                    }
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .padding(innerPadding)
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    if (selectedModule == MenuModule.MorseDigit) {
                        Text(
                            text = "模型导入状态: ${if (currentMorseModelReady) "模型已就绪" else "模型未就绪"}",
                            color = if (currentMorseModelReady) Color(0xFF1B5E20) else Color(0xFFB71C1C)
                        )
                        val morseStatus = when {
                            !currentMorseModelReady -> "等待模型导入"
                            morseRecognizing -> "识别中"
                            else -> "已就绪（未开始）"
                        }
                        Text("识别状态: $morseStatus")
                        Text("窗口最可能结果: ${if (morseWindowBest.isBlank()) "-" else morseWindowBest}")
                        Text("序列(原始): ${if (morseSequence.isEmpty()) "-" else morseSequence.joinToString(" ")}")
                        Text("序列(解析): ${buildMorseSequenceDisplay(morseMode, morseSequence, digitDictionary, dictionaryEnabled)}")
                    } else {
                        Text(
                            text = "模型导入状态: ${if (modelImported) "已成功导入" else "未导入"}",
                            color = if (modelImported) Color(0xFF1B5E20) else Color(0xFFB71C1C)
                        )
                    }
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = { selectedMode = MonitorMode.Numeric }) { Text("数值模式") }
                Button(onClick = { selectedMode = MonitorMode.Waveform }) { Text("波形模式") }
            }

            BreathingMonitorPanel(
                mode = selectedMode,
                signalFolder = activeSignalFolder,
                playbackVersion = playbackVersion,
                exerciseAnalysisEnabled = exerciseAnalysisEnabled || morseRecognizing,
                isRecognitionRunning = isRecognizing || morseRecognizing,
                modelLabel = if (morseRecognizing) morseLabel else modelLabel,
                onRateUpdate = { bpm -> currentBpm = bpm },
                onSignalTimeUpdate = { _ -> },
                onPlaybackCursorUpdate = { fileName, fileTimeSec ->
                    if (exerciseAnalysisEnabled && modelImported && isRecognizing) {
                        val pred = pickPrediction(healthPredictions, fileName, fileTimeSec)
                        modelLabel = pred?.let {
                            val pct = (it.confidence * 100f).roundToInt().coerceIn(0, 100)
                            "${it.label} ($pct%)"
                        } ?: "待识别"
                    }

                    if (!morseRecognizing || !currentMorseModelReady) return@BreathingMonitorPanel
                    val pred = pickPrediction(morsePredictions, fileName, fileTimeSec)
                    morseLabel = pred?.label ?: "-"
                    if (pred == null) return@BreathingMonitorPanel

                    val segmentKey = "$fileName#${pred.segmentIndex}"
                    if (morseCurrentSegmentKey == null) {
                        morseCurrentSegmentKey = segmentKey
                    } else if (segmentKey != morseCurrentSegmentKey) {
                        val best = morseWindowVotes.maxByOrNull { it.value }?.key.orEmpty()
                        if (best.isNotBlank()) morseSequence = morseSequence + best
                        morseCurrentSegmentKey = segmentKey
                        morseWindowVotes = emptyMap()
                        morseWindowBest = ""
                    }
                    val updated = morseWindowVotes.toMutableMap()
                    updated[pred.label] = (updated[pred.label] ?: 0) + 1
                    morseWindowVotes = updated
                    morseWindowBest = updated.maxByOrNull { it.value }?.key.orEmpty()
                }
            )

            when (selectedModule) {
                MenuModule.ExerciseAnalysis -> {
                    ExerciseAnalysisPanel(
                        ptlModelPath = ptlModelPath,
                        onPathChange = {
                            ptlModelPath = it
                            onSaveModelPath(it)
                            modelLoadStatus = "路径已更新，点击“尝试读取模型”验证。"
                        },
                        onTryLoad = {
                            val result = tryLoadPtlFromAssets(context, ptlModelPath)
                            modelImported = result.success
                            isRecognizing = false
                            healthPredictions = emptyMap()
                            modelLoadStatus = result.message
                            importResultMessage = if (result.success) {
                                "模型导入成功。"
                            } else {
                                "模型导入失败。\n${result.message}"
                            }
                            showImportResultDialog = true
                        },
                        enabled = exerciseAnalysisEnabled,
                        isRecognizing = isRecognizing,
                        onStartRecognition = {
                            if (!modelImported) {
                                importResultMessage = "请先成功导入健康模型后再开始识别。"
                                showImportResultDialog = true
                                return@ExerciseAnalysisPanel
                            }
                            modelLabel = "准备中..."
                            isRecognizing = false
                            scope.launch {
                                val result = withContext(Dispatchers.Default) {
                                    inferFolderWithPtl(context, "demo_signals/yundong", ptlModelPath, InferenceTask.Health)
                                }
                                if (result.success) {
                                    healthPredictions = result.predictionsByFile
                                    isRecognizing = true
                                    modelLabel = "待识别"
                                    playbackVersion += 1
                                } else {
                                    isRecognizing = false
                                    modelLabel = "待识别"
                                    importResultMessage = "健康识别推理失败：${result.message}"
                                    showImportResultDialog = true
                                }
                            }
                        },
                        onStopRecognition = {
                            isRecognizing = false
                            modelLabel = "待识别"
                        },
                        modelLabel = modelLabel,
                        loadStatus = modelLoadStatus
                    )
                }
                MenuModule.MorseDigit -> {
                    MorseDigitPanel(
                        mode = morseMode,
                        modelPath = if (morseMode == MorseRecognitionMode.Digits) digitModelPath else letterModelPath,
                        onModelPathChange = { path ->
                            if (morseMode == MorseRecognitionMode.Digits) digitModelPath = path else letterModelPath = path
                        },
                        onTryLoadModel = {
                            val path = if (morseMode == MorseRecognitionMode.Digits) digitModelPath else letterModelPath
                            val result = tryLoadPtlFromAssets(context, path)
                            if (morseMode == MorseRecognitionMode.Digits) {
                                digitLoadStatus = result.message
                                digitModelLoaded = result.success
                            } else {
                                letterLoadStatus = result.message
                                letterModelLoaded = result.success
                            }
                            val fileName = path.substringAfterLast('/').ifBlank { path }
                            importResultMessage = if (result.success) {
                                "模型导入成功: $fileName"
                            } else {
                                "模型导入失败: $fileName\n${result.message}"
                            }
                            showImportResultDialog = true
                        },
                        loadStatus = if (morseMode == MorseRecognitionMode.Digits) digitLoadStatus else letterLoadStatus,
                        modelReady = currentMorseModelReady,
                        isRecognizing = morseRecognizing,
                        onStartRecognition = {
                            if (!currentMorseModelReady) {
                                importResultMessage = "请先成功导入当前模式下的模型后再开始识别。"
                                showImportResultDialog = true
                                return@MorseDigitPanel
                            }
                            morseRecognizing = false
                            morseLabel = "准备中..."
                            scope.launch {
                                val folder = if (morseMode == MorseRecognitionMode.Digits) "demo_signals/num" else "demo_signals/zimu"
                                val path = if (morseMode == MorseRecognitionMode.Digits) digitModelPath else letterModelPath
                                val task = if (morseMode == MorseRecognitionMode.Digits) InferenceTask.Digits else InferenceTask.Letters
                                val result = withContext(Dispatchers.Default) {
                                    inferFolderWithPtl(context, folder, path, task)
                                }
                                if (result.success) {
                                    morsePredictions = result.predictionsByFile
                                    morseSequence = emptyList()
                                    morseWindowVotes = emptyMap()
                                    morseWindowBest = ""
                                    morseCurrentSegmentKey = null
                                    morseRecognizing = true
                                    morseLabel = "-"
                                    playbackVersion += 1
                                } else {
                                    morseRecognizing = false
                                    morseLabel = "-"
                                    importResultMessage = "序列识别推理失败：${result.message}"
                                    showImportResultDialog = true
                                }
                            }
                        },
                        onStopRecognition = {
                            morseRecognizing = false
                            val best = morseWindowVotes.maxByOrNull { it.value }?.key.orEmpty()
                            if (best.isNotBlank()) morseSequence = morseSequence + best
                            morseCurrentSegmentKey = null
                            morseWindowVotes = emptyMap()
                            morseWindowBest = ""
                        },
                        currentLabel = morseLabel,
                        currentWindowBest = morseWindowBest,
                        sequenceRaw = morseSequence.joinToString(" "),
                        sequenceDisplay = buildMorseSequenceDisplay(morseMode, morseSequence, digitDictionary, dictionaryEnabled)
                    )
                }
                MenuModule.Dictionary -> {
                    DictionaryPanel(
                        dictionary = digitDictionary,
                        dictionaryEnabled = dictionaryEnabled,
                        onToggleDictionaryEnabled = onToggleDictionaryEnabled,
                        onSaveDictionary = { mapping ->
                            onSaveDigitDictionary(mapping)
                            importResultMessage = "词库已保存。"
                            showImportResultDialog = true
                        }
                    )
                }
                MenuModule.DataManagement -> {
                    DataManagementPanel(
                        currentUser = currentUser,
                        currentBpm = currentBpm,
                        rateHistory = rateHistory,
                        onSaveProfile = onSaveProfile,
                        onAddRateHistory = onAddRateHistory,
                        onClearRateHistory = onClearRateHistory,
                        onClearAccount = onClearAccount,
                        onLogout = onLogout
                    )
                }
                null -> {
                    Card(modifier = Modifier.fillMaxWidth()) {
                        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text("功能菜单")
                            Text("点击右上角三点菜单，选择需要的模块。")
                        }
                    }
                }
            }
        }
    }

    if (showExerciseConfirmDialog) {
        AlertDialog(
            onDismissRequest = { showExerciseConfirmDialog = false },
            title = { Text("开启健康运动分析") },
            text = { Text("是否开启健康运动分析？开启后将在实时呼吸监测区域显示模型识别标签。") },
            confirmButton = {
                TextButton(onClick = {
                    showExerciseConfirmDialog = false
                    exerciseAnalysisEnabled = true
                    val result = tryLoadPtlFromAssets(context, ptlModelPath)
                    modelImported = result.success
                    modelLoadStatus = result.message
                    importResultMessage = if (result.success) {
                        "模型导入成功，可点击“开始识别”。"
                    } else {
                        "模型导入失败。\n${result.message}"
                    }
                    showImportResultDialog = true
                }) { Text("开启") }
            },
            dismissButton = {
                TextButton(onClick = { showExerciseConfirmDialog = false }) { Text("取消") }
            }
        )
    }

    if (showMorseModeDialog) {
        AlertDialog(
            onDismissRequest = { showMorseModeDialog = false },
            title = { Text("选择识别模式") },
            text = { Text("请选择摩斯/数字编码模块的识别模式") },
            confirmButton = {
                TextButton(onClick = {
                    morseMode = MorseRecognitionMode.Digits
                    morseRecognizing = false
                    morseCurrentSegmentKey = null
                    morseWindowVotes = emptyMap()
                    morseWindowBest = ""
                    morseSequence = emptyList()
                    showMorseModeDialog = false
                }) { Text("数字识别") }
            },
            dismissButton = {
                TextButton(onClick = {
                    morseMode = MorseRecognitionMode.Letters
                    morseRecognizing = false
                    morseCurrentSegmentKey = null
                    morseWindowVotes = emptyMap()
                    morseWindowBest = ""
                    morseSequence = emptyList()
                    showMorseModeDialog = false
                }) { Text("字母识别") }
            }
        )
    }

    if (showImportResultDialog) {
        AlertDialog(
            onDismissRequest = { showImportResultDialog = false },
            title = { Text("结果") },
            text = { Text(importResultMessage) },
            confirmButton = {
                TextButton(onClick = { showImportResultDialog = false }) { Text("确定") }
            }
        )
    }
}

private fun pickPrediction(
    predictions: Map<String, List<WindowPrediction>>,
    fileName: String,
    fileTimeSec: Double
): WindowPrediction? {
    val list = predictions[fileName]
        ?: predictions.entries.firstOrNull { fileName.endsWith(it.key) || it.key.endsWith(fileName) }?.value
        ?: return null
    val epsilon = 1e-6
    return list.firstOrNull { pred ->
        fileTimeSec + epsilon >= pred.startSec && fileTimeSec < pred.endSec - epsilon
    } ?: list.lastOrNull { pred ->
        fileTimeSec + epsilon >= pred.startSec
    } ?: list.firstOrNull()
}

private fun buildMorseSequenceDisplay(
    mode: MorseRecognitionMode,
    sequence: List<String>,
    digitDictionary: Map<Int, String>,
    dictionaryEnabled: Boolean
): String {
    if (sequence.isEmpty()) return "-"
    if (!dictionaryEnabled) return "不启用"
    return if (mode == MorseRecognitionMode.Digits) {
        sequence.joinToString(" ") { token ->
            token.toIntOrNull()?.let { digitDictionary[it].orEmpty() }.orEmpty().ifBlank { "未知" }
        }
    } else {
        sequence.joinToString("")
    }
}
