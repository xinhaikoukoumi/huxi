package com.example.myapp.ui.home

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

private data class LabelColorStyle(val bg: Color, val fg: Color)

@Composable
fun RecognitionLabelBadge(labelText: String) {
    val style = labelStyleForLabel(labelText)
    Box(
        modifier = Modifier
            .background(style.bg, RoundedCornerShape(10.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp)
    ) {
        Text(text = labelText, color = style.fg)
    }
}

private fun labelStyleForLabel(labelText: String): LabelColorStyle {
    val normalized = labelText.substringBefore('(').trim()
    return when {
        normalized.substringBefore("->").trim().toIntOrNull() == 0 || normalized == "正常" || normalized == "口呼吸" -> {
            LabelColorStyle(bg = Color(0xFFFFE0B2), fg = Color(0xFF8A4B00))
        }
        normalized.substringBefore("->").trim().toIntOrNull() == 1 || normalized == "咳嗽" -> {
            LabelColorStyle(bg = Color(0xFFFFCDD2), fg = Color(0xFF8B0000))
        }
        normalized.substringBefore("->").trim().toIntOrNull() == 2 || normalized == "锻炼" -> {
            LabelColorStyle(bg = Color(0xFFC8E6C9), fg = Color(0xFF1B5E20))
        }
        normalized.substringBefore("->").trim().toIntOrNull() == 3 || normalized == "鼻塞" -> {
            LabelColorStyle(bg = Color(0xFFE1BEE7), fg = Color(0xFF4A148C))
        }
        normalized.substringBefore("->").trim().toIntOrNull() == 4 || normalized == "鼻子呼吸" -> {
            LabelColorStyle(bg = Color(0xFFBBDEFB), fg = Color(0xFF0D47A1))
        }
        else -> LabelColorStyle(bg = Color(0xFFE0E0E0), fg = Color(0xFF333333))
    }
}
