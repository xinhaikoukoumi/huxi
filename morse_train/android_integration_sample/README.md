# Android Integration (PyTorch Lite)

This sample integrates 4 task-level models:

- `letters`
- `digits`
- `health_seed52`
- `health_seed62`

## 1) Add dependencies

```gradle
dependencies {
    implementation "org.pytorch:pytorch_android_lite:2.4.0"
    implementation "org.pytorch:pytorch_android_torchvision_lite:2.4.0"
}
```

## 2) Place model assets

Copy the exported folders from `android_export_manifest.json` into your Android assets:

- `assets/models/letters/morse_char_model_android.ptl`
- `assets/models/letters/label_map.json`
- `assets/models/digits/morse_char_model_android.ptl`
- `assets/models/digits/label_map.json`
- `assets/models/health_seed52/morse_char_model_android.ptl`
- `assets/models/health_seed52/label_map.json`
- `assets/models/health_seed62/morse_char_model_android.ptl`
- `assets/models/health_seed62/label_map.json`

## 3) Input tensor shape

- Letters: `[1, C, 300]` (`C` follows training config, usually `2`)
- Digits: `[1, C, 300]` (current best is `C=1`)
- Health: `[1, 2, 300]`

## 4) Inference rule

- Letters/Digits: single-model logits -> softmax -> `argmax`.
- Health: run seed52 and seed62 logits, average element-wise, then softmax -> `argmax`.

## 5) First release strategy

Use pure model output only:

- no lexicon decoding
- no offset search
- no adaptive boundary alignment

Keep preprocessing aligned with Python (`clip/median/smooth/detrend/resample-to-300`) to preserve accuracy.
