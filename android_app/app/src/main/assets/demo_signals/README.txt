Put your demo CSV files in this folder.

Folder layout:
- demo_signals/num/      -> numeric mode demo signals
- demo_signals/zimu/     -> letter mode demo signals
- demo_signals/yundong/  -> exercise analysis demo signals

Loading rules:
1) Files are read in filename order (.csv only).
2) Each file is used as signal input from top to bottom.
3) After one file ends, the next file starts automatically.
4) After all files end, playback loops from the first file.

CSV parsing rule:
- For each line, the LAST numeric column is used as the signal value.
- Empty lines and non-numeric lines are ignored.

Examples:
- time,value
- 0.00,0.12
- 0.06,0.18

Recommended path:
app/src/main/assets/demo_signals/
