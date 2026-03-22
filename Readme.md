# huxi 仓库总览

本仓库当前包含两条协同工作线：

1. `morse_train/`
   - 负责字母、数字、健康呼吸场景三类任务的训练、评估、导出和报告生成。
2. `android_app/`
   - 负责 Android 端完整演示客户端，串通“呼吸监测 + 本地账户 + 模型推理 + 数据管理”主流程。

`morse_train` 是模型与算法主线，`android_app` 是移动端落地主线。

## 1. 仓库结构

```text
huxi/
├─ morse_train/                      # Python 训练/评估/导出主工程
│  ├─ src/
│  ├─ tests/
│  ├─ README.md
│  ├─ PROJECT_TUTORIAL.md
│  ├─ run_export_android_task_models.py
│  ├─ run_export_android_digit_models.py
│  └─ android_integration_sample/    # 轻量 Android 接入示例
├─ android_app/                      # 完整 Android 演示客户端
│  ├─ app/
│  ├─ gradle/
│  ├─ README.md
│  └─ gradlew.bat
├─ .github/
└─ REPO_HYGIENE.md
```

## 2. 快速导航

- 想看训练、评估、模型效果和完整命令：看 `morse_train/README.md`
- 想看更细的教学式说明：看 `morse_train/PROJECT_TUTORIAL.md`
- 想看 Android 端完整客户端：看 `android_app/README.md`
- 想看最小 Android 接入范例：看 `morse_train/android_integration_sample/README.md`

## 3. 快速开始

### 3.1 Python 训练与测试

```powershell
cd morse_train
pip install -r requirements.txt
python -m pytest -q
```

### 3.2 Android 客户端验证

```powershell
cd android_app
.\gradlew.bat testDebugUnitTest --no-daemon
```

## 4. Android 端模型接入方式

Android 客户端已经并入本仓库，但为了遵守仓库卫生规则，`.ptl` 和 APK 这类新的二进制产物不会直接提交进 Git 历史。

推荐流程：

1. 先在 `morse_train/` 中完成训练或选定现有最佳模型。
2. 使用以下脚本导出 Android 可用的 TorchScript Lite 模型：
   - `run_export_android_task_models.py`
   - `run_export_android_digit_models.py`
3. 将导出的 `.ptl` 文件保存在本地工作目录中：
   - 可复制到 `android_app/app/src/main/assets/models/` 后本地运行
   - 或直接在 App 界面里填写本机绝对路径加载

说明：

- `morse_train/android_integration_sample/` 提供的是最小接入示例。
- `android_app/` 提供的是完整可运行客户端。
- 本仓库默认忽略 `.ptl`、APK 和 Android 构建缓存，避免继续污染 Git 历史。

## 5. 当前仓库定位

截至 2026-03-22，本仓库已经具备：

- 字母、数字、健康三类任务的训练与验证链路
- Android 端侧模型加载与推理落地代码
- Android 演示客户端源码
- Python 测试、PR 模板和基础 CI 工作流

## 6. 仓库协作约束

提交前请优先检查以下事项：

- 不要新增 `.zip`、`.pt`、`.pth`、`.ptl`、`.apk` 等大体积二进制到 Git 历史
- 优先走分支 + PR 流程，遵守 `.github/pull_request_template.md`
- 修改 Python 侧后至少运行 `python -m pytest -q`
- 修改 Android 侧后至少运行 `.\gradlew.bat testDebugUnitTest --no-daemon`
- 如需清理历史大文件，先阅读 `REPO_HYGIENE.md`

## 7. 一句话说明

`huxi` 现在既包含呼吸信号识别模型研发主线，也包含 Android 端演示客户端，是一个“训练导出 + 移动端接入”闭环仓库。
