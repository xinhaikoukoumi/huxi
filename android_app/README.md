# android_app - 实时呼吸监测与识别演示客户端

本文档基于当前源码整理，更新时间为 2026-03-22。`android_app/` 是 `huxi` 仓库里的 Android 端完整演示客户端，用来承接 `morse_train/` 导出的模型，并在手机端串通“呼吸监测 + 本地账户 + 模型推理 + 数据管理”主流程。

## 1. 项目定位

本客户端使用 Kotlin + Jetpack Compose + PyTorch Android Lite 开发，当前更接近可运行原型 / 演示版，重点在于：

- 提供 Android 端交互界面
- 演示呼吸信号回放与实时频率估算
- 演示健康 / 数字 / 字母三类模型在端侧的加载与推理
- 将 `morse_train/` 的模型研究成果落到移动端

## 2. 当前已实现能力

### 2.1 本地账户与数据管理

- 本地注册、登录
- 用户资料编辑
- 呼吸频率历史记录保存与清空
- 退出登录与本地账号注销
- 数据存储基于 `SharedPreferences`

### 2.2 呼吸监测主界面

- 数值模式与波形模式切换
- 从 `app/src/main/assets/demo_signals/` 读取演示 CSV
- 按顺序循环播放演示信号
- 根据波峰周期估算呼吸频率
- 在识别开启后同步展示模型标签

### 2.3 模型识别模块

- 健康运动分析
- 摩斯 / 数字编码识别
- 数字词库映射与解析展示
- 模型路径支持手工输入
- 支持尝试加载 `.ptl` 并反馈成功 / 失败原因

说明：当前代码已经接入真实 PyTorch Lite 推理，不只是 UI 占位。

## 3. 和 `morse_train/` 的关系

推荐把本目录理解为 `morse_train/` 的 Android 客户端：

- `morse_train/` 负责训练、评估、导出模型
- `android_app/` 负责本地 UI、信号播放、端侧推理和演示
- `morse_train/android_integration_sample/` 提供最小接入范例
- `android_app/` 提供完整客户端工程

## 4. 模型文件接入方式

为了遵守仓库卫生规则，本仓库不会继续把新的 `.ptl` 二进制直接提交进 Git 历史。因此，当前目录默认只保留模型目录说明文件，不直接跟踪模型本体。

你有两种使用方式：

### 4.1 方式一：通过 App UI 直接填写本机模型绝对路径

App 已支持直接加载本地文件路径，只要指向你本机上真实存在的 `.ptl` 文件即可。

### 4.2 方式二：把模型复制到 `assets` 后本地运行

如果希望使用默认的 `asset:///...` 路径，可以把本地导出的 `.ptl` 文件复制到：

- `android_app/app/src/main/assets/models/health_status_recognition_best.ptl`
- `android_app/app/src/main/assets/models/digits_recognition_best.ptl`
- `android_app/app/src/main/assets/models/letters_recognition_best.ptl`

### 4.3 模型来源

推荐从 `morse_train/` 导出 Android 模型，相关脚本在：

- `morse_train/run_export_android_task_models.py`
- `morse_train/run_export_android_digit_models.py`

如果你只需要看最小接入逻辑，可参考：

- `morse_train/android_integration_sample/README.md`

## 5. 演示数据

当前仓库保留了可直接运行的演示信号：

- `app/src/main/assets/demo_signals/yundong/`
- `app/src/main/assets/demo_signals/num/`
- `app/src/main/assets/demo_signals/zimu/`

加载规则：

1. 仅读取 `.csv`
2. 按文件名顺序播放
3. 单文件结束后自动切到下一个
4. 播放完后循环

## 6. 技术实现概览

### 6.1 技术栈

- Kotlin
- Jetpack Compose
- Material 3
- PyTorch Android Lite
- SharedPreferences
- Gradle Kotlin DSL

### 6.2 推理链路

当前推理链路包括：

1. 读取演示 CSV
2. 解析单通道或双通道时间序列
3. 信号补齐、裁剪、平滑、去趋势、归一化
4. 固定或自适应分窗
5. 重采样到固定长度输入
6. 调用 `.ptl` 执行前向推理
7. softmax 取最大标签
8. 将窗口结果映射回实时播放界面

## 7. 代码结构

- `app/src/main/java/com/example/myapp/MainActivity.kt`
  - Activity 入口
- `app/src/main/java/com/example/myapp/ui/BreathMonitorApp.kt`
  - 应用级状态编排
- `app/src/main/java/com/example/myapp/ui/home/`
  - 主页、监测面板、健康分析、数字识别、词库、数据管理
- `app/src/main/java/com/example/myapp/data/UserRepository.kt`
  - 本地数据存储
- `app/src/main/java/com/example/myapp/util/AssetModelLoader.kt`
  - 模型加载
- `app/src/main/java/com/example/myapp/util/SignalTaskInference.kt`
  - 分窗、预处理、重采样、推理

## 8. 运行与验证

### 8.1 运行单元测试

```powershell
cd android_app
.\gradlew.bat testDebugUnitTest --no-daemon
```

### 8.2 构建调试版本

```powershell
cd android_app
.\gradlew.bat assembleDebug
```

说明：

- APK 不会提交进仓库
- `.ptl` 不会提交进仓库
- 构建产物请只保留在本地或外部制品存储

## 9. 当前限制

- 仍以本地演示信号为主，尚未接入真实传感器
- 用户数据仅本地存储，不支持云端同步
- 测试覆盖仍偏少
- 健康 / 数字 / 字母识别效果仍依赖外部导出的模型质量
- 端侧模型管理目前仍偏演示化，尚未接入正式发布流程

## 10. 一句话总结

`android_app/` 是 `huxi` 仓库中的 Android 端完整演示客户端，负责把 `morse_train/` 的模型能力落到移动端 UI 和交互流程里。
