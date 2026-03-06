# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-03-04

### Added
- 字母识别模型（A-Z，26 类），使用 `enhanced_reslstm` 架构
- 数字识别模型（0-9），支持两阶段搜索与单文件预测
- 健康场景识别（非深度学习扫描 + 深度学习分类）
- 双通道/单通道训练支持（`dual/ch1/ch2`）
- 词典约束动态规划解码（`lexicon_commands.txt`）
- 多 seed 子集集成训练流程
- 完整单元测试套件（`morse_train/tests/`）
- HTML 训练报告生成（`run_report.py`）
- 信号可视化（`run_visualize.py`）

### Changed
- 数字训练集扩充至 52 个文件（新增 `0/5/7/9` 共 7 个文件）

[Unreleased]: https://github.com/xinhaikoukoumi/huxi/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/xinhaikoukoumi/huxi/releases/tag/v1.0.0
