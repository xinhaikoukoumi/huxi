# Contributing Guide

感谢你为 `huxi` 项目贡献代码。为了保证仓库可维护、可复现，请在提交前遵守以下规则。

## 开发流程

1. 从最新 `main` 分支创建功能分支。
2. 每个 PR 聚焦单一主题，避免混合“功能 + 大量无关重构”。
3. 提交前运行测试并更新必要文档。

## Commit Message 规范

推荐使用 Conventional Commits：

- `feat: ...` 新功能
- `fix: ...` 缺陷修复
- `docs: ...` 文档变更
- `refactor: ...` 重构（无行为变化）
- `test: ...` 测试相关
- `ci: ...` 持续集成相关
- `chore: ...` 杂项维护

示例：

```text
fix: make preprocess tests independent from local dataset
```

## 代码与测试要求

- Python 依赖安装：

```bash
cd morse_train
pip install -r requirements.txt
```

- 提交前至少通过：

```bash
cd morse_train
python -m pytest -q
```

## 数据与产物管理

- 不要将新的大体积二进制文件直接提交进 Git 历史（如 `zip/pt/pth/npz`）。
- 推荐将模型和数据通过 Release、外部对象存储或 Git LFS 分发。
- 实验中间产物请保存在被 `.gitignore` 忽略的目录中。

## Pull Request 要求

- 标题清晰描述变更目的。
- 描述中包含：
  - 变更内容
  - 验证方式（命令和结果）
  - 潜在风险与回滚方式（如有）
