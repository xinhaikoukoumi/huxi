# 贡献指南 / Contributing Guide

感谢您有兴趣为 **huxi** 项目作出贡献！请在提交 PR 或 Issue 之前阅读本指南。

---

## 目录 / Table of Contents

- [行为准则](#行为准则)
- [如何报告 Bug](#如何报告-bug)
- [如何提出新功能](#如何提出新功能)
- [开发流程](#开发流程)
- [提交消息规范](#提交消息规范)
- [代码风格](#代码风格)
- [测试](#测试)

---

## 行为准则

请尊重所有参与者，保持友好、专业的交流氛围。

---

## 如何报告 Bug

1. 确认该 Bug 尚未在 [Issues](https://github.com/xinhaikoukoumi/huxi/issues) 中被报告。
2. 使用 **Bug Report** 模板创建新 Issue。
3. 请提供以下信息：
   - 操作系统与 Python 版本
   - 完整的错误信息与堆栈跟踪
   - 最小可复现示例

---

## 如何提出新功能

1. 在 [Issues](https://github.com/xinhaikoukoumi/huxi/issues) 中使用 **Feature Request** 模板描述您的想法。
2. 等待维护者反馈后再开始实现，避免重复劳动。

---

## 开发流程

```bash
# 1. Fork 本仓库，然后克隆您的 Fork
git clone https://github.com/<your-username>/huxi.git
cd huxi

# 2. 创建功能分支（基于 main）
git checkout -b feat/your-feature-name

# 3. 安装依赖
cd morse_train
pip install -r requirements.txt

# 4. 进行修改并运行测试
python -m pytest -q

# 5. 提交更改（遵循提交消息规范）
git add .
git commit -m "feat: 添加新功能描述"

# 6. 推送并创建 Pull Request
git push origin feat/your-feature-name
```

---

## 提交消息规范

本项目遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/) 规范。

### 格式

```
<type>(<scope>): <subject>

[body]

[footer]
```

### 类型（type）

| 类型       | 说明                               |
|------------|----------------------------------|
| `feat`     | 新功能                             |
| `fix`      | Bug 修复                          |
| `docs`     | 仅文档变更                         |
| `style`    | 不影响代码逻辑的格式调整（空格、缩进等） |
| `refactor` | 既不修复 Bug 也不添加新功能的代码重构  |
| `test`     | 添加或修改测试                     |
| `chore`    | 构建过程或辅助工具变动               |
| `perf`     | 性能优化                           |
| `ci`       | CI/CD 配置相关变更                 |

### 示例

```
feat(model): 添加 Transformer 编码器变体

在 model.py 中新增 transformer_encoder 选项，支持通过 --model_variant 参数指定。

Closes #42
```

```
fix(preprocess): 修复空 CSV 文件导致崩溃的问题

当 zip 内 CSV 为空时，现在会跳过该文件并记录警告，而不是抛出异常。
```

```
docs: 修复 Readme.md 中代码块格式错误
```

---

## 代码风格

- Python 代码遵循 [PEP 8](https://pep8.org/) 规范。
- 建议使用 `flake8` 或 `ruff` 检查代码风格：

  ```bash
  pip install flake8
  flake8 morse_train/src/ morse_train/tests/
  ```

---

## 测试

```bash
cd morse_train
python -m pytest -q
```

提交 PR 前请确保所有测试通过。若新增功能，请同步添加对应测试用例。
