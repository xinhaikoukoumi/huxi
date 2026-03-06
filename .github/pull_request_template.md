## Summary

<!-- 1-3 句概括本次 PR 目标 -->
- 

## Background

<!-- 为什么要改；不改会有什么影响 -->
- 

## Scope

<!-- 变更范围（代码/文档/CI/数据处理等） -->
- In scope:
  - 
- Out of scope:
  - 

## Changes

<!-- 按文件或模块列出关键改动 -->
1. 
2. 
3. 

## Breaking Changes

<!-- 如无，请写：None -->
- None

## Validation

<!-- 请贴实际执行过的命令和关键结果 -->
```bash
cd morse_train
python -m pytest -q
```

Result:

```text
# paste key output
```

Additional evidence (if behavior changed):
- Before:
- After:
- Logs/Screenshots/Artifacts:

## Risk Assessment

<!-- 说明风险和影响面 -->
- Risk level: Low / Medium / High
- Potential impact:
  - 
- Mitigations:
  - 

## Rollback Plan

<!-- 出问题时如何回滚 -->
1. 
2. 

## Data/Model Artifacts

<!-- 是否新增或修改大文件（zip/pt/pth/npz 等） -->
- [ ] No new large binary artifacts are added into git history
- [ ] If artifacts are required, they are stored outside git history (Release/LFS/object storage)

## Related Issues

<!-- 例如: Closes #123 -->
- 

## Checklist

- [ ] PR scope is focused (single topic)
- [ ] Commit messages follow `type: summary`
- [ ] Documentation is updated when behavior/config changes
- [ ] Tests are added/updated for the changed behavior
- [ ] Local tests pass
- [ ] CI workflow changes (if any) are justified and minimal
