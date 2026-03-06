# Repository Hygiene Checklist

本文件用于处理“历史中已存在大量二进制文件”的情况。

## 1. 查看当前仓库大文件

```bash
git ls-files | xargs -I{} sh -c 'test -f "{}" && du -h "{}"' | sort -hr | head -n 50
```

Windows PowerShell 可用：

```powershell
$files = git -c core.quotePath=false ls-files
$items = foreach($f in $files){ if(Test-Path $f){ Get-Item $f } }
$items | Sort-Object Length -Descending | Select-Object -First 50 FullName,Length
```

## 2. 将大文件从索引移除（保留本地文件）

按需执行（会产生一次“删除跟踪”的提交）：

```bash
git rm -r --cached -- '*.zip' '*.npz' '*.pt' '*.pth'
git rm -r --cached morse_train/experiments morse_train/model_output
```

然后提交：

```bash
git commit -m "chore: stop tracking large generated artifacts"
```

## 3. 清理历史（可选，破坏性操作）

如果仓库历史已被大文件污染，建议在团队确认后使用 `git filter-repo` 或 BFG 清理历史并强推。  
这一步会改写历史，只应在全员确认后执行。
