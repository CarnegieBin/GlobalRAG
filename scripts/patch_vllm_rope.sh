#!/bin/bash
# 临时脚本：修复 vllm 0.6.3 不支持 rope_scaling type="default"（Qwen3）的问题

FPATH="/home/work/miniconda3/envs/globalrag/lib/python3.10/site-packages/vllm/model_executor/layers/rotary_embedding.py"

# Step 1: 还原备份并重新打补丁（自动检测缩进）
python - <<EOF
import shutil, os

fpath = "$FPATH"

# 如果已有备份则先还原，保证幂等
if os.path.exists(fpath + '.bak'):
    shutil.copy(fpath + '.bak', fpath)
    print(f"Restored from backup: {fpath}.bak")
else:
    shutil.copy(fpath, fpath + '.bak')
    print(f"Backup created: {fpath}.bak")

with open(fpath) as f:
    lines = f.readlines()

target = 'raise ValueError(f"Unknown RoPE scaling type {scaling_type}")'
new_lines = []
patched = False

for line in lines:
    if target in line and not patched:
        indent = len(line) - len(line.lstrip())
        ind  = ' ' * indent
        ind4 = ' ' * (indent + 4)
        new_lines.append(f'{ind}if scaling_type == "default":\\n')
        new_lines.append(f'{ind4}rope_scaling = None\\n')
        new_lines.append(f'{ind4}return get_rope(head_size, rotary_base, max_position, rope_scaling)\\n')
        new_lines.append(line)
        patched = True
        print(f"Patched (raise indent={indent} spaces)")
    else:
        new_lines.append(line)

if patched:
    with open(fpath, 'w') as f:
        f.writelines(new_lines)
    print(f"Written: {fpath}")
else:
    print("ERROR: target pattern not found in file")
EOF

# Step 2: 语法检查
echo ""
echo "=== Syntax check ==="
python -m py_compile "$FPATH" && echo "Syntax OK" || echo "Syntax ERROR"

# Step 3: 功能验证（文件语法正常后再 import）
echo ""
echo "=== Verify patch ==="
python -c "
import vllm.model_executor.layers.rotary_embedding as m
import inspect
src = inspect.getsource(m.get_rope)
if 'default' in src:
    print('OK: get_rope now handles rope_scaling type=default')
else:
    print('FAILED: patch not detected')
"
