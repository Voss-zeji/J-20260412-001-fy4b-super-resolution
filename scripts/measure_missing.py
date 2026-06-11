#!/usr/bin/env python3
"""
补全缺失的参数量和推理时间
遍历 lv2_micro/lv2-save/ 中缺少字段的方法，只创建模型、不训练
"""

import json, sys, time, subprocess, os
from pathlib import Path

PROJECT_ROOT = Path('/root/jobs/J-20260412-001-fy4b-super-resolution')
LV3_RESULTS = PROJECT_ROOT / 'lv3_fusion' / 'results'
LV2_SAVE = PROJECT_ROOT / 'lv2_micro' / 'lv2-save'
PYTHON_BIN = '/root/miniconda3/envs/mamba2/bin/python'

METHODS_TO_FIX = [
    '15_method_ntire2026_ir_sr',
    '17_method_emambair',
    '18_method_gprof_ir',
    '19_method_impa_net',
    '20_method_multispectral_sr',
]

def measure_method(name):
    source_dir = LV2_SAVE / name
    main_py = source_dir / 'main.py'
    result_file = LV3_RESULTS / name / 'result.json'

    if not main_py.exists():
        print(f'  [跳过] {main_py} 不存在')
        return
    if not result_file.exists():
        print(f'  [跳过] {result_file} 不存在')
        return

    # 读取现有结果
    with open(result_file) as f:
        data = json.load(f)

    # 如果已经有完整字段，跳过
    if data.get('model_params') and data.get('inference_time_ms'):
        print(f'  {name}: 已有完整字段，跳过')
        return

    print(f'  {name}: 测量中...')

    # 写临时脚本：只创建模型并测速
    tmp_script = source_dir / '_measure_tmp.py'
    tmp_script.write_text(f'''
import sys, torch, json, time
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# 导入模型（通过 main.py 中的定义）
exec(open('{main_py}').read().replace('def main():', 'def _orig_main():'))

# 尝试找到模型类
import types
for obj_name in globals():
    obj = globals()[obj_name]
    if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj_name not in ['Module']:
        try:
            model = obj()
            break
        except:
            continue
else:
    # 如果没找到，尝试常见类名
    for cls_name in ['EmambaIR', 'NTIRE2026IRSR', 'GPROFIR', 'IMPANet', 'MultiSpectralSR', 'SwinIR']:
        if cls_name in globals():
            model = globals()[cls_name]()
            break

model_params = sum(p.numel() for p in model.parameters())
device = torch.device('cuda')
model = model.to(device)
dummy = torch.randn(1, 1, 64, 64).to(device)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad(): _ = model(dummy)
torch.cuda.synchronize()
infer_ms = (time.time() - t0) * 1000

result = {{'params': model_params, 'infer_ms': round(infer_ms, 2)}}
print(json.dumps(result))
''')

    try:
        proc = subprocess.run([PYTHON_BIN, str(tmp_script)], capture_output=True, text=True, timeout=60)
        output = proc.stdout.strip().split('\n')[-1]
        measured = json.loads(output)

        # 更新 result.json
        data['model_params'] = measured['params']
        data['inference_time_ms'] = measured['infer_ms']
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'    -> Params: {measured["params"]:,}, Infer: {measured["infer_ms"]:.1f}ms')
    except Exception as e:
        print(f'    [失败] {e}')
    finally:
        if tmp_script.exists():
            tmp_script.unlink()

def main():
    print('=' * 60)
    print('补全缺失的参数量和推理时间')
    print('=' * 60)
    for name in METHODS_TO_FIX:
        measure_method(name)
    print('\n✅ 完成')

if __name__ == '__main__':
    main()
