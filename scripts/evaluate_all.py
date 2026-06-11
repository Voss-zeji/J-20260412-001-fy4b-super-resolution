#!/usr/bin/env python3
import json, math
from pathlib import Path
from datetime import datetime

LV3_RESULTS = Path('/root/jobs/J-20260412-001-fy4b-super-resolution/lv3_fusion/results')
OUTPUT_DIR = Path('/root/jobs/J-20260412-001-fy4b-super-resolution/lv3_fusion')

def load_result(method_dir):
    result_file = method_dir / 'result.json'
    if not result_file.exists():
        return None
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return None

    method_name = method_dir.name
    psnr = data.get('best_psnr') or data.get('val_psnr') or data.get('final_psnr')
    ssim = data.get('best_ssim') or data.get('val_ssim') or data.get('final_ssim')
    epoch = data.get('best_epoch') or data.get('train_epochs') or data.get('epochs')

    # 排除无效值
    if psnr is None or (isinstance(psnr, float) and math.isinf(psnr)):
        return None

    return {
        'method': method_name,
        'band': data.get('band', 'CH07'),
        'status': data.get('status', 'unknown'),
        'psnr': round(float(psnr), 4),
        'ssim': round(float(ssim), 4) if ssim else None,
        'epoch': epoch,
        'params': data.get('model_params'),
        'infer_ms': data.get('inference_time_ms'),
        'runtime_min': data.get('runtime_minutes') or (data.get('runtime_seconds', 0) / 60),
        'psnr_source': 'best_checkpoint' if 'best_psnr' in data else 'final_epoch',
    }

def main():
    print('=' * 70)
    print('FY-4B SR — 统一评估体系 (排除无效值)')
    print('=' * 70)

    all_results = []
    skipped = []
    for method_dir in sorted(LV3_RESULTS.iterdir()):
        if not method_dir.is_dir():
            continue
        result = load_result(method_dir)
        if result is None:
            skipped.append(method_dir.name)
            continue
        all_results.append(result)

    all_results.sort(key=lambda x: x['psnr'], reverse=True)

    lines = []
    lines.append('# FY-4B 超分辨率 — 统一评估汇总 (200 epoch)\n')
    lines.append(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  ')
    lines.append(f'**评估口径**: 统一取 best checkpoint PSNR（如无则取 final epoch），排除 Bicubic(inf)  ')
    lines.append(f'**有效方法**: {len(all_results)} | **跳过**: {len(skipped)}  ')
    lines.append('')
    lines.append('## 完整排名 (PSNR 降序)\n')
    lines.append('| 排名 | 方法 | PSNR (dB) | SSIM | Epoch | 参数量 | 推理(ms) | 运行(分) | 数据来源 |')
    lines.append('|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|')

    for i, r in enumerate(all_results, 1):
        psnr = f"{r['psnr']:.2f}"
        ssim = f"{r['ssim']:.4f}" if r['ssim'] else '?'
        ep = str(r['epoch']) if r['epoch'] else '?'
        params = f"{r['params']:,}" if r['params'] else '?'
        infer = f"{r['infer_ms']:.1f}" if r['infer_ms'] else '?'
        rt = f"{r['runtime_min']:.0f}" if r['runtime_min'] else '?'
        lines.append(f"| {i} | {r['method']} | {psnr} | {ssim} | {ep} | {params} | {infer} | {rt} | {r['psnr_source']} |")

    lines.append('')
    lines.append('## Top-5 方法\n')
    top5 = all_results[:5]
    for i, r in enumerate(top5, 1):
        lines.append(f'### {i}. {r["method"]}')
        lines.append(f"- **PSNR**: {r['psnr']:.2f} dB")
        lines.append(f"- **SSIM**: {r['ssim']:.4f}")
        if r['params']: lines.append(f"- **参数量**: {r['params']:,}")
        if r['infer_ms']: lines.append(f"- **推理时间**: {r['infer_ms']:.1f} ms")
        lines.append(f"- **训练时长**: {r['runtime_min']:.0f} 分钟")
        lines.append(f"- **数据来源**: {r['psnr_source']}")
        lines.append('')

    lines.append('## 全部方法柱状图数据\n')
    lines.append('')

    if skipped:
        lines.append('')
        lines.append('## 跳过/无效的方法\n')
        lines.append(', '.join(skipped))

    md_content = '\n'.join(lines)
    md_path = OUTPUT_DIR / 'UNIFIED_EVALUATION.md'
    json_path = OUTPUT_DIR / 'unified_summary.json'

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'generated_at': datetime.now().isoformat(), 'total': len(all_results), 'skipped': skipped, 'results': all_results}, f, indent=2, ensure_ascii=False)

    print(f'\n✅ 统一评估完成!')
    print(f'   Markdown: {md_path}')
    print(f'   JSON: {json_path}')
    print(f'\n📊 Top-5 (排除 Bicubic):')
    for i, r in enumerate(top5, 1):
        print(f"   {i}. {r['method']}: {r['psnr']:.2f} dB, SSIM={r['ssim']:.4f}")

if __name__ == '__main__':
    main()
