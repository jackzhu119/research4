#!/usr/bin/env python3
"""Collect actual module receipts without turning an absent result into success."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/focused_followup_20261116'


def main():
    runtime = REPORT / 'runtime/state.json'
    state = json.loads(runtime.read_text()) if runtime.exists() else {}
    receipts = {}
    for path in sorted((REPORT / 'task_results').glob('*.json')):
        if path.stem != 'FINAL':
            receipts[path.stem] = json.loads(path.read_text())
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'experiment_id': 'focused-followup-20261116-v1',
               'receipts': receipts, 'runtime_state': state,
               'paper_modified': False,
               'interpretation': 'Retrospective internal evidence. Repeated predictions do not create independent specimens or batches.'}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'module_status.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 有界补充实验执行报告', '',
             '本报告收集实际模块回执。缺失输出不视为完成。论文、原始工作簿和历史结果保持原字节。', '',
             '新授权预算为累计 990,000 CPU 秒；按固定队列执行，不按外折成绩决定继续、重试或停止。',
             '当前实验与旧 1,360,000 CPU 秒任务的账本独立，不重新启动旧任务。', '',
             '|任务|实际状态|证据入口|', '|---|---|---|']
    for name, receipt in receipts.items():
        status = receipt.get('status', 'UNKNOWN')
        detail = str(receipt.get('detail', receipt.get('reason', ''))).replace('|', '/').replace('\n', ' ')
        lines.append(f'|{name}|{status}|task_results/{name}.json; {detail}|')
    lines += ['', '## 问题与证据', '',
              'E0 的预测配对审计、共同完整重复、指标及固定存档预测的配对重采样区间见 e0/。',
              'E1 的旧 V4 同划分全谱桥接见 e1/；它与扩展 inner5×2 协议分开。',
              'E2 的跨流程、训练集变化、固定训练集 MC 随机性分开报告于 e2/。',
              'E3 的掩码与实际数值 gamma 参数交叉诊断见 e3/，不从四格挑选新的无偏最终模型。',
              'E4 的完整历史 attempts 与本轮实际运行成本见 e4/ 及 runtime/。',
              'E5 的单位、厚度与 PE 证据见 e5/；未知字段保持 unknown。',
              'E6 默认关闭，不新增 MLP 搜索；E7 的新批次采样协议与空模板见 e7/。', '',
              '## 仍不能说明什么', '',
              '重复划分和重采样区间不涵盖完整模型开发、历史选择或未来批次的不确定性。',
              '内部工作簿验证不等于真实新采购、制备和测量批次验证；未提供新批次时不训练。',
              '62 维分类器仍可能依赖完整 612 点 SNV，不能据此推算采集时间下降。',
              '掩码随机性诊断不证明 MC seed 下准确率稳定，排名上下文结果不建立化学因果作用。', '',
              '## 执行与恢复', '',
              '入口、冻结配置、全部 seed、队列和代码哈希见 experiments/focused_followup_20261116/。',
              'GitHub Actions 在每个原子任务后提交回执、日志和 CPU 账本；续跑读取同一任务身份与完成记录。',
              '平台暂停、停用 Actions、权限变化或服务中断仍可能阻断运行；状态文件和 Actions 日志记录实际情况。',
              '不保证当前聊天云环境被销毁后本地进程继续运行。持久执行只以 Actions 的实际运行与提交为证据。', '']
    (REPORT / 'EXPERIMENT_REPORT_zh.md').write_text('\n'.join(lines))
    task = os.getenv('FOCUSED_TASK_ID', 'FINAL')
    receipt_dir = REPORT / 'task_results'
    receipt_dir.mkdir(exist_ok=True)
    (receipt_dir / f'{task}.json').write_text(json.dumps({
        'status': 'COMPLETED', 'detail': 'Collected existing receipts without upgrading blocked module statuses.',
        'artifacts': ['reports/focused_followup_20261116/module_status.json',
                      'reports/focused_followup_20261116/EXPERIMENT_REPORT_zh.md']}, indent=2) + '\n')


if __name__ == '__main__':
    main()
