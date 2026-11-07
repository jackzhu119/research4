# V8 新96小时授权下的独立预算守卫

状态：**准备完成，作者未注册新账本、未启动量子计算、未执行 Git**。

- `scripts/run_pyscf_continuation_budget_20261105_v8.py`：`7bc773076ecc5e0f90fb58b81bf0641948dd09c14b2816138a0469f16e511c65`。
- `tests/test_pyscf_continuation_budget_v8.py`：`e12c6448adfcce6d6030a72b520d28c68f660e8b9c1d49b3200d0681bd63eaee`，36项标准库测试通过；独立复跑36项通过。
- 独立审查：`reports/pyscf-v8-guard-independent-review-20261105-v1/review.json`，`e1c721f8d08e3714cf44a935ff642da9404049aaa25644529bdb11b2174a2aa0`。

## 保留历史及精确预算

原 V6b 守卫 b985、旧账本及全部历史结果不改写。新账本复制完整133项历史phase前缀，包括失败响应的实际7080.874229 CPU秒。源旧账本实际已用为210160.8308359999；新cap为该原精度实际值加180000，等于390160.8308359999 CPU秒。初始新余量恰为180000 CPU秒，不另加旧剩余5839.169164，不做退款或重复分配。

用户新的96小时授权观测于2026-11-05T12:33:48Z，保守截止固定为2026-11-09T12:30:00Z；不从注册/恢复时间重启计时。两线程、虚拟AS24GiB、RSS10GiB/cgroup12GiB和原16GiB环境保持不变。单阶段仍须明确有界请求，首个真实B8响应21600 CPU秒/14400墙钟秒。

## 证明分派

Live只接受真实新V8父进程、源码SHA、PIDbirth/UID/PGID/session及账本。旧V6b live证明不能用作新执行授权。Closed接口先核验不可变proof字节，再按schema、supervisor路径和SHA精确分派：旧v6b回交未修改的b985完整闭合验证，新v8执行自身验证；未知类型拒绝，旧证明不重标为v8。消费者仍须将旧A/H1交给原旧worker资格函数，而不是以新consumer源码SHA冒充旧缓存来源。

实际CPU/RSS/PID、终态reap、信号身份绑定和phase_limits核心函数原文继承；独审确认23个共享函数AST一致。新层仅管理新授权、单次注册claim、历史前缀、cap、截止和两时代证明。守卫测试不代表分子计算完成，也不替代新consumer水/乙醇实际数值控制。

## Root后续操作

复制禁用模板 `reports/pyscf-spinosin-96h-continuation-20261105-v8/resource-protocol/authorization_v8_TEMPLATE_DISABLED.json` 到独立root批准文件，核对实际源码/授权/源账本SHA和source_base_commit后将root_approved_for_execution设true。保留禁用模板不变。

注册CLI（所有ACTUAL参数必须来自真实文件）：

```text
.runtime/bin/python scripts/run_pyscf_continuation_budget_20261105_v8.py register
 --ledger reports/pyscf-spinosin-96h-continuation-20261105-v8/cpu_budget.json
 --source-v6b-ledger reports/pyscf-spinosin-staged-20261105-v6b/cpu_budget.json
 --source-v6b-sha256 cbd66979a1888893ccae6fd90d3e1d65618f7c988bea5da22f6d758d63a87328
 --authorization ACTUAL_ROOT_APPROVED_JSON --authorization-sha256 ACTUAL_SHA
 --source-base-commit ACTUAL_FULL40
```

注册不启动Q。注册claim按原V6b预算身份全局防重复，另一个target/授权文件不能重复开户；claim写入后若发布账本前异常须root取证，不能删除claim自动重试。新阶段通过`run`调用，并始终使用同一新账本、唯一phase和新log/output。旧全局和V6b/V5/V4/V3 lineage锁继续持有；严禁其他旧队列另行花费旧余量，新旧源账本SHA若漂移会fail closed。

原响应没有保存MO1/MO_E1，不能从CPU中断位置恢复；复用完整A/H1后，需要新独立响应attempt。全部成功和失败成本继续累计，IR仍NOT_COMPUTED；只有真实完整Hessian和219带符号模式及独审后才能报告频率。
