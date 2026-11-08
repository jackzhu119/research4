# V7元数据勘误与已完成的partial差分

本增补保留[原操作说明](PYSCF_V7_XC_CHUNK_RECOVERY_20261105.md)及其SHA `5d6cfa9b75e0017401cb886f64b0777b77ea8912968641799715c96e9ad3f7c7`，不修改冻结core `da1e2b...`、worker `00e710eb...`、旧索引或失败结果。

## 必须更正的索引约定

原说明将component index引用的`bytes`表述为可选，这对本worker的**依赖字典恒等门禁**不成立。冻结worker第242行要求`chunk.plan_result_dependency == index.XC_plan`；实际chunk写入的是仅含`path`与`sha256`的字典。给index的XC_plan添加bytes即使内容相同，也会拒绝。

为保持所有依赖引用一致，`component_index.json`中RHF_partial、XC_plan和每个XC_chunks引用**统一恰好两个键**，使用相同绝对路径字符串与真实SHA，不附bytes、描述或其他字段：

```json
{"path": "ACTUAL_ABSOLUTE_RESULT_JSON", "sha256": "ACTUAL_SHA256"}
```

顶层仍仅为schema、RHF_partial、XC_plan、XC_chunks；schema仍是`research4-v7-XC-component-index-v1`。这只纠正操作元数据，不改变grid、数学式、source/G身份、参数、块范围、阈值或任何数值计算。

实际`control-water-components-001.json`含bytes，导致`control-water-rks-partial-001/result.json`以PARTIAL、`Chunk source/G/plan changed`失败。root完成逐字段核验：六份chunk的G、boot、kind均一致，失配在plan引用字典。root通报该失败parent闭合CPU为0.855151秒，照常计费；worker自身processCPU 0.219996598秒是更窄的计量范围，不能替换parent闭合总成本。旧失败全部保留。

root创建仅两键引用的`control-water-components-002.json`后，独立新phase `control-water-rks-partial-002`实际STAGE_COMPLETE。其result SHA为`81807f2352ce30e3a05e57fb88d6e7e80eaa4d7a00d58af515ae0ee98490ed18`；没有重算已健康关闭的RHF或六份XC贡献。

## 真实完成的水／乙醇partial对照

下表从各自已完成的`partial_differential.json`读取，固定绝对阈值均为1e−8。这里是原生完整AO及RKS partial的全元素差分，**尚不包括CPHF、核Hessian或完整频率对照**。

| 实际对照 | 水最大绝对差 | 乙醇最大绝对差 |
| --- | ---: | ---: |
| XCdiag完整AO | 7.105427357601002e−14 | 7.105427357601002e−14 |
| XCderiv2完整AO | 2.842170943040401e−14 | 2.842170943040401e−14 |
| RHF partial | 1.7763568394002505e−14 | 1.2434497875801753e−14 |
| 完整RKS partial | 3.2951419370874646e−13 | 1.1248779685502086e−12 |

两份实际status均为`PASS_ACTUAL_NATIVE_AO_AND_FULL_PARTIAL_DIFFERENTIAL`：

- [水partial差分](../reports/pyscf-spinosin-grid-checkpoint-20261105-v7/control-water-partial-compare-001/partial_differential.json)，SHA `6317a2b188e8eabda4fcaeda12582da44fc2ed49a495d32801076666f2d2e621`。
- [乙醇partial差分](../reports/pyscf-spinosin-grid-checkpoint-20261105-v7/control-ethanol-partial-compare-001/partial_differential.json)，SHA `2aa557c8df9313a37fe03bafb743493ba1faaf2c8e6c5cce8701ec2ae374cf25`。

本次文档整理只读取完成的小JSON，不独立重算AO／Hessian；不能写成第二次独立数值重放。读取时两个result均明确`frequencies_status=NOT_RUN`、`complete_Hessian_frequency_control_required_separately=true`。后续真实B／C与新V7分析器完成情况以各自新产物为准，不回写此快照。当前partial成功不等于Spinosin成功，也不产生理论IR强度或分类准确率改善。
