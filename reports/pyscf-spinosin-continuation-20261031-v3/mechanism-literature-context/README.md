# 机制解释文献背景台账

**状态：BACKGROUND_ONLY_NO_NEW_QUANTUM_RESULTS。**

此目录只保存文献元数据、少量带来源的公开片段、读取限制回执，以及当前安装版PySCF函数的静态源码核查。没有导入或执行PySCF，没有SCF、梯度、Hessian、频率计算，没有读取checkpoint，也没有填入或推测斯皮诺素理论峰位。后台计算状态应由其真实计算日志另行判断。

## 可以使用的背景结论

1. **选择斯皮诺素有药材化学背景依据。** 2023年酸枣仁质量控制综述将其与酸枣仁皂苷A/B列为关键Q-markers（S01，DOI `10.1016/j.jep.2023.116204`）。这支持把它选为代表性候选成分，不证明当前855份药材的该成分含量，也不证明其随匿名组变化。来源分类标志与化学质量标志不是同一项证据。
2. **孤立分子模式与固体药材谱有明确边界。** THz综述讨论柔软分子内运动、分子间运动、氢键网络与整个分子的集体位移（S02、S03）。孤立分子谐振DFT可以提供局部正常模式候选，却缺少周期晶格、完整分子间网络、基质、含水状态和固态集体运动。单纯频率接近不能确认化学归属。
3. **低频结果需要数值与方法敏感性检查。** S04讨论柔软低频模式的积分网格误差；S05展示某核碱基共晶的色散相互作用对结构与THz模拟的重要性。它们支持将当前无D3和试算网格的局限如实写明，但不能直接给出斯皮诺素误差大小、修正方向或通用缩放因子。S07补充低频RRHO近似与构象系综的热化学背景；准谐热力学校正不等于THz峰位修正。
4. **标准PySCF `harmonic_analysis`不输出IR强度。** 官方源码在相关位置写有`#TODO: IR intensity`（S06）。本环境PySCF2.14.0源码的AST静态检查确认其8个输出字段包括频率、正常模式、约化质量、振动温度及力常数，没有IR强度字段。未独立实现并验证偶极导数时，保持IR强度`NOT_COMPUTED`；不能画成有强度的理论吸收曲线。

## 引文与核验范围

| ID | 年份 | 文献／软件背景 | DOI或官方来源 | 证据读取方式 |
| --- | --- | --- | --- | --- |
| S01 | 2023 | 酸枣仁质量控制与Q-markers综述 | `10.1016/j.jep.2023.116204` | PubMed及出版商索引中的题名、日期、DOI和摘要片段；直接页受限 |
| S02 | 2014 | Terahertz Vibrations and Hydrogen-Bonded Networks in Crystals | `10.3390/cryst4020074` | 出版商索引及东北大学研究者条目；出版商直接页429 |
| S03 | 2023 | Investigating the function and design of molecular materials through terahertz vibrational spectroscopy | `10.1038/s41570-023-00487-w` | 直接读取思克莱德大学机构出版页，核对题名、作者、年份、DOI和摘要 |
| S04 | 2022 | How Reliable Are Modern Density Functional Approximations to Simulate Vibrational Spectroscopies? | `10.1021/acs.jpclett.2c01278` | 官方PubMed／ACS／PMC索引；直接页403或验证码 |
| S05 | 2011 | Noncovalent Interactions in Paired DNA Nucleobases Investigated by Terahertz Spectroscopy and Solid-State Density Functional Theory | `10.1021/jp111878h` | 官方PubMed摘要索引与ACS期刊目录／文章索引；直接页403 |
| S06 | 未断言出版年 | PySCF `pyscf.hessian.thermo.harmonic_analysis` | <https://pyscf.org/_modules/pyscf/hessian/thermo.html> | 官方源码页直接读取，并静态核对本地2.14.0源文件 |
| S07 | 2020 | GoodVibes: automated thermochemistry for heterogeneous computational chemistry data | `10.12688/f1000research.22758.1` | F1000出版索引、Oxford机构库及开发者软件库索引；不用于斯皮诺素峰归属 |

题名、年份、DOI与URL按实际返回证据逐项记录；**索引核验不表示已经阅读全文**。S03直接读到的是机构摘要和书目信息，而非出版商全文。S07的version1公开索引标记两位评审“approved with reservations”，本台账仅把它作为软件／热化学背景。不存在以其评价代替本项目计算验证的做法。

S01搜索索引还返回了与2023文章日期不协调的“2025版药典”内容，本台账不采用该药典版次、阈值或相关定量表述。需要这些断言时应另查原版药典。

## 没有找到及仍然缺少的证据

- 使用精确关键词`"spinosin" "terahertz"`及`"斯皮诺素" "太赫兹"`，本次未核实到斯皮诺素纯品THz测量／专门DFT归属论文。返回结果主要是把不同论文并列的作者页、刊物目录或无关条目，不能作为阳性证据。搜索回执见`spinosin_thz_search_receipts.json`。这是有限公开网页检索，不是系统综述，也不能证明文献绝不存在。
- 当前台账不提供纯品实测THz谱、斯皮诺素晶体结构或固态周期计算、实际药材斯皮诺素含量、组间成分含量差异，也不确认实验峰的化学归属。
- 后续解释至少应区分：真实计算的正常模式、文献支持的化学候选、纯品／化学测量确认。构象、泛函／色散、网格、驻点／虚频检查应使用真实新计算结果，不由本台账预填。
- 文献背景不改变现有分类器、标签、划分和96.02%历史开发指标，也不声称提升分类准确率。

## 文件与重现

- `sources.json`／`sources.csv`：机器可读书目、DOI、证据等级、可支持结论与限制。
- `snapshots/`：每篇仅保留精简书目信息和一小段带来源片段，不是原始HTML或全文镜像。
- `access_limitations.txt`：403／429／验证码／工具不可访问等真实回执；没有绕过限制。
- `pyscf_harmonic_analysis_static_check.json`与`pyscf_harmonic_analysis_excerpt.txt`：安装版函数静态核验及少量Apache-2.0代码片段。完整源文件SHA为`8143ebed7653bd79b4b8d7b2f5933671a33285b11aba47cf0d8134c229a031f8`。
- `source_snapshot_hashes.json`及`files_sha256.json`：来源片段和目录封存哈希。封存索引自身除外。

从仓库根目录运行下列静态核验命令，使用新输出目录避免覆盖；脚本只读取源文本和包元数据，不导入PySCF：

```bash
.runtime/bin/python reports/pyscf-spinosin-continuation-20261031-v3/mechanism-literature-context/inspect_pyscf_thermo_source.py --output /tmp/research4-pyscf-harmonic-source-review
```

没有下载付费全文；公开片段均短小且注明来源。无法核实许可的文章只保留书目信息与有限引用，不把“公开可读”误写为任意再分发许可。
