"""Render user-confirmed acquisition methods beside frozen dataset tables."""

import csv
import io
import json
from pathlib import Path


DECLARATION = "data/annotations/acquisition_conditions_user_declaration_20261112.json"
CONTENT_PATHS = {
    "methods/MATERIALS_AND_METHODS_ZH.md",
    "methods/equipment.csv",
    "methods/pending_parameters.csv",
    "methods/acquisition_conditions.json",
}


def csv_data(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def table(headers, values):
    def cell(value):
        return str(value).replace("|", "／").replace("\n", " ")
    return "\n".join([
        "|" + "|".join(headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
        *("|" + "|".join(cell(v) for v in row) + "|" for row in values),
    ])


def generate(root: Path, tables):
    source = root / DECLARATION
    declaration = json.loads(source.read_text())
    if set(declaration["applies_to_species"]) != {"枣仁", "酸枣仁", "黄芪", "天麻"}:
        raise ValueError("Acquisition declaration does not cover the four current kinds")
    if declaration["reference"]["reference_response_correction_applied"] is not True:
        raise ValueError("Current equations require the confirmed PE reference correction")
    equipment = declaration["equipment"]
    species = rows(tables["methods/dataset_by_species.csv"])
    workbooks = rows(tables["methods/dataset_by_workbook.csv"])
    tasks = rows(tables["methods/task_role_counts.csv"])
    # Table modules expose their stable English keys; keep presentation separate.
    def field(row, *names):
        for name in names:
            if name in row:
                return row[name]
        raise KeyError(names)
    species_table = table(
        ["药材种类", "工作簿数", "样品记录总数", "TRAIN", "CAL", "PRED"],
        [[field(r, "species"), field(r, "workbooks", "workbook_count"),
          field(r, "records", "total", "record_count"), field(r, "train"),
          field(r, "calibration"), field(r, "prediction")] for r in species]
        + [["合计", 18, 1672, 999, 334, 339]],
    )
    workbook_table = table(
        ["来源组", "种类", "用户确认产地", "有效工作簿", "总数", "TRAIN", "CAL", "PRED"],
        [[field(r, "origin_group_id"), field(r, "species"), field(r, "origin"),
          field(r, "filename", "workbook", "source_workbook"),
          field(r, "records", "total", "record_count"), field(r, "train"),
          field(r, "calibration"), field(r, "prediction")] for r in workbooks],
    )
    task_table = table(
        ["任务", "类别数", "TRAIN", "CAL", "PRED", "任务总数"],
        [[field(r, "task_name", "task", "task_id"), field(r, "n_classes", "class_count"),
          field(r, "train"), field(r, "calibration"), field(r, "prediction"),
          field(r, "records", "total", "record_count")] for r in tasks],
    )
    equipment_table = table(["主要实验仪器名称", "型号", "设备厂商"],
                            [[r["name"], r["model"], r["manufacturer"]] for r in equipment])
    pending_table = table(["需补充项目", "当前记录"],
                          [[r["parameter"], r["status"]] for r in declaration["pending_parameters"]])
    text = r'''# 材料、仪器、光谱计算与数据划分（论文写作稿）

本稿汇总四种药材的实验条件、光谱计算公式及现有数据划分，仪器与采购信息依据2026-11-12实验者确认记录，样品数量和角色依据18份有效Excel及冻结清单。待补实验参数集中列于第7节。

## 1 材料来源与样品制备

本研究所用枣仁、酸枣仁、黄芪和天麻商品均通过网络渠道购买，经样品制备后进行太赫兹时域光谱测量。以实验者确认的商品种类和标称产地建立来源组，各来源名称、样品数量与对应工作簿见表3。

四种药材采用相同的仪器、制备流程和测量条件。制备用主要设备包括行星球磨机、恒温干燥箱、电子天平和手动压片机，设备信息列于表1。球磨、干燥、配比和压片的具体操作参数待实验记录补齐。

**表1 主要实验仪器及设备厂商**

{{EQUIPMENT_TABLE}}

## 2 太赫兹时域光谱采集

实验在国家农业信息化工程技术中心太赫兹实验室进行，采用德国Menlo Systems公司的TERA K15太赫兹时域光谱仪（THz-TDS），采集四种药材样品的时域光谱。发生／探测光路的光学中心波长为1550 nm，光功率为33 mW。

测量在室温292 K下进行。向仪器封闭罩中充入氮气，将罩内湿度控制在4%以下，以降低空气中水分对太赫兹测量的影响。采用不含药材样品的聚乙烯空白压片扫描获得参考时域信号，再将样品置于样品架上扫描，获得携带样品光学信息的时域信号。聚乙烯在所研究太赫兹频段具有较低吸收，适合作为空白参考材料。

在导出光学参数前，对聚乙烯空白参考的响应进行校正。参考校正已由实验者确认，具体软件、版本和设置仍待归档；本次方法整理沿用既有工作簿中的光学参数。

现有时域页的原始列头包含标称厚度：1670条为1.5 mm，云南枣仁ZY02-001和ZY02-002两条为1 mm。上述数值属于原文件列头的标称记录。逐条记录见[样品角色与表头明细](record_role_assignment.csv)，逐表分布见[厚度表](nominal_thickness_by_workbook.csv)。参考片厚度、实测厚度与不确定度仍需原始实验记录补充。

## 3 复透射函数、折射率与吸收系数

对样品和参考时域信号作快速傅里叶变换（FFT），得到频域信号。以下沿用用户所给公式记号，以\(T(\omega)\)表示**经聚乙烯参考响应校正后的等效空气参考复透射函数**，式中的比值为相应等效信号之比；未经校正的原始样品／聚乙烯比值另记为\(T_{\mathrm{raw}}\)，不能与\(T\)直接混同。采用均匀平行平板、近法向入射并忽略或另行处理多次内部反射的模型，可写为：

$$
T(\omega)=\frac{E_{\mathrm{sam}}(\omega)}{E_{\mathrm{ref}}(\omega)}
=\frac{4n(\omega)}{[n(\omega)+1]^2}
\exp\left[-\frac{\alpha(\omega)d}{2}\right]
\exp\left[-i\frac{[n(\omega)-1]\omega d}{c}\right]. \tag{1}
$$

定义展开后的相位延迟为
\(\Phi(\omega)=\operatorname{unwrap}[\phi_{\mathrm{ref}}(\omega)-\phi_{\mathrm{sam}}(\omega)]\)，与式（1）的负指数保持一致，则

$$
n(\omega)=1+\frac{c\Phi(\omega)}{\omega d}. \tag{2}
$$

由幅值信息计算吸收系数：

$$
\alpha(\omega)=-\frac{2}{d}\ln\left\{|T(\omega)|\,
\frac{[n(\omega)+1]^2}{4n(\omega)}\right\}. \tag{3}
$$

其中，\(d\)为对应样品厚度，\(c\)为真空光速，\(\omega=2\pi f\)为角频率，\(n\)无量纲。若采用SI单位，频率从THz换为Hz，厚度从mm换为m，\(c=299\,792\,458\ \mathrm{m\,s^{-1}}\)，式（3）的吸收系数单位为\(\mathrm{m^{-1}}\)。若使用其他长度单位，须同步换算。当前工作簿的吸收系数单位和时域坐标单位未完整标注，不能据此擅写其导出单位为\(\mathrm{cm^{-1}}\)或ps；公式单位说明也不表示本次对数据作了单位变换。

聚乙烯参考片厚度、响应参数、校正软件以及相位、时窗、回波和FFT处理设置需与原始实验记录一并归档。公式来源的完整文献条目亦列入待补清单。

## 4 有效数据量与各表样品数

当前有效数据版本为`2026-11-08-confirmed-origin-kind-v1`，共18份工作簿、1672条样品谱记录。按实验者此前的独立实物声明，每条登记记录对应一份实物样品；原855条与其余817条分别有独立性声明。其中817条由黄芪403条、天麻405条和新增枣仁9条组成，实际批次编号仍待补充。

每个工作簿的时域、频域、折射率、吸收系数四页描述同一组样品，参考谱列也不作为额外样品。当前原表每本时域页有6750行数据、频域页1001行、折射率及吸收系数页各2578行（均不含表头）；这些是采样／导出点的行数，不是样品数，也不据此推断不同页使用同一等长FFT网格。四页不重复计数为四倍样本。统计按各页真实样品列与冻结记录逐条对应，原文件格式检查记录另存于统计核验附件。

**表2 四种药材的样品总数与固定角色**

{{SPECIES_TABLE}}

**表3 18份有效工作簿的具体划分**

{{WORKBOOK_TABLE}}

`天麻安徽毫州.xlsx`按原上传拼写保留；山西与山西灵石表示两个既有来源组，不据名称认定其行政范围互斥。每份工作簿也不自动等同于一个已独立验证的采购批次。

## 5 训练集、校准集和预测集

主划分锁定为TRAIN 999、CAL 334和PRED 339，约为6∶2∶2，逐样品角色以[record_role_assignment.csv](record_role_assignment.csv)和原始[冻结划分](../inputs/split_manifest.csv)为准，不按表中列序截取“前60%”。原855条仅沿用既有逐ID角色，种类、产地均用修正后的标签；其余817条按来源组、`split_seed=20261101`和冻结的确定性规则分配。各任务采用修正后的种类与产地标签。

对其余样品的分配规则为：组内按record_id排序，以SHA-256(seed:group)的前8字节形成该组NumPy PCG64生成器种子，在不使用光谱或模型得分的条件下生成固定分配；TRAIN为max(1,floor(0.60N))，CAL为max(1,floor(0.20N))，其余为PRED并至少留1条。三样品小组为1/1/1；原ZY02五样品组已固定为3/1/1。非整齐组按实际角色统计，如103条组为61/20/22、102条组为61/20/21，不为了凑整比例重新移动样品。

TRAIN用于预处理、变量选择器和分类器的拟合，以及需要选参时的训练内部评价。CAL用于源侧方案锁定后的校准拟合。固定六任务的已保存方法包括两类anchor校准和`train_plus_calibration_full_retrain`对照；后者沿用锁定的源配置，在TRAIN与CAL合并后重新拟合预处理、变量选择器及分类器。CAL参与这些预设步骤的拟合，与使用CAL选择最终方法或超参数有所区别。

PRED仅用于锁定方案的评分，不参与特征、频段、超参数、种子、校准强度或模型选择。现有样品有历史开发使用，固定角色隔离的是当前流程；这仍是回顾性内部验证，不称为未曝光的外部盲测或真实跨批次评价。

**表4 六项修正任务的角色分母**

{{TASK_TABLE}}

各任务会复用同一批记录，不能把这六行相加当成总样品量。纯酸枣仁只使用ZY03—ZY07；纯枣仁来源使用ZY01、ZY02、ZR01—ZR03。枣仁新增三来源各仅3份，每个主角色只有1份，相关类别结果须同时报告小样本局限。

## 6 固定留出、交叉验证和内部Support／Query的区别

- **六任务固定留出：** 使用表4的角色。二分类与纯酸枣仁来源主任务在TRAIN内作2折源侧选择；四种类、黄芪、天麻及纯枣仁来源任务使用预设固定参数，不新增调参。模型种子为42。不能把所有六任务都描述成同一套嵌套5折实验。
- **V4纯酸枣仁主比较：** 在纯酸枣仁的300条主TRAIN内，按种子20261031、20261108、42进行3次分区重复，每次外5折、源侧内2折；每外折Source240、Query60，5来源各48/12。这里没有消耗主CAL100或主PRED100。612个主频点取自既有0.1—1.5 THz坐标，不把模型分析频段写成仪器硬件带宽；迁移另有128点表示。原纠正1305矩阵还含其他目标配置，不能把本段的300条与折数统一套用于所有配置，各配置范围见协议目录。
- **B第1—5轮：** Source、Support候选池、Query均从主TRAIN内部构建，在各个任务内互斥。E01二分类常用240/20/40，Source的枣仁/酸枣仁比例可为180/60、120/120或60/180；Support每类10、Query每类20。E03五产地常用190/50/60，每类38/10/12。这些是内部任务角色，不是原334条CAL或339条PRED。
- **Support预算：** 常规K=1/3/5/10指每类支持量，二类总2/6/10/20、五类总5/15/25/50；多次抽样不增加独立样本。特定组成／获取实验按总预算设计，不能一律声称每类都有K条。B2 F04的240端点已包含Source190＋Support50，不能再加一次Support。
- **B5 I01内层选择：** 仅在外层Source内形成内层Source／Support／validation：二类180/20/40，五类100/50/40；不以外层Support或Query选择方案。
- **元学习框架：** 45上下文为3表征×3种子×5折，重复使用同一300条酸枣仁主TRAIN。外层190/50/60与B任务对应；每个训练episode仅从Source190抽取support25＋query25，而非访问外层Support50／Query60。未完成的最终元学习结果不列为论文已完成成绩。

已保存正确实验的正负结果全部保留。真实独立新批次E05仍为`NOT_RUN_NEEDS_NEW_BATCH_DATA`，仪器条件一致或重复抽样均不能补造新批次数。

## 7 待补充的实验记录

{{PENDING_TABLE}}

当前数据量和冻结角色已可据原表核准；其余实验参数在投稿前以原始实验记录补齐。

## 8 可直接使用与复核的附件

- [Excel汇总与逐样品分配](data_splits.xlsx)：按药材、工作簿和角色筛选，可用于后续论文制表。
- [逐药材统计](dataset_by_species.csv)、[逐工作簿统计](dataset_by_workbook.csv)、[六任务统计](task_role_counts.csv)。
- [逐样品记录](record_role_assignment.csv)：包含原ID、来源、原工作簿和列头、角色及表头标称厚度，核验时不重排样品。
- [仪器表](equipment.csv)、[用户本轮声明](../../../data/annotations/acquisition_conditions_user_declaration_20261112.json)、[统计与来源核对](dataset_split_audit.json)、[待补参数](pending_parameters.csv)。

本文件及表格由显式的当前数据入口生成，纳入论文白名单及文件哈希校验；不检索已撤回的旧分类输出，不改动冻结角色和原始光谱。
'''
    for marker, value in {
        "{{EQUIPMENT_TABLE}}": equipment_table,
        "{{SPECIES_TABLE}}": species_table,
        "{{WORKBOOK_TABLE}}": workbook_table,
        "{{TASK_TABLE}}": task_table,
        "{{PENDING_TABLE}}": pending_table,
    }.items():
        text = text.replace(marker, value)
    text = text.replace(r"\(", "$" ).replace(r"\)", "$")
    return {
        "methods/MATERIALS_AND_METHODS_ZH.md": text.encode(),
        "methods/equipment.csv": csv_data(equipment),
        "methods/pending_parameters.csv": csv_data(declaration["pending_parameters"]),
        "methods/acquisition_conditions.json": source.read_bytes(),
    }
