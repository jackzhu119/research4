# 最终检查点的只读恢复与续算准备：已实际完成

本阶段只读取已退出的`spinosin_cached_optimization`阶段，提取并核验几何、保存坐标及生成命令；**没有实际运行续算worker，没有新增SCF、梯度、优化或频率。**

[continuation_manifest.json](../continuation/continuation_manifest.json)状态为READY。原量子CPU账本剩余633.765852秒；请求900秒被限制为631秒，wall上限947秒、单线程、原12 GiB内存上限、优化最多35步。631秒不保证完成35步或收敛，命令不会重置或扩大原账本。

检查点SHA为`8b450cb82ee53b9080c1b8a3d15d0f9553814ca96c75e6211697cc436467215f`。实际从PySCF Mole读取完整精度75原子坐标，核验C28H32O15、原子顺序、320电子、中性单重态、零自由基图及3D立体InChIKey。共价键长为0.97347—1.54910 Å。已保存Angstrom／Bohr两份.npy和新XYZ，后者SHA为`92a7228d3e0977793e2cc180a381607ba761ed103f8e6b2d1c8ddff22e6a3074`。

**恢复的检查点不是最后一个完成梯度的几何。** 相对完成的cycle4，其最大原始坐标分量差0.26970997 Å；正确旋转对齐后全原子映射RMSD为0.14165277 Å，重原子RMSD为0.12934631 Å。后两者排除了仅有整体平移／旋转的解释。cycle4的能量−2210.43596049 Hartree和梯度仅属于该步，不能赋给恢复XYZ；检查点可能是下一试探几何，不能冒称驻点。

可核验准备结果：

```bash
.runtime/bin/python scripts/prepare_pyscf_pilot_resume_20261031.py verify \
  --output reports/pyscf-spinosin-pilot-20261031/continuation
```

已准备、**尚未执行**的有界续算入口：

```bash
bash reports/pyscf-spinosin-pilot-20261031/continuation/resume_same_budget.sh
```

它首先核对全部冻结输入及原账本SHA，再调用原预算控制器和cached-DF worker，使用新输出目录`calculations/spinosin-optimize-resume-001`。这是几何＋SCF检查点猜测续算；它新建geomeTRIC优化器，不恢复旧优化器的Hessian近似、trust radius或迭代历史，必须进行新的SCF和梯度验证。

真实prepare／verify命令起止和退出码见[commands.jsonl](commands.jsonl)，输出见[prepare.log](prepare.log)及[verify.log](verify.log)。[坐标差异数据](checkpoint_vs_completed_geometry.json)由保存的完整精度坐标及最后完成步骤直接计算，不包含新的量子化学计算。
