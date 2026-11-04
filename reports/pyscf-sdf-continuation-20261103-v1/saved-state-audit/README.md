# 旧SCF保存状态组件核验

观察时间（用户时区）：2026-11-03T23:15:39.538904-08:00。状态：**SCF_COMPONENT_VERIFIED**。

本次按照root明确授权，只读打开已不存在原进程的V4检查点，以h5py读取mol纯JSON及scf/e_tot、mo_coeff、mo_energy、mo_occ。没有调用gto.loads或解释atom/basis的repr字符串，没有SCF、电子积分、梯度或Hessian新求值。

核验覆盖：762×762系数和762个轨道能／占据数的完整float64 payload SHA、75×3 Bohr坐标、分子JSON规范化身份、保存总能与旧binding及progress、C28H32O15/75原子/320电子、160个双占据轨道及762球谐AO。只在序列化副本中按旧V8规则规范化_env[17]非物理记账标志，不修改任何原数据。HDF mol JSON没有显式spin字段；旧记录指定spin=0，限制性0/2占据与该输入一致，未编造JSON自带的spin或独立自旋计算。

另只读加载旧gradient、Bohr/Angstrom normal_mode_geometry NPY（allow_pickle=False），实际均75×3即225个元素；完整SHA、坐标及梯度统计与原记录比对。旧input.xyz还匹配量子计算前锁定的字节SHA；原子顺序与检查点坐标保持一致，未载入新SDF替换它。normal_mode_geometry只是几何，不是已得到的正常模式。

**SCF_COMPONENT_VERIFIED不等于整个旧阶段DONE。** 它证明保存的数据组件与原已记录SCF状态一致，不是新做SCF收敛或电子能／残差复算。网格实际数组未存于检查点，未重建或验证网格payload；未计算重叠积分验证MO正交性。旧最终wait4、退出码、停止时刻仍未知，旧ledger保持原样。完整RHF／PBE Hessian及振型／频率仍不可恢复，IR强度NOT_COMPUTED。

如组件核验通过，此checkpoint可作为本轮fresh SCF的初猜；仍要求新fresh SCF／梯度通过并创建健康新snapshot，不能为了省几分钟而放宽clean-parent恢复门槛。原V3已闭合native优化checkpoint（AA哈希）继续作为备用。所有原始文件在读取前后的字节SHA必须相同；损坏时只标UNVERIFIED，不修原件。
