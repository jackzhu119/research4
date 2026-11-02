# 纯PBE的DF Hessian数据真正读用审计

结论：**存在可删除的未使用存储，但仅跳过它们仍不能解除当前磁盘阻塞。** 本次只读审查PySCF2.14.0的`with_k=False, auxbasis_response=2`控制流，未修改库／worker，未调用积分、SCF、梯度、响应或Hessian。另一只读代理独立交叉核对了下面的行号和分支。

下表行号属于本目录原样快照`df/hessian/rhf.py`；详细writer／reader／guard保存在JSON。

| 对象 | writer与reader | pure J判定 |
|---|---|---|
| partial `rho_ip1` | 137创建、149完整写；aux guard242下243读、244缩并；261/263/265进入J的`_ej`，267—268进入`ej` | **真实J／辅助基响应数据，不能直接删除** |
| partial `wk_ip2_Ipk`，磁盘名`wk_ip2` | 172创建；181写受`with_k`180限制；254读受`with_k`250限制 | K-only；pure J没有数值写入／读取，声明的大尺寸不是已占用数据 |
| partial `wk_ip2_P__` | 173的aux>1下174分配；183写、357—358读均还受`with_k`限制 | pure J有未使用的1,510,809,600字节ndarray地址预留；不是磁盘 |
| partial `rhok_ip1_IkP/PkI` | 138—139创建空group；152—153写及167/206读均受`with_k`限制 | pure J仅空组元数据，不是巨量payload；255只是注释 |
| H1 `rho0_Pij` | 406创建、418逐AO行写；435读后438生成J缓冲，aux下462读后464/466进入`vj1` | **pure J真实读取，不能直接删除** |
| H1 `wj_ip1_pij` | 407创建、aux下423写；aux下463读，467进入`vj1` | **pure J真实读取，不能直接删除** |
| H1 `vj1_buf`内存数组 | 430分配、438累加；450读取RAM，474对称化，478传给RKS | **J数值必需，保留** |
| H1 `ftmp['vj1_buf']`磁盘副本 | 443写出，函数内没有读取该磁盘副本 | **真正未使用的约0.973 GiB磁盘写，可去掉副本而保留RAM计算** |
| H1 `vk1_buf` | 429分配；441累加和455读取均受`with_k`限制 | pure J未使用的约13.94 MB内存工作区，不解除磁盘瓶颈 |

特别注意443的Python语义：

```python
vj1_buf = ftmp['vj1_buf'] = vj1_buf
```

局部`vj1_buf`仍是原ndarray，**并没有变成HDF5 Dataset**。因此450读取RAM并不能证明磁盘副本被使用。审查附有小型纯Python链式赋值负控；它没有运行任何量子方法。

RKS入口也已追到：`rks.py:57—60`以`with_k=hybrid=False`取得并加入`ej`；105—107把`vj1`加入H1，只有108—109的`vk1`受hybrid guard限制。因此不能把名字含`wk`或`ip`的J辅助响应误当作可丢弃交换项。

## 能节省多少，仍缺多少

当前默认H1阶段写入46,734,874,272字节，删除443未使用副本可省1,045,159,200字节，剩下真正被读取的`rho0_Pij + wj_ip1_pij`仍为**45,689,715,072字节（42.5519 GiB）**。partial阶段真正被读取的`rho_ip1`仍有34,267,286,304字节（约31.91 GiB）。两个不同生命周期不能相加冒称单一峰值，但任一都超出现环境磁盘。

K-only的未写HDF5 dataset／空group只有元数据，不能宣称删除它们回收31.9 GiB。已安装h5py的小型metadata-only对照也确认默认未写contiguous dataset的数值storage为0；没有创建巨大数据文件。跳过K-only `np.empty`可避免无用地址预留，但不是同量RSS或磁盘实测。

这描述**现有代码**的读写，不是所有数学等价DF实现的理论最低存储。可严格观察到：149的转置经过243反向排列恢复，244的`rhoj1_P`缩并与147已计算的`rhoj1[i0]`表达式、操作数对应相同；原值在158被释放。保留已有预收缩也许能形成另一种存储调度，但那需要替换真实的数据流并独立数值验证，不能以“未使用”为由直接删`rho_ip1`。本审查没有开发或验证任何新导数公式／存储实现，H1的42.55 GiB读取问题也未解决。

## 精确源码

- `pyscf/df/hessian/rhf.py` SHA-256：`e62940949da49ae7879b5a9f5e20b8000ee086f7a41241bfa36c8aed0542c380`。
- `pyscf/df/hessian/rks.py` SHA-256：`6aebab36cdb4b88ce6741c3eea9681e4aaa38b5b2e782b7c1ac43684690c1da1`。
- `pure_j_dataset_read_use.json`包含逐项guard、J下游收缩、精确尺寸与小型控制结果；两个`source snapshot`为实际安装源码字节副本，未改旧封存索引。
