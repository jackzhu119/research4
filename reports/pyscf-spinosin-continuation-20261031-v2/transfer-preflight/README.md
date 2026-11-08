# 几何来源转移工具预检：实际转移尚未执行

新工具`scripts/prepare_pyscf_geometry_transfer_20261031.py`只准备几何和来源，不计算积分，不启动worker，不读取检查点。12项纯防护测试通过；测试中的DONE／PARTIAL、能量和梯度是明确标注的临时合成夹具，不能作为科研结果。

工具等待root在实际源阶段结束后提供`allow_geometry_transfer:true`、带时区`created_at_utc`、最终`cpu_budget_sha256`和`source_result_sha256`放行文件。共享锁及全账本RUNNING检查防止读取活动阶段；只有被账本唯一记录、实际退出的源结果才能进入准备。

优化用途优先采用真正DONE且优化／新SCF／新梯度均通过的全精度几何。源为PARTIAL时只选择最后一个完整SCF／梯度步骤，核对坐标／梯度SHA及重算统计；不读取试探检查点，不沿用未完成几何的能量。新优化器必须重新SCF／梯度，不恢复geomeTRIC历史。

频率用途仅接受单独的V4非DF PBE／def2-SVP／grid1／无D3方法、原阈值与完成状态，并重新核对实际最终梯度。输出XYZ和完整精度NPY采用原字节复制，保持频率worker需要的优化结果SHA，避免改写标题或小数精度使方法一致性门槛失效。

75原子顺序、组成、全精度坐标和立体身份复用可信ChEBI图核验。输出包括`transfer_geometry.xyz`、`transfer_coordinates_angstrom.npy`、`transfer_geometry_provenance.json`、`transfer_manifest.json`及逐文件索引；源能量／梯度仅作来源配对，不赋给另一计算方法。

本预检没有实际读取当前优化阶段的最终几何，没有生成真实科研转移目录。root后续明确GO后才执行。
