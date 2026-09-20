# WE11 Conda 环境验证记录

负责人：汪成浩
验证日期：2026-08-04
目标分支：`Walking_Eagle-Unilab_final`

## 验证环境

- OS/架构：Linux x86_64
- Conda 环境：`we11_conda_test_20260801`
- Python：3.13.14
- PyTorch：2.7.0
- 安装合同：Conda + pip editable；不依赖仓库内虚拟环境或 `uv.lock`

## 已完成检查

1. `bash -n install_conda_environment.txt`：通过。
2. setuptools wheel 构建：通过，生成 `unilab-0.1.0-py3-none-any.whl`。
3. Flat/Rough `model_1500.pt` SHA-256 校验：通过。
4. Flat/Rough checkpoint 在 CPU 上反序列化：通过；均包含 actor、critic、optimizer
   和 UniLab environment/logger state。
5. 全部非 slow 测试：`153 passed`。
6. native command-delay PD：已在该 Conda 环境本地编译；Torch-first 导入后
   `native_mixed_pd=true`、`native_command_delay_pd=true`。
7. Flat/Rough 最小 PPO：各完成 1 iteration、64 steps；安装脚本会重复执行同一
   冒烟合同，任何一个失败都会非零退出。
8. Ruff：117 个 Python 文件格式检查通过，lint 无问题。
9. MyPy：75 个源码文件无问题。

## 复验命令

```bash
bash install_conda_environment.txt unilab_cuda cu128
conda activate unilab_cuda
python -m pytest -m "not slow"
(cd models/we11 && sha256sum -c SHA256SUMS)
```

此记录证明归档源码在独立 Conda 环境中的安装、打包、测试和模型读取合同；GPU、
驱动、CAN 与真机安全链路仍需在接收机器上按部署验收流程复验。
