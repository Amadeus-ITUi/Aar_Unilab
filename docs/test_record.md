# WE11 UniLab 测试与验收记录

负责人：汪成浩
归档日期：2026-08-04
上一归档基线：`5a4c7a200f7264c88c9e217cacc5c74907187988`

## 1. 本次归档变更

本次新增 Flat/Rough 最终 checkpoint、Conda 一键安装脚本和模型 manifest；将
`pyproject.toml` 切换为适配 `pip install -e ".[dev]"` 的 setuptools 配置并删除
`uv.lock`。不修改训练任务、环境动力学、网络、Reward、资产或实验数据。

## 2. 最小验证

| 检查项 | 命令 | 预期结果 | 结果 |
|---|---|---|---|
| Conda 脚本语法 | `bash -n install_conda_environment.txt` | 无错误 | 通过 |
| Python 打包 | Conda 验证环境执行 `pip wheel --no-deps --no-build-isolation .` | wheel 成功 | 通过 |
| 模型 SHA-256 | `(cd models/we11 && sha256sum -c SHA256SUMS)` | 两个成功 | 通过 |
| Checkpoint 加载 | CPU 加载 Flat/Rough 并核对 checkpoint 字典 | 均成功 | 通过 |
| native PD 构建/导入 | 本机 GCC/G++ 编译并 Torch-first 导入 | 两项可用 | 通过 |
| Flat 最小 PPO | CPU、8 env、8 step、1 iteration | 64 steps | 通过 |
| Rough 最小 PPO | CPU、8 env、8 step、1 iteration | 64 steps | 通过 |
| Ruff | `python -m ruff format --check .`、`python -m ruff check .` | 全部通过 | 通过 |
| MyPy | `python -m mypy src/unilab` | 无问题 | 通过 |
| Git whitespace | `git diff --check` | 无错误 | 通过 |
| 全部非 slow 测试 | 见下方命令 | 全部通过 | 153 passed |

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
MPLBACKEND=Agg \
MPLCONFIGDIR=/tmp/matplotlib-we11-final-test \
python -m pytest -m "not slow"
```

未构建本机 native 扩展时，测试会出现 5 条 fallback warning；Conda 一键脚本会在
正式测试前本机构建扩展，并通过 Torch-first 检查阻止 fallback 环境被误判为成功。

## 3. 接收方验收

| 检查项 | 状态 | 备注 |
|---|---|---|
| 代码可拉取 | 待接收人确认 | 克隆正式分支 |
| 环境可安装 | 待接收人确认 | 目标机器执行安装流程 |
| 最小测试可运行 | 待接收人确认 | 执行 focused tests |
| Flat 仿真可启动 | 待接收人确认 | 至少完成短步数 smoke |
| Rough 仿真可启动 | 待接收人确认 | 至少完成短步数 smoke |
| 日志可输出 | 待接收人确认 | TensorBoard 可读取 event |
| 模型导出/parity | 待验收 | 与 Play/Deploy 合同逐项核对 |
| 真机低风险测试 | 不属于本仓库 | 由 Deploy 流程完成 |

## 4. 异常判定

出现观测/动作维度不匹配、NaN、模型加载 shape mismatch、关节方向异常、action 持续 clip、轮速/力矩持续饱和或 delay 语义不一致时，停止评估，不得将模型交给真机链路。
