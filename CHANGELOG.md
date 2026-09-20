# CHANGELOG

## Unreleased

### 删除

- 删除过时的 WE11 Flat/Rough 135D checkpoint、ONNX、示例 release 和兼容入口。
- Flat、Rough 和 Getup 训练环境统一只接受当前 145D 翼速度观测合同。

## v1.1.0 - 2026-08-04

### 新增

- 提交 Flat `2026-07-30_21-41-17/model_1500.pt`。
- 提交 Rough `2026-07-31_13-01-10/model_1500.pt`。
- 增加 `models/we11/README.md` 和 `SHA256SUMS`。
- 增加根目录 Conda/pip 一键安装脚本和完整安装说明。

### 修改

- 环境入口统一为 Conda + pip，README 和交接命令不再使用 `uv run`。
- `pyproject.toml` 改用 setuptools，并提供可由 pip 安装的 `dev` extra。
- 将 PACE 必需的 `cmaes`、`matplotlib`、`scipy` 纳入项目依赖，避免新
  Conda 环境在拟合测试收集阶段缺包。
- Conda 归档使用 MyPy 作为可移植类型检查入口，不保留依赖固定虚拟环境路径的
  Pyright 配置。
- 移除 CSV 外力默认配置中的本机绝对路径；WE11 任务继续使用仓库内相对资产路径。

### 删除

- 删除 `uv.lock` 和 `pyproject.toml` 中的 UV 专用配置。

### 运行影响

- 不改变 WE11 训练环境、网络、观测、动作、Reward、动力学或控制语义。

## v1.0.0 - 2026-08-04

### 归档

- 负责人统一记录为汪成浩。
- 增加算法交接说明、接口说明和测试验收记录。
- 在 README 增加正式交付文档入口。
- 记录 GitLab 正式分支和功能冻结基线。

### 运行影响

- 无。本版本记录为文档归档提交，不修改训练代码、配置、模型、资产或数据。

### 已知问题

- checkpoint、日志、视频和本机构建的 native `.so` 不纳入 Git，需要在交付介质中单独归档并记录 SHA-256。
- 真机验收需在 `Walking_Eagle-Deploy_final` 链路中独立完成。
