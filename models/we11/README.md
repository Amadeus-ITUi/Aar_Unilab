# WE11 正式模型

本目录只保留与当前 145D 观测合同一致的 WE11 Getup 正式模型。Flat 和 Rough
环境仍然可训练，但旧的 135D checkpoint、ONNX 和兼容入口已经移除。

## Getup

```text
checkpoint: getup/model_9999.pt
onnx: getup/policy.onnx
observation: we11_v2_145
action: 6D
```

同时保留训练配置、运行摘要和导出 manifest。验证文件完整性：

```bash
(cd models/we11 && sha256sum --check SHA256SUMS)
```

新训练的 Flat/Rough/Getup 模型均必须使用当前 `we11_v2_145` 合同，不再接受
缺少翼关节速度观测的旧模型。
