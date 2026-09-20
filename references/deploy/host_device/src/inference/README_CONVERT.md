# ONNX 转 MNN 模型转换指南

本目录提供了将 ONNX 模型转换为 MNN 格式的工具脚本。

## 快速开始

### 方法 1: 使用简化版脚本（推荐）

```bash
# 安装依赖
pip install MNN onnx

# 转换单个模型
python3 convert_onnx_to_mnn_simple.py models/policy.onnx

# 指定输出路径
python3 convert_onnx_to_mnn_simple.py models/policy.onnx models/policy.mnn
```

### 方法 2: 使用完整版脚本

完整版脚本会自动尝试多种转换方法：

```bash
python3 convert_onnx_to_mnn.py models/policy.onnx models/policy.mnn
```

## 批量转换

```bash
# 转换所有 ONNX 模型
cd src/inference/models
for f in *.onnx; do
    python3 ../convert_onnx_to_mnn_simple.py "$f"
done
```

## 转换方法说明

### 方法 1: Python MNN 包（最简单）

**优点:**
- 安装简单: `pip install MNN`
- 使用方便，无需编译

**缺点:**
- 需要安装 Python 包
- 可能不支持所有 ONNX 操作符

### 方法 2: MNNConverter 二进制文件

**优点:**
- 性能更好
- 支持更多操作符

**缺点:**
- 需要从源码编译

**编译步骤:**

```bash
cd src/inference/thirdparty/MNN-master
mkdir -p build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON
make -j4

# 转换器位置: build/MNNConverter 或 build/converter/MNNConverter
```

## 验证转换结果

转换完成后，可以检查：

1. **文件是否存在:**
   ```bash
   ls -lh models/*.mnn
   ```

2. **文件大小对比:**
   ```bash
   ls -lh models/policy.onnx models/policy.mnn
   ```

3. **在代码中测试加载:**
   运行 lab_inference_node，检查是否能正常加载 MNN 模型。

## 常见问题

### Q: 转换失败，提示找不到 MNN 包

**A:** 安装 MNN Python 包:
```bash
pip install MNN
```

### Q: 转换失败，提示某些操作符不支持

**A:** MNN 可能不支持某些 ONNX 操作符。可以：
1. 检查 MNN 文档，查看支持的操作符列表
2. 尝试简化模型结构
3. 使用其他转换工具（如 ONNX Runtime）

### Q: 转换后的模型无法加载

**A:** 检查：
1. 模型文件路径是否正确
2. 模型文件是否完整（文件大小是否正常）
3. 查看 lab_inference_node 的日志输出

### Q: 如何验证转换后的模型是否正确

**A:** 可以：
1. 使用相同的输入数据，对比 ONNX 和 MNN 模型的输出
2. 在 lab_inference_node 中实际运行，观察行为是否正常

## 模型文件位置

- **ONNX 模型:** `src/inference/models/*.onnx`
- **MNN 模型:** `src/inference/models/*.mnn`

## 相关文件

- `convert_onnx_to_mnn.py` - 完整版转换脚本（支持多种方法）
- `convert_onnx_to_mnn_simple.py` - 简化版转换脚本（使用 Python MNN 包）
- `src/inference/src/lab_inference_node.cpp` - Lab 推理节点代码（使用 MNN 加载模型）

## 参考链接

- [MNN 官方文档](https://www.yuque.com/mnn)
- [MNN GitHub](https://github.com/alibaba/MNN)
- [ONNX 官方文档](https://onnx.ai/)
