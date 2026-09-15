# ONNX Runtime 和 CNPY 库安装指南

## 快速安装

运行提供的脚本即可自动下载和安装：

```bash
cd /home/mytheo/project/deploy_cpp/src/inference
bash download_onnxruntime.sh
```

## 手动安装步骤

### 1. 安装 ONNX Runtime

#### x86_64 架构
```bash
cd /home/mytheo/project/deploy_cpp/src/inference/thirdparty
wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz
tar -xzf onnxruntime-linux-x64-1.21.0.tgz
mv onnxruntime-linux-x64-1.21.0 onnxruntime-linux-x64-1.21.0
```

#### aarch64 架构
```bash
cd /home/mytheo/project/deploy_cpp/src/inference/thirdparty
wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-aarch64-1.21.0.tgz
tar -xzf onnxruntime-linux-aarch64-1.21.0.tgz
mv onnxruntime-linux-aarch64-1.21.0 onnxruntime-linux-aarch64-1.21.0
```

### 2. 安装 CNPY 库

CNPY 需要从源码编译：

```bash
cd /home/mytheo/project/deploy_cpp/src/inference/thirdparty

# 克隆 CNPY 仓库
git clone https://github.com/rogersce/cnpy.git cnpy-source

# 编译 CNPY
cd cnpy-source
mkdir build && cd build
cmake ..
make -j$(nproc)

# 复制库文件到正确位置
cd ../..
mkdir -p cnpy-linux-x64/lib cnpy-linux-x64/include
cp cnpy-source/build/libcnpy.so cnpy-linux-x64/lib/
cp cnpy-source/cnpy.h cnpy-linux-x64/include/
```

**注意**：如果是 aarch64 架构，将 `cnpy-linux-x64` 替换为 `cnpy-linux-aarch64`。

### 3. 验证安装

检查库文件是否存在：

```bash
# x86_64 架构
ls /home/mytheo/project/deploy_cpp/src/inference/thirdparty/onnxruntime-linux-x64-1.21.0/lib/libonnxruntime.so
ls /home/mytheo/project/deploy_cpp/src/inference/thirdparty/cnpy-linux-x64/lib/libcnpy.so

# aarch64 架构
ls /home/mytheo/project/deploy_cpp/src/inference/thirdparty/onnxruntime-linux-aarch64-1.21.0/lib/libonnxruntime.so
ls /home/mytheo/project/deploy_cpp/src/inference/thirdparty/cnpy-linux-aarch64/lib/libcnpy.so
```

## 重新编译 inference 包

安装完库文件后，重新编译 inference 包：

```bash
cd /home/mytheo/project/deploy_cpp
rm -rf build/inference install/inference
colcon build --paths src/inference --symlink-install
source install/setup.sh
```

## 验证编译结果

检查 lab_inference_node 是否已编译：

```bash
ros2 pkg executables inference
# 应该看到：
# inference lab_inference_node
# inference sweep_frequency_node

# 检查可执行文件
ls install/inference/lib/inference/
# 应该看到：
# lab_inference_node
# sweep_frequency_node
```

## 目录结构

安装完成后的目录结构应该是：

```
src/inference/thirdparty/
├── onnxruntime-linux-x64-1.21.0/    # x86_64 架构
│   ├── lib/
│   │   ├── libonnxruntime.so -> libonnxruntime.so.1
│   │   ├── libonnxruntime.so.1 -> libonnxruntime.so.1.21.0
│   │   └── libonnxruntime.so.1.21.0
│   └── include/
├── cnpy-linux-x64/                   # x86_64 架构
│   ├── lib/
│   │   └── libcnpy.so
│   └── include/
│       └── cnpy.h
└── ...
```

## 常见问题

### 问题 1: 下载失败（网络问题）

如果直接下载失败，可以：
1. 使用代理
2. 从其他镜像源下载
3. 手动下载并放到正确位置

### 问题 2: CNPY 编译失败

确保已安装必要的依赖：
```bash
sudo apt update
sudo apt install cmake build-essential zlib1g-dev
```

### 问题 3: 找不到库文件

检查 CMakeLists.txt 中的路径是否正确：
- 使用 `${CMAKE_CURRENT_SOURCE_DIR}` 而不是 `${CMAKE_SOURCE_DIR}`
- 确保路径与你的系统架构匹配（x86_64 或 aarch64）

## 版本信息

- **ONNX Runtime**: 1.21.0
- **CNPY**: 最新版本（从 GitHub 仓库克隆）

## 参考链接

- ONNX Runtime: https://github.com/microsoft/onnxruntime
- CNPY: https://github.com/rogersce/cnpy
