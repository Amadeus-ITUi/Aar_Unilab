#!/usr/bin/env python3
"""
简化版 ONNX 转 MNN 转换脚本（使用 MNN Python API）

安装依赖:
    pip install MNN onnx

使用方法:
    python3 convert_onnx_to_mnn_simple.py <onnx_model_path> [output_mnn_path]
"""

import os
import sys
import argparse
from pathlib import Path

def convert_onnx_to_mnn(onnx_path, mnn_path=None):
    """使用 MNN Python API 将 ONNX 模型转换为 MNN 格式"""
    
    try:
        import MNN
        from MNN.tools import mnnconvert
    except ImportError:
        print("❌ 错误: 未安装 MNN Python 包")
        print("请运行: pip install MNN")
        return False
    
    # 检查输入文件
    onnx_path = Path(onnx_path).absolute()
    if not onnx_path.exists():
        print(f"❌ 错误: ONNX 文件不存在: {onnx_path}")
        return False
    
    # 生成输出路径
    if mnn_path is None:
        mnn_path = onnx_path.with_suffix(".mnn")
    else:
        mnn_path = Path(mnn_path).absolute()
    
    # 确保输出目录存在
    mnn_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ONNX 转 MNN 模型转换工具 (简化版)")
    print("=" * 60)
    print(f"输入文件: {onnx_path}")
    print(f"输出文件: {mnn_path}")
    print()
    
    try:
        # 保存原始命令行参数
        original_argv = sys.argv.copy()
        
        # 设置转换参数
        sys.argv = [
            "mnnconvert",
            "-f", "ONNX",
            "--modelFile", str(onnx_path),
            "--MNNModel", str(mnn_path)
        ]
        
        print("开始转换...")
        # 调用 MNN 转换函数
        mnnconvert.convert(sys.argv)
        
        # 恢复原始参数
        sys.argv = original_argv
        
        # 检查输出文件
        if mnn_path.exists():
            input_size = onnx_path.stat().st_size / (1024 * 1024)  # MB
            output_size = mnn_path.stat().st_size / (1024 * 1024)  # MB
            print()
            print("✅ 转换成功!")
            print(f"   输入文件大小: {input_size:.2f} MB")
            print(f"   输出文件大小: {output_size:.2f} MB")
            print(f"   压缩比: {input_size/output_size:.2f}x" if output_size > 0 else "")
            return True
        else:
            print("❌ 转换失败: 输出文件未生成")
            return False
            
    except Exception as e:
        print(f"❌ 转换过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="将 ONNX 模型转换为 MNN 格式（简化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动生成输出文件名
  python3 convert_onnx_to_mnn_simple.py models/policy.onnx
  
  # 指定输出文件名
  python3 convert_onnx_to_mnn_simple.py models/policy.onnx models/policy.mnn
        """
    )
    
    parser.add_argument(
        "onnx_path",
        type=str,
        help="输入的 ONNX 模型文件路径"
    )
    
    parser.add_argument(
        "mnn_path",
        type=str,
        nargs="?",
        default=None,
        help="输出的 MNN 模型文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    success = convert_onnx_to_mnn(args.onnx_path, args.mnn_path)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
