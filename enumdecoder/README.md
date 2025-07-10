# QLDPC Sample Decoder

这是一个基于采样的QLDPC（Quantum Low-Density Parity-Check）解码器，使用CUDA并行计算来处理大规模错误模式。

## 项目概述

当错误数量很多时，传统的穷举低权重错误模式的方法变得不可行。本项目通过采样方法来近似计算likelihood，显著提高了处理大规模问题的能力。

### 主要特性

- **采样解码**: 使用随机采样代替穷举，适用于大规模错误模式
- **CUDA加速**: 利用GPU并行计算提高性能
- **灵活配置**: 可调整采样数量和最大权重
- **覆盖率统计**: 提供解码覆盖率和性能统计

### 文件结构

```
enumdecoder/
├── qldpc_decoder.cpp    # 原始穷举解码器
├── sample_decoder.cpp   # 新的采样解码器
├── sample.cu           # CUDA核函数
├── Makefile            # 编译配置
└── README.md           # 本文件
```

## 环境要求

### 硬件要求
- NVIDIA GPU (计算能力 >= 5.0)
- 至少2GB GPU内存

### 软件要求
- **CUDA Toolkit** (>= 9.0)
- **GCC/G++** (>= 5.4)
- **Make**

## 安装和编译

### 1. 检查CUDA环境

```bash
# 检查CUDA是否正确安装
make check_cuda

# 或者手动检查
nvcc --version
nvidia-smi
```

### 2. 编译项目

```bash
# 编译所有目标
make all

# 或者只编译采样解码器
make sample_decoder

# 或者只编译原始解码器
make qldpc_decoder
```

### 3. 清理编译文件

```bash
make clean
```

## 运行方法

### 运行采样解码器

```bash
# 使用默认设置运行
make run_sample

# 或者直接运行
./sample_decoder
```

### 运行原始解码器（对比）

```bash
# 运行原始穷举解码器
make run_orig

# 或者直接运行
./qldpc_decoder
```

## 性能对比

### 采样解码器优势

1. **内存效率**: 不需要存储所有低权重错误模式
2. **时间效率**: 采样时间远小于穷举时间
3. **可扩展性**: 可以处理更大的错误空间

### 性能参数

- **采样数量**: 默认50,000个样本
- **最大权重**: 可配置
- **并行度**: 取决于GPU规格

## 使用说明

### 基本用法

```cpp
// 创建采样解码器
SampleDecoder decoder(D, DL, priors, max_weight, num_samples);

// 解码
Vec syndrome = get_syndrome();
Vec logical_syndrome = decoder.decode(syndrome);

// 测试性能
Real error_rate = decoder.test_logical_error_rate(0.01, 1000);
```

### 参数调整

```cpp
// 调整采样数量
decoder.set_num_samples(100000);

// 更新权重
decoder.update_weights(new_priors);

// 查看覆盖率统计
decoder.print_coverage_stats();
```

## 算法原理

### 采样策略

1. **随机权重生成**: 在[0, max_weight]范围内随机选择错误权重
2. **随机位置选择**: 随机选择错误位置
3. **并行计算**: 使用CUDA并行计算症状和权重

### CUDA实现

- **错误生成**: `generate_random_errors` kernel
- **症状计算**: `compute_syndrome` kernel  
- **权重计算**: `compute_likelihood_weights` kernel

### 近似精度

采样精度取决于：
- 采样数量
- 错误分布
- 最大权重设置

## 故障排除

### 常见问题

1. **CUDA未找到**
   ```bash
   # 检查CUDA路径
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

2. **内存不足**
   - 减少采样数量
   - 使用更小的最大权重

3. **编译错误**
   - 检查CUDA版本兼容性
   - 调整Makefile中的架构设置

### 性能调优

1. **采样数量优化**
   ```cpp
   // 根据问题规模调整
   int optimal_samples = estimate_samples(problem_size);
   decoder.set_num_samples(optimal_samples);
   ```

2. **GPU内存使用**
   - 监控GPU内存使用情况
   - 必要时分批处理

## 扩展和定制

### 自定义采样策略

可以修改`sample.cu`中的采样核函数来实现：
- 偏置采样
- 重要性采样
- 自适应采样

### 集成到大型系统

```cpp
// 创建工厂函数
std::unique_ptr<SampleDecoder> create_decoder(
    const Matrix& D, const Matrix& DL, 
    const Config& config) {
    return std::make_unique<SampleDecoder>(
        D, DL, config.priors, config.max_weight, config.samples);
}
```

## 许可证

本项目基于MIT许可证发布。

## 联系方式

如有问题或建议，请联系项目维护者。 