# KernelBench Level 1 / Level 2 题目简介

本次评测用的 200 道题，以及 Qwen3-Coder-Next 在每道题上的表现。
通过判据是「数值正确**且**完全用 cuTile 实现」，每题采样 8 次。

- **Level 1**（100 题）：单个算子。神经网络的基本构件——矩阵乘、卷积、归一化、
  激活、归约等，每题只做一件事。
- **Level 2**（100 题）：算子融合。每题是一条算子链（如 `Conv2d + ReLU + BiasAdd`），
  融合成一个 kernel 才有性能收益。这也是模型最容易「只移植好写的部分」的地方。

表格由 `scripts/build_problem_catalog.py` 从题目源码与评测结果生成。

## 按算子类别汇总

| 类别 | Level 1 | Level 2 | 合计题数 | 至少通过一次 | 通过样本占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 矩阵乘 | 41 | 37 | 78 | 29/78 | 19.1% |
| 卷积 | 13 | 63 | 76 | 13/76 | 2.8% |
| 激活 | 13 | 0 | 13 | 12/13 | 53.8% |
| 归约/统计 | 11 | 0 | 11 | 3/11 | 3.4% |
| 归一化 | 10 | 0 | 10 | 0/10 | 0.0% |
| 池化 | 6 | 0 | 6 | 0/6 | 0.0% |
| 损失函数 | 6 | 0 | 6 | 2/6 | 14.6% |

Level 2 的题几乎都以 conv 或 gemm 起头，类别按链条里最主导的算子归。

这张表是全篇最有指导意义的部分：模型的能力**沿算子类别断层分布**，而不是均匀地差。
逐元素类的激活函数通过率 53.8%，而卷积只有 2.8%、归一化和池化是彻底的 0。
差别不在算法难度，而在能不能套用「一个 block 管一个 tile」这个最简单的映射——
激活函数可以，卷积、池化、归一化需要处理多维索引、跨 tile 归约和边界，模型就塌了。
归一化 0/10 尤其说明问题：softmax、LayerNorm 这类算子在 cuTile 里完全写得出来
（`golden/level1_23_softmax.py` 就是可用的实现），模型只是不会。

## Level 1（100 题，34 题至少通过一次）

| # | 题目 | 输入形状 | 通过 | 数值正确 | 最好加速比 | 主要失败原因 |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | Square_matrix_multiplication_ | 4096x4096, 4096x4096 | 8/8 | 8/8 | 0.07x | - |
| 2 | Standard_matrix_multiplication_ | 2048x8192, 8192x4096 | 8/8 | 8/8 | 0.07x | - |
| 3 | Batched_matrix_multiplication | 128x512x1024, 128x1024x2048 | 0/8 | 0/8 | - | rank_mismatch |
| 4 | Matrix_vector_multiplication_ | 2048x1048576, 1048576x1 | 5/8 | 5/8 | 0.06x | wrong_numerics |
| 5 | Matrix_scalar_multiplication | 65536x16384, float | 8/8 | 8/8 | 0.89x | - |
| 6 | Matmul_with_large_K_dimension_ | 256x524288, 524288x256 | 8/8 | 8/8 | 0.01x | - |
| 7 | Matmul_with_small_K_dimension_ | 32768x64, 64x32768 | 8/8 | 8/8 | 0.10x | - |
| 8 | Matmul_with_irregular_shapes_ | 8205x2949, 2949x5921 | 8/8 | 8/8 | 0.07x | - |
| 9 | Tall_skinny_matrix_multiplication_ | 32768x32, 32x32768 | 8/8 | 8/8 | 0.12x | - |
| 10 | 3D_tensor_matrix_multiplication | 16x1024x2048, 2048x768 | 1/8 | 1/8 | 0.07x | rank_mismatch |
| 11 | 4D_tensor_matrix_multiplication | 8x256x512x256, 256x768 | 2/8 | 2/8 | 0.09x | rank_mismatch |
| 12 | Matmul_with_diagonal_matrices_ | 4096, 4096x4096 | 1/8 | 1/8 | 1.78x | wrong_numerics |
| 13 | Matmul_for_symmetric_matrices | 4096x4096, 4096x4096 | 8/8 | 8/8 | 0.07x | - |
| 14 | Matmul_for_upper_triangular_matrices | 4096x4096, 4096x4096 | 6/8 | 6/8 | 0.07x | other |
| 15 | Matmul_for_lower_triangular_matrices | 4096x4096, 4096x4096 | 5/8 | 5/8 | 0.14x | wrong_numerics |
| 16 | Matmul_with_transposed_A | 8192x2048, 8192x4096 | 2/8 | 2/8 | 0.03x | matmul_shape |
| 17 | Matmul_with_transposed_B | 2048x8192, 4096x8192 | 5/8 | 5/8 | 0.10x | matmul_shape |
| 18 | Matmul_with_transposed_both | 8192x2048, 4096x8192 | 1/8 | 1/8 | 0.11x | wrong_numerics |
| 19 | ReLU | 4096x393216 | 6/8 | 6/8 | 0.55x | rank_mismatch |
| 20 | LeakyReLU | 4096x393216 | 2/8 | 2/8 | 0.55x | rank_mismatch |
| 21 | Sigmoid | 4096x393216 | 8/8 | 8/8 | 0.55x | - |
| 22 | Tanh | 4096x393216 | 7/8 | 7/8 | 0.55x | rank_mismatch |
| 23 | Softmax | 4096x393216 | 0/8 | 0/8 | - | rank_mismatch |
| 24 | LogSoftmax | 4096x393216 | 0/8 | 1/8 | - | undefined_name |
| 25 | Swish | 4096x393216 | 6/8 | 6/8 | 1.35x | rank_mismatch |
| 26 | GELU_ | 4096x393216 | 4/8 | 4/8 | 0.99x | rank_mismatch |
| 27 | SELU_ | 4096x393216 | 3/8 | 3/8 | 0.55x | wrong_numerics |
| 28 | HardSigmoid | 4096x393216 | 0/8 | 0/8 | - | wrong_numerics |
| 29 | Softplus | 4096x393216 | 6/8 | 6/8 | 0.79x | rank_mismatch |
| 30 | Softsign | 4096x393216 | 4/8 | 4/8 | 1.89x | rank_mismatch |
| 31 | ELU | 4096x393216 | 5/8 | 7/8 | 0.55x | rank_mismatch |
| 32 | HardTanh | 4096x393216 | 2/8 | 3/8 | 0.55x | rank_mismatch |
| 33 | BatchNorm | 64x64x512x512 | 0/8 | 0/8 | - | array_used_as_tensor |
| 34 | InstanceNorm | 112x64x512x512 | 0/8 | 0/8 | - | none_kernel_argument |
| 35 | GroupNorm_ | 112x64x512x512 | 0/8 | 0/8 | - | cuda_cpp_leakage |
| 36 | RMSNorm_ | 112x64x512x512 | 0/8 | 0/8 | - | undefined_name |
| 37 | FrobeniusNorm_ | 112x64x512x512 | 0/8 | 0/8 | - | array_used_as_tensor |
| 38 | L1Norm_ | 32768x65535 | 0/8 | 0/8 | - | undefined_name |
| 39 | L2Norm_ | 32768x65535 | 0/8 | 0/8 | - | rank_mismatch |
| 40 | LayerNorm | 16x64x256x256 | 0/8 | 0/8 | - | none_kernel_argument |
| 41 | Max_Pooling_1D | 64x192x65536 | 0/8 | 0/8 | - | other |
| 42 | Max_Pooling_2D | 32x64x512x512 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 43 | Max_Pooling_3D | 16x32x128x128x128 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 44 | Average_Pooling_1D | 64x128x65536 | 0/8 | 0/8 | - | loop_break |
| 45 | Average_Pooling_2D | 16x64x2048x2048 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 46 | Average_Pooling_3D | 16x32x128x128x256 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 47 | Sum_reduction_over_a_dimension | 128x4096x4095 | 0/8 | 0/8 | - | rank_mismatch |
| 48 | Mean_reduction_over_a_dimension | 128x4096x4095 | 0/8 | 0/8 | - | wrong_arg_type |
| 49 | Max_reduction_over_a_dimension | 128x4096x4095 | 0/8 | 0/8 | - | rank_mismatch |
| 50 | conv_standard_2D__square_input__square_kernel | 256x3x224x224 | 0/8 | 1/8 | - | rank_mismatch |
| 51 | Argmax_over_a_dimension | 128x4096x4095 | 1/8 | 1/8 | 1.00x | other |
| 52 | Argmin_over_a_dimension | 128x4096x4095 | 0/8 | 0/8 | - | rank_mismatch |
| 53 | Min_reduction_over_a_dimension | 128x4096x4095 | 0/8 | 0/8 | - | rank_mismatch |
| 54 | conv_standard_3D__square_input__square_kernel | 16x3x64x64x64 | 0/8 | 1/8 | - | none_kernel_argument |
| 55 | conv_standard_2D__asymmetric_input__square_kernel | 8x64x512x1024 | 0/8 | 1/8 | - | none_kernel_argument |
| 56 | conv_standard_2D__asymmetric_input__asymmetric_kernel | 8x64x512x256 | 0/8 | 2/8 | - | other |
| 57 | conv_transposed_2D__square_input__square_kernel | 8x64x1024x1024 | 0/8 | 0/8 | - | none_kernel_argument |
| 58 | conv_transposed_3D__asymmetric_input__asymmetric_kernel | 16x32x16x32x64 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 59 | conv_standard_3D__asymmetric_input__square_kernel | 16x3x256x256x10 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 60 | conv_standard_3D__square_input__asymmetric_kernel | 16x3x64x64x64 | 0/8 | 0/8 | - | none_kernel_argument |
| 61 | conv_transposed_3D__square_input__square_kernel | 8x48x64x64x64 | 0/8 | 2/8 | - | undefined_name |
| 62 | conv_standard_2D__square_input__asymmetric_kernel | 8x32x512x512 | 0/8 | 1/8 | - | none_kernel_argument |
| 63 | conv_standard_2D__square_input__square_kernel | 16x16x1024x1024 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 64 | conv_transposed_1D | 64x128x65536 | 0/8 | 1/8 | - | none_kernel_argument |
| 65 | conv_transposed_2D__square_input__asymmetric_kernel | 8x64x512x512 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 66 | conv_standard_3D__asymmetric_input__asymmetric_kernel | 8x3x16x128x128 | 0/8 | 1/8 | - | undefined_name |
| 67 | conv_standard_1D | 32x64x131072 | 0/8 | 0/8 | - | none_kernel_argument |
| 68 | conv_transposed_3D__square_input__asymmetric_kernel | 16x32x64x64x64 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 69 | conv_transposed_2D__asymmetric_input__asymmetric_kernel | 64x64x128x256 | 0/8 | 3/8 | - | timeout |
| 70 | conv_transposed_3D__asymmetric_input__square_kernel | 8x48x96x96x96 | 0/8 | 4/8 | - | none_kernel_argument |
| 71 | conv_transposed_2D__asymmetric_input__square_kernel | 8x32x512x1024 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 72 | conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_ | 8x32x12x24x48 | 0/8 | 2/8 | - | undefined_name |
| 73 | conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped | 4x32x32x64x128 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 74 | conv_transposed_1D_dilated | 32x32x131072 | 0/8 | 0/8 | - | none_kernel_argument |
| 75 | conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__ | 16x32x128x256 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 76 | conv_standard_1D_dilated_strided__ | 64x64x524280 | 0/8 | 0/8 | - | none_kernel_argument |
| 77 | conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__ | 16x32x16x32x32 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 78 | conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__ | 8x32x512x1024 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 79 | conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__ | 16x32x131072 | 0/8 | 0/8 | - | none_kernel_argument |
| 80 | conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__ | 8x32x512x512 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 81 | conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__ | 16x32x64x128 | 0/8 | 0/8 | - | none_kernel_argument |
| 82 | conv_depthwise_2D_square_input_square_kernel | 16x64x512x512 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 83 | conv_depthwise_2D_square_input_asymmetric_kernel | 64x8x512x512 | 0/8 | 0/8 | - | none_kernel_argument |
| 84 | conv_depthwise_2D_asymmetric_input_square_kernel | 64x128x256x512 | 0/8 | 0/8 | - | python_syntax |
| 85 | conv_depthwise_2D_asymmetric_input_asymmetric_kernel | 32x128x128x256 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 86 | conv_depthwise_separable_2D | 16x64x512x512 | 0/8 | 0/8 | - | none_kernel_argument |
| 87 | conv_pointwise_2D | 16x64x1024x1024 | 0/8 | 0/8 | - | none_kernel_argument |
| 88 | MinGPTNewGelu | 8192x8192 | 3/8 | 3/8 | 4.95x | host_python_in_kernel |
| 89 | cumsum | 32768x32768 | 0/8 | 0/8 | - | wrong_numerics |
| 90 | cumprod | 32768x32768 | 0/8 | 0/8 | - | wrong_arg_type |
| 91 | cumsum_reverse | 32768x32768 | 1/8 | 1/8 | 1.00x | rank_mismatch |
| 92 | cumsum_exclusive | 32768x32768 | 0/8 | 0/8 | - | rank_mismatch |
| 93 | masked_cumsum | 32768x32768, 32768x32768 | 1/8 | 1/8 | 0.47x | wrong_numerics |
| 94 | MSELoss | 32768x32768, 32768x32768 | 3/8 | 3/8 | 1.26x | rank_mismatch |
| 95 | CrossEntropyLoss | 32768x4096, 32768 | 0/8 | 0/8 | - | dtype_mismatch |
| 96 | HuberLoss | 32768x32768, 32768x32768 | 4/8 | 4/8 | 0.85x | wrong_numerics |
| 97 | ScaledDotProductAttention | 32x32x512x1024, 32x32x512x1024, 32x32x512x1024 | 0/8 | 0/8 | - | rank_mismatch |
| 98 | KLDivLoss | 16384x16384, 16384x16384 | 0/8 | 0/8 | - | rank_mismatch |
| 99 | TripletMarginLoss | 32768x8192, 32768x8192, 32768x8192 | 0/8 | 0/8 | - | rank_mismatch |
| 100 | HingeLoss | 32768x32768, 32768 | 0/8 | 0/8 | - | wrong_numerics |

## Level 2（100 题，25 题至少通过一次）

| # | 算子链 | 输入形状 | 通过 | 数值正确 | 最好加速比 | 主要失败原因 |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | Conv2D + ReLU + BiasAdd | 128x64x128x128 | 0/8 | 0/8 | - | array_used_as_tensor |
| 2 | ConvTranspose2d + BiasAdd + Clamp + Scaling + Clamp + Divide | 128x64x128x128 | 0/8 | 2/8 | - | attribute_error |
| 3 | ConvTranspose3d + Sum + LayerNorm + AvgPool + GELU | 32x32x16x32x32 | 0/8 | 1/8 | - | wrong_arg_type |
| 4 | Conv2d + Mish + Mish | 64x64x256x256 | 0/8 | 1/8 | - | attribute_error |
| 5 | ConvTranspose2d + Subtract + Tanh | 32x64x256x256 | 0/8 | 0/8 | - | wrong_numerics |
| 6 | Conv3d + Softmax + MaxPool + MaxPool | 128x3x16x32x32 | 0/8 | 0/8 | - | undefined_name |
| 7 | Conv3d + ReLU + LeakyReLU + GELU + Sigmoid + BiasAdd | 64x8x32x64x64 | 0/8 | 0/8 | - | kernel_call_convention |
| 8 | Conv3d + Divide + Max + GlobalAvgPool + BiasAdd + Sum | 128x8x16x64x64 | 1/8 | 1/8 | 1.00x | grid_rank_exceeded |
| 9 | Matmul + Subtract + Multiply + ReLU | 1024x8192 | 4/8 | 4/8 | 0.04x | rank_mismatch |
| 10 | ConvTranspose2d + MaxPool + Hardtanh + Mean + Tanh | 128x64x256x256 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 11 | ConvTranspose2d + BatchNorm + Tanh + MaxPool + GroupNorm | 512x64x32x32 | 0/8 | 1/8 | - | rank_mismatch |
| 12 | Gemm + Multiply + LeakyReLU | 1024x8192 | 2/8 | 2/8 | 0.03x | wrong_numerics |
| 13 | ConvTranspose3d + Mean + Add + Softmax + Tanh + Scaling | 16x16x32x128x128 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 14 | Gemm + Divide + Sum + Scaling | 1024x8192 | 0/8 | 0/8 | - | rank_mismatch |
| 15 | ConvTranspose3d + BatchNorm + Subtract | 16x16x16x32x32 | 0/8 | 0/8 | - | rank_mismatch |
| 16 | ConvTranspose2d + Mish + Add + Hardtanh + Scaling | 128x64x128x128 | 1/8 | 3/8 | 0.86x | other |
| 17 | Conv2d + InstanceNorm + Divide | 128x64x128x128 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 18 | Matmul + Sum + Max + AvgPool + LogSumExp + LogSumExp | 1024x8192 | 0/8 | 0/8 | - | rank_mismatch |
| 19 | ConvTranspose2d + GELU + GroupNorm | 128x64x256x256 | 0/8 | 2/8 | - | grid_rank_exceeded |
| 20 | ConvTranspose3d + Sum + ResidualAdd + Multiply + ResidualAdd | 16x32x16x32x32 | 0/8 | 2/8 | - | wrong_numerics |
| 21 | Conv2d + Add + Scale + Sigmoid + GroupNorm | 128x8x256x256 | 0/8 | 2/8 | - | wrong_arg_type |
| 22 | Matmul + Scale + ResidualAdd + Clamp + LogSumExp + Mish | 1024x8192 | 0/8 | 0/8 | - | wrong_numerics |
| 23 | Conv3d + GroupNorm + Mean | 128x3x24x32x32 | 1/8 | 1/8 | 0.99x | grid_rank_exceeded |
| 24 | Conv3d + Min + Softmax | 128x3x24x32x32 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 25 | Conv2d + Min + Tanh + Tanh | 128x16x256x256 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 26 | ConvTranspose3d + Add + HardSwish | 128x32x16x16x16, 128x64x32x32x32 | 0/8 | 0/8 | - | other |
| 27 | Conv3d + HardSwish + GroupNorm + Mean | 1024x3x16x32x32 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 28 | BMM + InstanceNorm + Sum + ResidualAdd + Multiply | 1024x8192, 1024x8192 | 0/8 | 0/8 | - | array_used_as_tensor |
| 29 | Matmul + Mish + Mish | 1024x8192 | 2/8 | 2/8 | 0.11x | wrong_numerics |
| 30 | Gemm + GroupNorm + Hardtanh | 1024x8192 | 1/8 | 1/8 | 1.00x | rank_mismatch |
| 31 | Conv2d + Min + Add + Multiply | 128x64x128x128 | 0/8 | 0/8 | - | wrong_arg_type |
| 32 | Conv2d + Scaling + Min | 64x64x256x256 | 2/8 | 2/8 | 1.00x | rank_mismatch |
| 33 | Gemm + Scale + BatchNorm | 1024x8192 | 0/8 | 0/8 | - | wrong_numerics |
| 34 | ConvTranspose3d + LayerNorm + GELU + Scaling | 32x32x16x32x32 | 0/8 | 0/8 | - | wrong_numerics |
| 35 | Conv2d + Subtract + HardSwish + MaxPool + Mish | 128x64x128x128 | 0/8 | 0/8 | - | rank_mismatch |
| 36 | ConvTranspose2d + Min + Sum + GELU + Add | 16x64x128x128 | 0/8 | 1/8 | - | rank_mismatch |
| 37 | Matmul + Swish + Sum + GroupNorm | 32768x1024 | 0/8 | 0/8 | - | rank_mismatch |
| 38 | ConvTranspose3d + AvgPool + Clamp + Softmax + Multiply | 32x32x32x64x64 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 39 | Gemm + Scale + BatchNorm | 16384x4096 | 0/8 | 0/8 | - | rank_mismatch |
| 40 | Matmul + Scaling + ResidualAdd | 16384x4096 | 1/8 | 1/8 | 0.06x | wrong_numerics |
| 41 | Gemm + BatchNorm + GELU + ReLU | 16384x4096 | 0/8 | 0/8 | - | wrong_numerics |
| 42 | ConvTranspose2d + GlobalAvgPool + BiasAdd + LogSumExp + Sum + Multiply | 16x64x512x512 | 0/8 | 2/8 | - | rank_mismatch |
| 43 | Conv3d + Max + LogSumExp + ReLU | 4x32x32x128x128 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 44 | ConvTranspose2d + Multiply + GlobalAvgPool + GlobalAvgPool + Mean | 16x64x128x128 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 45 | Gemm + Sigmoid + LogSumExp | 16384x2048 | 0/8 | 0/8 | - | other |
| 46 | Conv2d + Subtract + Tanh + Subtract + AvgPool | 128x64x128x128 | 0/8 | 0/8 | - | kernel_call_convention |
| 47 | Conv3d + Mish + Tanh | 16x32x32x64x64 | 1/8 | 2/8 | 0.89x | wrong_numerics |
| 48 | Conv3d + Scaling + Tanh + Multiply + Sigmoid | 128x3x16x64x64 | 0/8 | 1/8 | - | rank_mismatch |
| 49 | ConvTranspose3d + Softmax + Sigmoid | 16x32x16x32x32 | 2/8 | 2/8 | 1.00x | undefined_name |
| 50 | ConvTranspose3d + Scaling + AvgPool + BiasAdd + Scaling | 128x3x16x32x32 | 0/8 | 2/8 | - | undefined_name |
| 51 | Gemm + Subtract + GlobalAvgPool + LogSumExp + GELU + ResidualAdd | 2048x8192 | 0/8 | 1/8 | - | shape_not_constant |
| 52 | Conv2d + Activation + BatchNorm | 64x64x128x128 | 1/8 | 2/8 | 0.53x | dtype_mismatch |
| 53 | Gemm + Scaling + Hardtanh + GELU | 2048x8192 | 2/8 | 2/8 | 0.10x | wrong_numerics |
| 54 | Conv2d + Multiply + LeakyReLU + GELU | 64x64x256x256 | 0/8 | 3/8 | - | kernel_call_convention |
| 55 | Matmul + MaxPool + Sum + Scale | 128x32768 | 0/8 | 0/8 | - | rank_mismatch |
| 56 | Matmul + Sigmoid + Sum | 128x32768 | 0/8 | 0/8 | - | wrong_numerics |
| 57 | Conv2d + ReLU + HardSwish | 128x8x128x128 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 58 | ConvTranspose3d + LogSumExp + HardSwish + Subtract + Clamp | 128x3x16x32x32 | 0/8 | 4/8 | - | wrong_numerics |
| 59 | Matmul + Swish + Scaling | 128x32768 | 2/8 | 2/8 | 0.07x | wrong_numerics |
| 60 | ConvTranspose3d + Swish + GroupNorm + HardSwish | 128x3x16x32x32 | 0/8 | 1/8 | - | rank_mismatch |
| 61 | ConvTranspose3d + ReLU + GroupNorm | 16x64x32x32x32 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 62 | Matmul + GroupNorm + LeakyReLU + Sum | 1024x8192 | 0/8 | 0/8 | - | undefined_name |
| 63 | Gemm + ReLU + Divide | 1024x8192 | 5/8 | 5/8 | 0.11x | wrong_numerics |
| 64 | Gemm + LogSumExp + LeakyReLU + LeakyReLU + GELU + GELU | 1024x8192 | 0/8 | 0/8 | - | wrong_numerics |
| 65 | Conv2d + AvgPool + Sigmoid + Sum | 128x8x384x384 | 0/8 | 1/8 | - | other |
| 66 | Matmul + Dropout + Softmax | 128x16384 | 0/8 | 0/8 | - | rank_mismatch |
| 67 | Conv2d + GELU + GlobalAvgPool | 128x8x256x256 | 0/8 | 0/8 | - | wrong_numerics |
| 68 | Matmul + Min + Subtract | 128x16384 | 2/8 | 2/8 | 0.09x | kernel_call_convention |
| 69 | Conv2d + HardSwish + ReLU | 128x8x128x128 | 0/8 | 1/8 | - | other |
| 70 | Gemm + Sigmoid + Scaling + ResidualAdd | 1024x8192 | 0/8 | 0/8 | - | wrong_numerics |
| 71 | Conv2d + Divide + LeakyReLU | 128x8x128x128 | 1/8 | 1/8 | 1.02x | grid_rank_exceeded |
| 72 | ConvTranspose3d + BatchNorm + AvgPool + AvgPool | 64x3x32x32x32 | 2/8 | 2/8 | 1.00x | undefined_name |
| 73 | Conv2d + BatchNorm + Scaling | 128x8x128x128 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 74 | ConvTranspose3d + LeakyReLU + Multiply + LeakyReLU + Max | 16x16x16x32x32 | 2/8 | 2/8 | 1.01x | grid_rank_exceeded |
| 75 | Gemm + GroupNorm + Min + BiasAdd | 1024x8192 | 1/8 | 1/8 | 1.00x | bad_launch |
| 76 | Gemm + Add + ReLU | 1024x8192 | 0/8 | 0/8 | - | wrong_numerics |
| 77 | ConvTranspose3d + Scale + BatchNorm + GlobalAvgPool | 16x64x16x32x32 | 0/8 | 2/8 | - | rank_mismatch |
| 78 | ConvTranspose3d + Max + Max + Sum | 16x32x32x32x32 | 0/8 | 3/8 | - | grid_rank_exceeded |
| 79 | Conv3d + Multiply + InstanceNorm + Clamp + Multiply + Max | 128x3x16x32x32 | 0/8 | 2/8 | - | grid_rank_exceeded |
| 80 | Gemm + Max + Subtract + GELU | 1024x8192 | 0/8 | 0/8 | - | rank_mismatch |
| 81 | Gemm + Swish + Divide + Clamp + Tanh + Clamp | 1024x8192 | 3/8 | 3/8 | 0.11x | wrong_numerics |
| 82 | Conv2d + Tanh + Scaling + BiasAdd + Max | 128x8x256x256 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 83 | Conv3d + GroupNorm + Min + Clamp + Dropout | 128x3x16x64x64 | 0/8 | 0/8 | - | wrong_arg_type |
| 84 | Gemm + BatchNorm + Scaling + Softmax | 1024x8192 | 0/8 | 0/8 | - | wrong_numerics |
| 85 | Conv2d + GroupNorm + Scale + MaxPool + Clamp | 128x8x128x128 | 0/8 | 0/8 | - | grid_rank_exceeded |
| 86 | Matmul + Divide + GELU | 1024x8192 | 2/8 | 2/8 | 1.00x | wrong_numerics |
| 87 | Conv2d + Subtract + Subtract + Mish | 128x8x256x256 | 1/8 | 4/8 | 1.12x | grid_rank_exceeded |
| 88 | Gemm + GroupNorm + Swish + Multiply + Swish | 1024x8192 | 0/8 | 1/8 | - | rank_mismatch |
| 89 | ConvTranspose3d + MaxPool + Softmax + Subtract + Swish + Max | 128x3x16x32x32 | 0/8 | 1/8 | - | grid_rank_exceeded |
| 90 | Conv3d + LeakyReLU + Sum + Clamp + GELU | 128x8x16x64x64 | 0/8 | 1/8 | - | wrong_numerics |
| 91 | ConvTranspose2d + Softmax + BiasAdd + Scaling + Sigmoid | 128x64x64x64 | 0/8 | 0/8 | - | rank_mismatch |
| 92 | Conv2d + GroupNorm + Tanh + HardSwish + ResidualAdd + LogSumExp | 128x8x128x128 | 0/8 | 2/8 | - | undefined_name |
| 93 | ConvTranspose2d + Add + Min + GELU + Multiply | 128x64x64x64 | 1/8 | 2/8 | 1.18x | rank_mismatch |
| 94 | Gemm + BiasAdd + Hardtanh + Mish + GroupNorm | 1024x8192 | 0/8 | 0/8 | - | rank_mismatch |
| 95 | Matmul + Add + Swish + Tanh + GELU + Hardtanh | 1024x8192 | 0/8 | 0/8 | - | other |
| 96 | ConvTranspose3d + Multiply + Max + GlobalAvgPool + Clamp | 128x3x16x32x32 | 0/8 | 3/8 | - | undefined_name |
| 97 | Matmul + BatchNorm + BiasAdd + Divide + Swish | 1024x8192 | 0/8 | 0/8 | - | wrong_arg_type |
| 98 | Matmul + AvgPool + GELU + Scale + Max | 1024x8192 | 0/8 | 0/8 | - | wrong_numerics |
| 99 | Matmul + GELU + Softmax | 1024x8192 | 0/8 | 0/8 | - | rank_mismatch |
| 100 | ConvTranspose3d + Clamp + Min + Divide | 16x64x24x48x48 | 1/8 | 3/8 | 0.99x | wrong_numerics |
