# Memboost: CUDA 2-bit Inference Engine for LLMs

**Goal:** Take fp16 weights and compress them to mixed-precision 2-bit/4-bit using custom CUDA kernels.

**Algorithm:** Intra-matrix mixed-precision quantization based on ["Fast and Efficient 2-bit LLM Inference on GPU" (arXiv:2311.16442)](https://arxiv.org/abs/2311.16442).

![Range-aware quantization](assets/image-1.png)
![Calculation of quantization parameters](assets/image.png)

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+
- PyTorch (with CUDA support)

### Install from Source

```bash
# Set CUDA_HOME if not in standard path
export CUDA_HOME=/usr/local/cuda

# Install in editable mode
pip install -e .
```

## Python Usage

Memboost provides a direct Python API for quantization and dequantization. All operations run on the GPU.

```python
import torch
import memboost

# 1. Create dummy weights (must be FP16 on CUDA)
weights = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

# 2. Quantize
# ratio_4bit=0.1 means 10% of groups will use 4-bit precision (rest 2-bit)
q_tensor = memboost.quantize(weights, ratio_4bit=0.1)

print(f"Compressed size: {q_tensor.avg_bits:.2f} bits/weight")

# 3. Dequantize
w_hat = memboost.dequantize(q_tensor)

# Check error
error = (weights - w_hat).abs().mean()
print(f"Reconstruction Error: {error.item():.4f}")

# 4. Save/Load
torch.save(q_tensor.state_dict(), "model_quantized.pt")
loaded_q = memboost.QuantizedTensor.from_state_dict(torch.load("model_quantized.pt"))
```

## Low-Level Operations

You can also use raw pack/unpack operations:

```python
# Pack 16 uint8 values (0-3) into one int32
values = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.uint8)
packed = memboost.pack_2bit(values)

# Unpack
unpacked = memboost.unpack_2bit(packed, num_elements=16)
```

## Running C++ Unit Tests

To compile and run the standalone CUDA C++ tests without Python:

```bash
nvcc -DTEST_QUANTIZE -o test_quantize core/quantize.cu -lcusparse
./test_quantize
```
