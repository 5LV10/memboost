// Torch-facing declarations for memboost CUDA operations
#pragma once

#include <torch/types.h>
#include <vector>

namespace memboost {

// Pack / Unpack (CPU tensor operations)
torch::Tensor pack_2bit_op(torch::Tensor values);
torch::Tensor unpack_2bit_op(torch::Tensor packed, int64_t num_elements);
torch::Tensor pack_4bit_op(torch::Tensor values);
torch::Tensor unpack_4bit_op(torch::Tensor packed, int64_t num_elements);

// Quantize / Dequantize (CUDA tensor operations)
std::vector<torch::Tensor> quantize_op(
    torch::Tensor weights,
    double ratio_4bit,
    torch::Tensor hessian_diag  // pass empty tensor if not provided
);

torch::Tensor dequantize_op(
    torch::Tensor packed_2bit,
    torch::Tensor packed_4bit,
    torch::Tensor scales_1st,
    torch::Tensor zeros_1st,
    torch::Tensor group_precision,
    int64_t M, int64_t K
);
}
