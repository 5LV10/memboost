import torch
import memboost

# creating dummy weights in torch
weights = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')

# Quantize
quantized_tensor = memboost.quantize(weights, ratio_4bit=0.1) # ratio_4bit is the ratio of 4-bit weights to 2-bit weights

# Compare the total memory footprint bw the two types
original_mem = weights.element_size() * weights.numel() / (1024 * 1024)
quantized_mem = quantized_tensor.total_mb  # exact footprint

print(f"Original size:  {original_mem:.2f} MiB")
print(f"Quantized size: {quantized_mem:.2f} MiB")
print(f"Compression:    {original_mem / quantized_mem:.2f}x")
print(f"\nBreakdown:")
quantized_tensor.memory_breakdown()

