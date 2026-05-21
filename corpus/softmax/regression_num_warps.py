import torch
import triton
import triton.language as tl

# Regression: num_warps=1
# For a large BLOCK_SIZE (4096), num_warps=1 provides very little instruction-level parallelism
# and poor latency hiding for the memory operations.

@triton.jit
def softmax_kernel_broken(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax_broken(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # SEEDED REGRESSION: num_warps=1 instead of 16 (for 4096 cols)
    num_warps = 1
    y = torch.empty_like(x)
    softmax_kernel_broken[(n_rows, )](
        y, x, x.stride(0), y.stride(0), n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn((182, 4096), device='cuda')
    y_triton = softmax_broken(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch, atol=1e-5, rtol=0)
    print("✅ Triton and Torch match")
