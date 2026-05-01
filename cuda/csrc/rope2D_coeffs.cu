// cuda/csrc/rope2D_coeffs.cu
// Torch
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <math_constants.h>

// RayRoPE
#include "rayrope.h"

namespace rayrope {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void thread_rope2D_coeffs_fwd (
    // inputs
    const uint32_t B,              // batch
    const uint32_t H,              // num_heads
    const uint32_t N,              // seqlen (= C * P)
    const uint32_t half_feat_dim,  // feat_dim
    const uint32_t num_freqs,      // len of freqs
    // const uint32_t coord_dim = 2,      //
    const uint32_t patches_x,
    const uint32_t patches_y,
    const scalar_t *__restrict__ feats,     // (B, H, N, feat_dim)
    const scalar_t *__restrict__ log_min_freqs, // (coord_dim,)
    const scalar_t *__restrict__ log_max_freqs, // (coord_dim,)
    // output
    scalar_t *__restrict__ out             // (B, H, N, feat_dim)
) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total_elements = B * H * N * half_feat_dim;
    if (idx >= total_elements) {
        return;
    }

    // Index Decoding
    uint32_t d_out = idx % half_feat_dim;
    uint32_t tmp   = idx / half_feat_dim;
    uint32_t n_idx = tmp % N;
    tmp            = tmp / N;
    uint32_t h_idx = tmp % H;
    uint32_t b_idx = tmp / H;

    uint32_t f_idx = d_out / 2;
    uint32_t c_idx = d_out % 2;
    uint32_t p_idx = n_idx % (patches_x * patches_y); 

    // Get frequency
    float l_min_f = static_cast<float>(log_min_freqs[c_idx]);
    float l_max_f = static_cast<float>(log_max_freqs[c_idx]);
    float f_step = (num_freqs > 1) ? (l_max_f - l_min_f) / static_cast<float>(num_freqs - 1) : 0.0f;

    float freq = expf(l_min_f + f_step * static_cast<float>(f_idx));

    // Get 2D position
    float pos_x = static_cast<float>(p_idx % patches_x);
    float pos_y = static_cast<float>(p_idx / patches_x);

    float pos = (c_idx == 0) ? pos_x : pos_y;
    float angle = pos * freq;
    
    // Get sinusoidal
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Features
    uint64_t feat_dim = 2 * half_feat_dim;
    uint64_t feat_idx = b_idx * (H * N * feat_dim) + h_idx * (N * feat_dim) + n_idx * feat_dim;
    
    uint64_t x1_idx = feat_idx + d_out;
    uint64_t x2_idx = feat_idx + half_feat_dim + d_out;

    float x1 = static_cast<float>(feats[x1_idx]);
    float x2 = static_cast<float>(feats[x2_idx]);

    // Rotation
    float x1_out = x1 * cos_val - x2 * sin_val;
    float x2_out = x1 * sin_val + x2 * cos_val;

    out[x1_idx] = static_cast<scalar_t>(x1_out);
    out[x2_idx] = static_cast<scalar_t>(x2_out);
}

template <typename scalar_t>
__global__ void thread_rope2D_coeffs_bwd(
    // fwd_inputs
    const uint32_t B,              // batch
    const uint32_t H,              // num_heads
    const uint32_t N,              // seqlen
    const uint32_t half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
    const uint32_t num_freqs,      // len of freqs
    const uint32_t patches_x,
    const uint32_t patches_y,
    // const uint32_t coord_dim = 2,      // 
    const scalar_t *__restrict__ log_min_freqs, // (coord_dim,)
    const scalar_t *__restrict__ log_max_freqs, // (coord_dim,)
    // fwd_output
    // grad_output
    const scalar_t *__restrict__ v_out,
    // grad_input
    scalar_t *__restrict__ v_feats // (B, H, N, feat_dim)
) {
    const uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total_elements = B * H * N * half_feat_dim;
    if (idx >= total_elements) {
        return;
    }

    // Index Decoding
    uint32_t d_out = idx % half_feat_dim;
    uint32_t tmp   = idx / half_feat_dim;
    uint32_t n_idx = tmp % N;
    tmp            = tmp / N;
    uint32_t h_idx = tmp % H;
    uint32_t b_idx = tmp / H;

    uint32_t f_idx = d_out / 2;
    uint32_t c_idx = d_out % 2;
    uint32_t p_idx = n_idx % (patches_x * patches_y); 
    
    // Get frequency
    float l_min_f = static_cast<float>(log_min_freqs[c_idx]);
    float l_max_f = static_cast<float>(log_max_freqs[c_idx]);
    float f_step = (num_freqs > 1) ? (l_max_f - l_min_f) / static_cast<float>(num_freqs - 1) : 0.0f;

    float freq = expf(l_min_f + f_step * static_cast<float>(f_idx));
    
    // Get 2D position
    float pos_x = static_cast<float>(p_idx % patches_x);
    float pos_y = static_cast<float>(p_idx / patches_x);

    float pos = (c_idx == 0) ? pos_x : pos_y;
    float angle = pos * freq;

    // Get sinusoidal
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Features
    uint64_t feat_dim = 2 * half_feat_dim;
    uint64_t feat_idx = b_idx * (H * N * feat_dim) + h_idx * (N * feat_dim) + n_idx * feat_dim;
    
    uint64_t x1_idx = feat_idx + d_out;
    uint64_t x2_idx = feat_idx + half_feat_dim + d_out;

    float v_out_x1 = static_cast<float>(v_out[x1_idx]);
    float v_out_x2 = static_cast<float>(v_out[x2_idx]);

    // Get features gradient
    float v_x1 = v_out_x1 * cos_val + v_out_x2 * sin_val;
    float v_x2 = -v_out_x1 * sin_val + v_out_x2 * cos_val;

    v_feats[x1_idx] = static_cast<scalar_t>(v_x1);
    v_feats[x2_idx] = static_cast<scalar_t>(v_x2);
}

void rope2D_coeffs_fwd(
    // inputs
    const uint32_t patches_x,
    const uint32_t patches_y,
    const at::Tensor& feats,     // (B, H, N, feat_dim)
    const at::Tensor& log_min_freqs, // (coord_dim)
    const at::Tensor& log_max_freqs, // (coord_dim)
    // output
    at::Tensor& out       // (B, H, N, feat_dim)
) {
    TORCH_CHECK(feats.is_contiguous(), "feats must be contiguous");
    TORCH_CHECK(log_min_freqs.is_contiguous(), "log_min_freqs must be contiguous");
    TORCH_CHECK(log_max_freqs.is_contiguous(), "log_max_freqs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    const uint32_t B = feats.size(0);
    const uint32_t H = feats.size(1);
    const uint32_t N = feats.size(2);
    const uint32_t feat_dim = feats.size(3);
    const uint32_t half_feat_dim =  feat_dim / 2;
    
    TORCH_CHECK(half_feat_dim * 2 == feat_dim, "feat_dim must be even");
    
    const uint32_t num_freqs = half_feat_dim / 2;

    const uint64_t total_elements = B * H * N * half_feat_dim;
    if (total_elements == 0) {
        // skip the thread if there are no elements
        return;
    }

    dim3 threads(256);
    dim3 grid((total_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, 
        at::ScalarType::BFloat16,
        feats.scalar_type(),
        "thread_rope2D_coeffs_fwd",
        [&]() {
            thread_rope2D_coeffs_fwd<scalar_t>
            <<<grid,
               threads,
               shmem_size,
               at::cuda::getCurrentCUDAStream()>>>(
                // inputs
                B,              // batch
                H,              // num_heads
                N,              // seqlen
                half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
                num_freqs,      // len of freqs
                // coord_dim,      // 
                patches_x,
                patches_y,
                feats.data_ptr<scalar_t>(),     // (B, H, N, feat_dim)
                log_min_freqs.data_ptr<scalar_t>(), // (coord_dim)
                log_max_freqs.data_ptr<scalar_t>(), // (coord_dim)
                // output
                out.data_ptr<scalar_t>()             // (B, H, N, feat_dim)
            );
        }
    );
}

void rope2D_coeffs_bwd(
    // inputs
    const uint32_t patches_x,
    const uint32_t patches_y,
    const at::Tensor& log_min_freqs, // (coord_dim)
    const at::Tensor& log_max_freqs, // (coord_dim)
    // output
    // grad_ouput
    const at::Tensor& v_out,// (B, H, N, feat_dim)
    // grad_inputs
    at::Tensor& v_feats     // (B, H, N, feat_dim)
) {
    TORCH_CHECK(log_min_freqs.is_contiguous(), "log_min_freqs must be contiguous");
    TORCH_CHECK(log_max_freqs.is_contiguous(), "log_max_freqs must be contiguous");
    TORCH_CHECK(v_out.is_contiguous(), "v_out must be contiguous");
    TORCH_CHECK(v_feats.is_contiguous(), "v_feats must be contiguous");

    const uint32_t B = v_out.size(0);
    const uint32_t H = v_out.size(1);
    const uint32_t N = v_out.size(2);
    const uint32_t feat_dim = v_out.size(3);
    const uint32_t half_feat_dim =  feat_dim / 2;
    
    TORCH_CHECK(half_feat_dim * 2 == feat_dim, "feat_dim must be even");
    
    const uint32_t num_freqs = half_feat_dim / 2;

    const uint64_t total_elements = B * H * N * half_feat_dim;
    if (total_elements == 0) {
        // skip the thread if there are no elements
        return;
    }

    dim3 threads(256);
    dim3 grid((total_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, 
        at::ScalarType::BFloat16,
        v_out.scalar_type(),
        "thread_rope2D_coeffs_bwd",
        [&]() {
            thread_rope2D_coeffs_bwd<scalar_t>
            <<<grid,
               threads,
               shmem_size,
               at::cuda::getCurrentCUDAStream()>>>(
                // inputs
                B,              // batch
                H,              // num_heads
                N,              // seqlen
                half_feat_dim,  // feat_dim / 2  = (num_freqs * coord_dim)
                num_freqs,      // len of freqs
                // coord_dim,      // 
                patches_x,
                patches_y,
                log_min_freqs.data_ptr<scalar_t>(), // (coord_dim)
                log_max_freqs.data_ptr<scalar_t>(), // (coord_dim)
                // output
                // grad_output
                v_out.data_ptr<scalar_t>(),             // (B, H, N, feat_dim)
                // grad_input
                v_feats.data_ptr<scalar_t>()
            );
        }
    );
}

}  // namespace rayrope