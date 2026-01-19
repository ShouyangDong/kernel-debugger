#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

__global__ void mha_kernel(const half* h_Q, const half* h_K, const half* h_V, half* h_output,
                           int batch_size, int num_heads, int seq_len, int head_dim) {
    // Calculate the global index for the current thread
    int b = blockIdx.x; // batch index
    int h = blockIdx.y; // head index
    int s = threadIdx.x; // sequence index

    if (b < batch_size && h < num_heads && s < seq_len) {
        // Allocate shared memory for Q, K, V
        __shared__ half Q_shared[64][64]; // Adjust size as needed
        __shared__ half K_shared[64][64]; // Adjust size as needed
        __shared__ half V_shared[64][64]; // Adjust size as needed

        // Load Q, K, V into shared memory
        Q_shared[threadIdx.x][threadIdx.y] = h_Q[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s];
        K_shared[threadIdx.x][threadIdx.y] = h_K[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s];
        V_shared[threadIdx.x][threadIdx.y] = h_V[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s];

        __syncthreads();

        // Compute attention scores
        float score = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            score += __half2float(Q_shared[s][i]) * __half2float(K_shared[s][i]);
        }

        // Apply softmax
        float max_score = -FLT_MAX;
        for (int i = 0; i < seq_len; i++) {
            max_score = fmaxf(max_score, score);
        }
        float exp_sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            exp_sum += expf(__half2float(score) - max_score);
        }
        float softmax_score = expf(__half2float(score) - max_score) / exp_sum;

        // Compute output
        for (int i = 0; i < seq_len; i++) {
            h_output[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s] +=
                __float2half(softmax_score * __half2float(V_shared[s][i]));
        }
    }
}

extern "C" void launch_mha_kernel(const half* h_Q, const half* h_K, const half* h_V, half* h_output,
                                   int batch_size, int num_heads, int seq_len, int head_dim) {
    dim3 blockSize(64, 1, 1);
    dim3 gridSize(batch_size, num_heads, 1);
    mha_kernel<<<gridSize, blockSize>>>(h_Q, h_K, h_V, h_output, batch_size, num_heads, seq_len, head_dim);
    cudaDeviceSynchronize();
}