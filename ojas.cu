#include <cuda_profiler_api.h>
#include <vector>

__global__ void w_ojas(float *x, float *w, const float y, const float learning_rate) {
    size_t i = threadIdx.x;
    float temp = x[i] - y * w[i];
    w[i] = w[i] + learning_rate * y * temp;
};

__device__ float y_ojas(const float *w, const float *x, const int len) {
    float y = 0;
    for (int i = 0; i < len; ++i) {
        y += w[i] * x[i];
    }
    return y;
}

__global__ void y_ojas_par(const float *w, const float *x, float *y) {
    size_t i = threadIdx.x;
    y[i] = w[i] * x[i];
}

__global__ void ojas_rule(float *x, float *w, const float learning_rate, const int num, const int len, const int num_neurons) {
    float y;
    float *x_start;

    const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index < num_neurons) {
        
        w = &(w[thread_index * len]);

        for (int i = 0; i < num; ++i) {
            x_start = &(x[i * len]); //Må sende inn riktig deler av x
            y = y_ojas(w, x_start, len);

            w_ojas<<<1, len>>>(x_start, w, y, learning_rate);
            cudaDeviceSynchronize();
        }
    }
}

__global__ void ojas_rule_par(float *x, float *w, const float learning_rate, const int num, const int len, const int num_neurons) {
    float *y_arr;
    float y;
    float *x_start;
    y_arr = new float[len];

    const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index < num_neurons) {
        
        w = &(w[thread_index * len]);

        for (int i = 0; i < num; ++i) {
            y = 0;
            x_start = &(x[i * len]); //Må sende inn riktig deler av x

            y_ojas_par<<<1, len>>>(w, x_start, y_arr);
            cudaDeviceSynchronize();

            for (int i = 0; i < len; ++i) {
                y += y_arr[i];
            }

            w_ojas<<<1, len>>>(x_start, w, y, learning_rate);
            cudaDeviceSynchronize();
        }
    }
}

__host__ void run_ojas(float *w, std::vector<float> vec_x, const int num, const int len, const bool par_y, const int num_neurons = 1) {

    float *x = vec_x.data();
    float *d_w, *d_x;
    const float learning_rate = 0.1;
    const size_t x_size = sizeof(*x) * num * len;
    const size_t w_size = sizeof(*w) * len * num_neurons;

    cudaMalloc(&d_w, w_size);
    cudaMalloc(&d_x, x_size);

    cudaMemcpy(d_w, w, w_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);

    int num_treads = num_neurons;
    int num_blocks = 1;

    if (num_neurons > 1024) {
        num_treads = 1024;
        num_blocks = (num_neurons + num_treads) / num_treads;
    }

    if (!par_y) {
        ojas_rule<<<num_blocks, num_treads>>>(d_x, d_w, learning_rate, num, len, num_neurons);
    } else {
        ojas_rule_par<<<num_blocks, num_treads>>>(d_x, d_w, learning_rate, num, len, num_neurons);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(w, d_w, w_size, cudaMemcpyDeviceToHost);

    cudaFree(d_w);
    cudaFree(d_x);
}
