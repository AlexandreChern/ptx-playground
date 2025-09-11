#include <cstdio>
#include <fstream>
#include <vector>

#include <cuda.h> // CUDA Driver API

// Error checking for CUDA Driver API calls,
// do not confuse this with the standard macro used for
// CUDA Runtime API calls (which use cudaError_t).
#define CUDA_CHECK(result)                                                     \
  if (result != CUDA_SUCCESS) {                                                \
    const char *msg;                                                           \
    cuGetErrorName(result, &msg);                                              \
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", msg, __FILE__, __LINE__);     \
    exit(EXIT_FAILURE);                                                        \
  }

int main() {
  // Vector size
  const int N = 128;

  // Initialize the CUDA Driver API
  CUDA_CHECK(cuInit(0));
  CUdevice device;
  CUDA_CHECK(cuDeviceGet(&device, 0));
  CUcontext context;
  CUDA_CHECK(cuCtxCreate(&context, 0, device));

  // Load the PTX file
  std::ifstream ptx_file("add_kernel.ptx");
  std::string ptx_code((std::istreambuf_iterator<char>(ptx_file)),
                       std::istreambuf_iterator<char>());
  ptx_file.close();

  // Store it in a module
  CUmodule module;
  CUDA_CHECK(cuModuleLoadData(&module, ptx_code.c_str()));

  // Get the kernel function from the module,
  // notice that the 'add_kernel' name is same as the one in the PTX file
  CUfunction kernel;
  CUDA_CHECK(cuModuleGetFunction(&kernel, module, "add_kernel"));

  // Define host vectors
  std::vector<float> h_a(N), h_b(N), h_c(N, 0.0f);
  for (int i = 0; i < N; i++) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }

  // Prepare device vectors
  CUdeviceptr d_a, d_b, d_c;
  CUDA_CHECK(cuMemAlloc(&d_a, N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_b, N * sizeof(float)));
  CUDA_CHECK(cuMemAlloc(&d_c, N * sizeof(float)));

  CUDA_CHECK(cuMemcpyHtoD(d_a, h_a.data(), N * sizeof(float)));
  CUDA_CHECK(cuMemcpyHtoD(d_b, h_b.data(), N * sizeof(float)));

  // Launch the kernel
  int threads_per_block = 128;
  int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

  // Kernel parameters must be an array of void pointers
  void *kernel_params[] = {&d_a, &d_b, &d_c, (void *)&N};

  CUDA_CHECK(cuLaunchKernel(kernel,
                            // Grid dimensions
                            blocks_per_grid, 1, 1,
                            // Block dimensions
                            threads_per_block, 1, 1,
                            // Shared memory size and stream
                            0, nullptr,
                            // Kernel parameters
                            kernel_params,
                            // Extra options
                            nullptr));

  // Copy the result back to host
  CUDA_CHECK(cuMemcpyDtoH(h_c.data(), d_c, N * sizeof(float)));

  // Verify the result
  bool success = true;
  for (int i = 0; i < N; i++) {
    float expected = h_a[i] + h_b[i];
    if (h_c[i] != expected) {
      printf("Mismatch at index %d: %f != %f\n", i, h_c[i], expected);
      success = false;
      break;
    }
  }

  if (success) {
    printf("Success! The PTX kernel worked correctly.\n");
  }

  // Cleanup
  CUDA_CHECK(cuMemFree(d_a));
  CUDA_CHECK(cuMemFree(d_b));
  CUDA_CHECK(cuMemFree(d_c));
  CUDA_CHECK(cuModuleUnload(module));
  CUDA_CHECK(cuCtxDestroy(context));

  return 0;
}