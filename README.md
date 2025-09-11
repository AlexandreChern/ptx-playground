# PTX Playground

A simple environment for writing and experimenting with hand-written CUDA PTX kernels. This repository is primarily for my blog post available at http://philipfabianek.com/posts/cuda-ptx-introduction.

## Prerequisites

- A C++ compiler (g++, clang++, etc.)
- The NVIDIA CUDA Toolkit (nvcc)
- GPU with compute capability of at least 7.0

## Build

You can compile the program using `nvcc`:

```bash
nvcc main.cu -o ptx_runner -lcuda
```

## Run

To execute the hand-written PTX kernel, simply run the compiled executable:

```bash
./ptx_runner
```

## License

This project is licensed under the MIT License.
