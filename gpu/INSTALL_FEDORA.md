# Installing ROCm on Fedora for RX 6400/6500 XT

## Your Hardware

**GPU:** AMD Radeon RX 6400/6500 XT (Navi 24)
**Architecture:** RDNA2 (gfx1034)
**Compute Units:** 16 (RX 6500 XT) or 12 (RX 6400)
**Memory:** 4GB GDDR6

Good news: RDNA2 is well-supported by ROCm.

## Installation Options

### Option 1: Install from AMD's ROCm Repository (Recommended)

AMD provides official ROCm packages, but primarily targets Ubuntu/RHEL. For Fedora, we need to use the RHEL packages.

**NOT RECOMMENDED** - Fedora and RHEL have different library versions, may cause conflicts.

### Option 2: Use Mesa's OpenCL (Minimal, but works)

Fedora ships with Mesa's `rusticl` OpenCL implementation, which works on RDNA2 for basic compute.

```bash
# Install Mesa OpenCL (rusticl)
sudo dnf install mesa-libOpenCL mesa-libOpenCL-devel clinfo

# Install LLVM and Clang
sudo dnf install llvm clang lld

# Verify OpenCL works
clinfo
```

**Limitation:** This gives you OpenCL, but NOT HIP. You'd need to write OpenCL kernels instead of HIP.

### Option 3: Build ROCm from Source (Advanced)

ROCm can be built from source on Fedora. Takes 2-4 hours.

**Steps:**
1. Clone ROCm: https://github.com/RadeonOpenCompute/ROCm
2. Follow build instructions for "unofficial platforms"
3. Target gfx1034 specifically

**Complexity:** High, but gives you full HIP support.

### Option 4: Use Container (Easiest for testing)

AMD provides official ROCm Docker containers:

```bash
# Install Docker/Podman
sudo dnf install podman

# Pull ROCm container
podman pull rocm/dev-ubuntu-22.04:latest

# Run with GPU access
podman run -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    rocm/dev-ubuntu-22.04:latest

# Inside container:
hipcc --version
```

**Advantage:** Full ROCm stack, isolated from system.
**Disadvantage:** Need to develop inside container.

## Recommended Path: Hybrid Approach

For **proving the concept** quickly:

1. **Use OpenCL for now** (Option 2)
2. **Rewrite kernel as OpenCL** (similar to HIP)
3. **Benchmark and validate hypothesis**
4. **Later:** Move to full ROCm in container if needed

### Why OpenCL is Fine

HIP and OpenCL are very similar. Our kernel translates easily:

**HIP version:**
```cpp
__global__ void spirix_matmul_kernel(...) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // ...
}
```

**OpenCL version:**
```c
__kernel void spirix_matmul_kernel(...) {
    int row = get_global_id(1);
    // ...
}
```

The **core Spirix operations** (`spirix_mul`, `spirix_add`) are IDENTICAL.

The **key hypothesis** (zero branch divergence) is testable with either.

## Quick Start: OpenCL Path

Let me create an OpenCL version of the kernel for you:

```bash
# Install OpenCL
sudo dnf install mesa-libOpenCL mesa-libOpenCL-devel clinfo opencl-headers

# Verify GPU is detected
clinfo | grep -A5 "Device Name"
```

Should show your Radeon RX 6500 XT.

Then I'll convert `spirix_matmul.hip` to `spirix_matmul.cl` (OpenCL).

## Alternative: Test on Your NVIDIA Box

You mentioned you have an NVIDIA GPU too. If that's more accessible:

```bash
# NVIDIA CUDA installation (much easier)
sudo dnf install cuda-toolkit

# Then use CUDA version of kernel
# (HIP code translates to CUDA almost 1:1)
```

**Question:** Which path do you prefer?

1. **OpenCL on AMD** (quick, minimal install, proves concept)
2. **ROCm in container on AMD** (full stack, isolated)
3. **CUDA on NVIDIA** (easiest install, if you have access)
4. **Build ROCm from source** (full control, most work)

I can set up whichever you choose.
