# AMD Navi 24 (RX 6500 XT) - Processor Counts

## GPU Architecture Overview
- **GPU Codename**: Navi 24 "Beige Goby"
- **Architecture**: RDNA 2.0
- **Process Node**: TSMC 6nm
- **Die Size**: 107 mm²
- **Transistor Count**: 5.4 billion transistors
- **Transistor Density**: 50.5M transistors/mm²

---

## Compute Resources

### Compute Units (CUs) and Shader Arrays
- **Compute Units (CUs)**: **16 total**
- **Work Group Processors (WGPs)**: **8 total** (2 CUs per WGP)
- **Shader Arrays**: **2 arrays** (4 WGPs per array)

### Stream Processors / Shader Cores
- **Stream Processors**: **1,024 total**
  - 64 stream processors per CU
  - 128 stream processors per WGP
  - 512 stream processors per shader array

In RDNA 2 architecture:
- Each CU contains 2 SIMD32 units = 64 shader cores
- Each wave executes on 32-wide SIMD units (RDNA 2 native wave size)
- **32 threads per wave** (changed from 64 in GCN/RDNA 1)

### Vector ALUs per CU
Each Compute Unit contains:
- **2 × SIMD32 vector units** (64 lanes total per CU)
- Each SIMD32 can execute:
  - 32 × FP32 operations per clock
  - 32 × INT32 operations per clock
  - 64 × FP16 operations per clock (dual-issue packed)
  
**Total Vector Processing Power:**
- 1,024 FP32 operations per clock (at full occupancy)
- 2,048 FP16 operations per clock (packed operations)

### Scalar ALUs
- **16 scalar units** (1 per CU)
- Each scalar unit can execute scalar operations for one wave

---

## Texture and Rasterization Units

### Texture Mapping Units (TMUs)
- **64 TMUs total**
  - 4 TMUs per CU
  - 8 TMUs per WGP
  - 32 TMUs per shader array

Each TMU can:
- Sample 1 texel per clock (bilinear)
- Address/sample from texture cache
- Support all texture formats and filtering modes

**Texture Fill Rate**: 180.2 GT/s (at 2,815 MHz boost)

### Render Output Units (ROPs)
- **32 ROPs total**
  - 16 ROPs per shader array

Each ROP can:
- Write 1 pixel per clock
- Handle depth/stencil operations
- Perform blend operations
- Apply MSAA resolve

**Pixel Fill Rate**: 90.1 GP/s (at 2,815 MHz boost)

---

## Ray Tracing Accelerators

### Ray Accelerators (RAs)
- **16 Ray Accelerators** (1 per CU)

Each Ray Accelerator can:
- Process 4 ray-box intersections per clock
- Process 1 ray-triangle intersection per clock
- Handle BVH (Bounding Volume Hierarchy) traversal
- Hardware-accelerated intersection testing

**Total RT Performance**:
- 64 box tests per clock (16 RAs × 4)
- 16 triangle tests per clock (16 RAs × 1)
- ~5.7 GRays/sec BVH traversal performance

---

## Memory Subsystem

### L0 Vector Cache
- **16 KB per CU** (256 KB total)
- Local to each CU
- Lowest latency access for vector operations

### L1 Cache
- **128 KB per shader array** (256 KB total)
- Shared by 4 WGPs (8 CUs) per array
- Unified for instructions and data

### L2 Cache
- **1 MB per shader array** (2 MB total)
- Shared across compute units
- Connects to Infinity Cache

### Infinity Cache (L3)
- **16 MB total** (Last Level Cache)
- Shared across entire GPU
- Acts as victim cache for L2 misses
- Provides high-bandwidth local storage
- Reduces main memory traffic

### Memory Controllers
- **2 × 32-bit GDDR6 controllers** (64-bit total bus width)
- Each controller interfaces with 2 GB GDDR6 (4 GB total standard)
- 18 Gbps effective memory speed
- **Raw Bandwidth**: 144 GB/s
- **Effective Bandwidth** (with Infinity Cache): ~231.6 GB/s

---

## Fixed-Function Units

### Display Controllers
- **1 Display Controller Engine**
- Supports:
  - 1 × HDMI 2.1 (4K@120Hz, 8K@60Hz)
  - 1 × DisplayPort 1.4a with DSC
  - HDR10, FreeSync Premium

### Video Encode/Decode
- **1 × VCN 3.0 (Video Core Next) engine**
  - H.264 decode: Up to 4K@120fps
  - H.265 (HEVC) decode: Up to 8K@60fps
  - AV1 decode: Up to 8K@60fps
  - **Note**: Navi 24 lacks hardware encode engines (encoder removed for market segmentation)

### DMA Engines
- **1 × SDMA (System DMA) engine**
  - Handles memory-to-memory copies
  - Transfers between VRAM and system RAM
  - Offloads data movement from compute units

### Geometry Processing
- **2 Geometry Processors** (1 per shader array)
- **2 Primitive Units** (1 per shader array)
- Handle:
  - Vertex fetch and assembly
  - Primitive assembly
  - Tessellation (if enabled)
  - Geometry culling

---

## Command Processors and Scheduling

### Graphics Command Processor
- **1 Graphics Command Processor (GCP)**
- Manages graphics workload submission
- Handles draw calls and state changes

### Asynchronous Compute Engines (ACE)
- **2 Asynchronous Compute Engines**
- Each can schedule independent compute workloads
- Enable concurrent graphics + compute
- Support up to 8 compute queues total

### Workgroup Schedulers
- **16 CU schedulers** (1 per CU)
- Schedule waves onto SIMD units
- Handle wave priority and resource allocation

---

## Clock Speeds

### Base Specifications (RX 6500 XT)
- **Base Clock**: 2,310 MHz
- **Game Clock**: 2,610 MHz (typical sustained)
- **Boost Clock**: 2,815 MHz (peak)

These are the highest clocks of any RDNA 2 GPU due to the small die size and 6nm process.

---

## PCIe Interface

### Host Connection
- **PCIe 4.0 × 4 lanes** (16 GB/s bidirectional)
- **Not × 16** - This is a major limitation for PCIe 3.0 systems
- PCIe 3.0 × 4 provides only ~4 GB/s each direction
- Can bottleneck in scenarios requiring host memory access

---

## Power Delivery

### Voltage Regulators
- Multiple voltage rails for:
  - GPU core (VDDGFX)
  - Memory (VDDMEM)  
  - I/O (VDDIO)
  - PCIe (VDD_PCI)

### Power Monitoring
- Per-rail current and voltage sensors
- Thermal monitoring (multiple temperature sensors)
- Power management controllers

### TDP
- **107W Total Graphics Power (TGP)**
  - ~85W GPU core
  - ~15W memory
  - ~7W I/O and other components

---

## Total Processor Summary

### Parallel Processing Elements
| Processor Type | Count | Total Operations/Clock |
|----------------|-------|------------------------|
| Stream Processors (FP32) | 1,024 | 1,024 FP32 ops |
| Stream Processors (FP16) | 1,024 | 2,048 FP16 ops (packed) |
| Scalar Units | 16 | 16 scalar ops |
| Texture Units | 64 | 64 texture samples |
| ROPs | 32 | 32 pixels |
| Ray Accelerators | 16 | 64 box tests, 16 tri tests |

### Computing Resources
| Resource Type | Count | Capacity |
|---------------|-------|----------|
| Compute Units | 16 | - |
| Work Group Processors | 8 | - |
| Shader Arrays | 2 | - |
| ACEs (Async Compute) | 2 | 8 queues |
| Display Engines | 1 | 2 outputs |
| Video Decode Engines | 1 | - |
| DMA Engines | 1 | - |

### Cache Hierarchy
| Cache Level | Total Size | Configuration |
|-------------|------------|---------------|
| L0 Vector Cache | 256 KB | 16 KB × 16 CUs |
| L1 Cache | 256 KB | 128 KB × 2 arrays |
| L2 Cache | 2 MB | 1 MB × 2 arrays |
| L3 (Infinity Cache) | 16 MB | Shared across GPU |

---

## Compute Performance

### Peak Theoretical Performance
- **FP32 (Single Precision)**: 5.765 TFLOPs
  - 1,024 cores × 2,815 MHz × 2 ops/clock
- **FP16 (Half Precision)**: 11.53 TFLOPs  
  - Packed dual-issue operations
- **INT32 (Integer)**: 5.765 TIOPS
- **INT8 (Dot Product)**: 23.06 TOPS
  - 4× throughput via packed operations

### Memory Bandwidth
- **Raw GDDR6**: 144 GB/s
- **Effective (with Infinity Cache)**: ~231.6 GB/s
- **Infinity Cache Bandwidth**: 512 GB/s (estimated internal)

---

## Comparison to Other RDNA 2 Chips

| Chip | CUs | Stream Processors | ROPs | TMUs | Infinity Cache | Memory Bus |
|------|-----|-------------------|------|------|----------------|------------|
| **Navi 24** (6500 XT) | **16** | **1,024** | **32** | **64** | **16 MB** | **64-bit** |
| Navi 23 (6600 XT) | 32 | 2,048 | 64 | 128 | 32 MB | 128-bit |
| Navi 22 (6700 XT) | 40 | 2,560 | 64 | 160 | 96 MB | 192-bit |
| Navi 21 (6800 XT) | 72 | 4,608 | 128 | 288 | 128 MB | 256-bit |

Navi 24 is exactly **1/4.5** the size of the flagship Navi 21 in terms of compute units.

---

## Notes

- Each CU in RDNA 2 is more powerful than GCN CUs due to architectural improvements
- Wave32 execution (vs Wave64 in GCN) improves efficiency and reduces divergence penalties
- Infinity Cache dramatically reduces memory bandwidth requirements
- PCIe × 4 limitation was primarily for cost reduction (intended for mobile/laptops)
- Missing hardware video encoders distinguish it from mobile variants
- The 6nm node allows for higher clocks than 7nm RDNA 2 parts
- Designed for 1080p gaming, struggles at higher resolutions due to memory bandwidth

---

## Architecture Efficiency

**Instructions Per Clock (IPC) Improvements over RDNA 1:**
- ~18% higher IPC for gaming workloads
- Better cache hit rates with Infinity Cache
- Improved instruction scheduling
- Ray tracing acceleration hardware

**Performance Per Watt:**
- ~65% better than Navi 14 (RX 5500 XT) at similar performance
- Efficient 6nm process
- Aggressive power gating
- Fine-grained clock control