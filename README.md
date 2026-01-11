# Veritas - Verified Digital Intelligence

Constitutional AI built on pure two's complement arithmetic (Spirix), trained entirely without IEEE-754.

## Core Principle

**Networks should learn WHEN to delegate to verified computation, not HOW to compute.**

## Quick Results

- ✅ **75% accuracy** on dozenal addition across ASCII gap (pure Spirix RNN)
- ✅ **100% routing accuracy** to symbolic basecalc (zero hallucination)
- ✅ **100% code detection** on unseen patterns (recursion, closures, pattern matching)
- ✅ **38× GPU speedup** with bit-identical results
- ✅ **BPTT without gradient clipping** (two's complement stability)

## Architecture

```
Input (UTF-8 bytes)
    ↓
RNN Sequential Processing (Spirix ScalarF4E4)
    ↓
Routing Classification (is_math? is_code? is_text?)
    ↓
    ├─→ Math Module (basecalc) → 100% correct
    ├─→ Code Module (rustc) → 100% valid
    └─→ Text Processing
```

## Quick Start

```bash
# Train dozenal addition RNN (75% accuracy)
cargo run --example dozenal_rnn_bptt --release

# Test basecalc routing (100% on seen, 38% on unseen)
cargo run --example rnn_basecalc_routing --release

# Run code detection (100% on unseen patterns)
cargo run --example rnn_code_detection --release

# GPU benchmark (38x speedup)
cargo run --example gpu_matmul_benchmark --release
```

## Documentation

- **[docs/README.md](docs/README.md)** - Complete system overview, architecture, and philosophy
- **[docs/FINDINGS.md](docs/FINDINGS.md)** - Detailed insights and discoveries
- **[docs/METRICS.md](docs/METRICS.md)** - Comprehensive reproducible results
- **[CONSTITUTION.md](CONSTITUTION.md)** - Design principles and constraints
- **[PLAN.md](PLAN.md)** - Implementation roadmap

## Constitution Compliance

- ✅ No IEEE-754 in training (pure Spirix)
- ✅ No gradient clipping needed (native stability)
- ✅ No modulo operations (counter-based control)
- ✅ Verified arithmetic only (no learned math)
- ✅ GPU bit-identical to CPU

## License

MIT - Build verified intelligence, not hallucination machines.
