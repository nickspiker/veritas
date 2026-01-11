# Bolt-on Architecture: Neural + Symbolic Verification

**Status:** Prototype working, training loop needs tuning

---

## The Actual Stack (Not Mandelbrot Poetry)

```
┌─────────────────────────────────────────┐
│      Neural Network (Rust + Spirix)     │
│   - Learns patterns from data           │
│   - Pure Spirix (no IEEE)               │
│   - Predicts answers                    │
└─────────────────┬───────────────────────┘
                  │
                  │ prediction
                  ▼
┌─────────────────────────────────────────┐
│      Symbolic Bolt-ons (Verifiers)      │
│   - basecalc: arithmetic                │
│   - Proof checker: logic                │
│   - Code executor: computation          │
│   - Returns VERIFIED ground truth       │
└─────────────────┬───────────────────────┘
                  │
                  │ verification
                  ▼
┌─────────────────────────────────────────┐
│        Training Signal Generator        │
│   error = neural_answer - symbolic_truth│
│   backprop(error)                       │
│   update_weights()                      │
└─────────────────────────────────────────┘
```

---

## What We Built

### 1. Arithmetic Bolt-on (`src/symbolic/arithmetic.rs`)

A lightweight symbolic arithmetic engine:
- Parses simple expressions: `"2 + 3"`, `"8 / 2"`, etc.
- Computes verified answers using **pure Spirix**
- Generates training problems
- No IEEE contamination

**Example:**
```rust
let problem = ArithProblem::new(
    ScalarF4E4::from(2u8),
    ScalarF4E4::from(3u8),
    ArithOp::Add
);

let result = problem.solve().unwrap();
// result.answer = 5 (VERIFIED, no guessing)
```

### 2. Training Loop (Phase 3 example)

Neural network learns arithmetic by:
1. Generator creates random problems
2. Symbolic engine solves them (ground truth)
3. Neural network predicts answers
4. Compare: `error = neural - symbolic`
5. Backprop the contradiction
6. Repeat until converged

**Key insight:** Symbolic engine provides EXACT answers. Neural learns the patterns, symbolic verifies the truth.

---

## Why This Works

### Traditional LLM Training:
```
Input → Neural network → Output
                         ↓
                    "Looks right" (pattern match)
```

### Veritas Bolt-on Training:
```
Input → Neural network → Prediction
                         ↓
                    Symbolic engine → VERIFIED answer
                         ↓
                    error = prediction - verified
                         ↓
                    backprop(error)
```

**Difference:** We have EXACT ground truth, not "looks approximately correct."

---

## The 1% Advantage

1. **No Python bottleneck** - Pure Rust
2. **No IEEE violations** - Pure Spirix
3. **Symbolic verification as native training signal** - Not post-hoc checking
4. **No token limits** - Compute until converged
5. **Verifiable answers** - Not probabilistic guessing

---

## Current Status

✅ **Working:**
- Arithmetic bolt-on generates verified answers
- Training loop runs (neural + symbolic)
- Pure Spirix (zero IEEE contamination)
- Problem generator working

⚠️ **Needs Tuning:**
- Network architecture (currently 3 → 4 → 1, too small?)
- Learning rate (0.05 might be wrong)
- Normalization (inputs are 0-10, outputs are 0-90, scale mismatch)
- More training epochs

❌ **Not Built Yet:**
- Full basecalc integration (complex expressions)
- Proof checker bolt-on
- Code executor bolt-on
- Input compression → tensor encoding
- Multi-head iteration

---

## Integration with basecalc

**basecalc** is a full-featured calculator:
- Arbitrary base (binary to base 36)
- Arbitrary precision (113 digits default)
- Complex numbers
- Trigonometry, logarithms, etc.
- Uses GMP/MPFR (not Spirix)

**Integration plan:**
1. Use basecalc parser for complex expressions
2. Convert basecalc results to Spirix for neural training
3. Neural learns patterns, basecalc verifies truth
4. For simple arithmetic, use our lightweight bolt-on
5. For complex math, delegate to basecalc

---

## Next Steps

1. **Fix Phase 3 training** - tune architecture/learning rate
2. **Add normalization** - scale inputs/outputs to [0, 1]
3. **Build proof checker bolt-on** - verify logic/symbolic expressions
4. **Build code executor bolt-on** - run and verify code
5. **Input compression working** - extract signal from noise
6. **End-to-end pipeline** - user query → verified answer

---

## The Core Philosophy

> **"Symbolic engines provide EXACT answers.**
> **Neural learns the patterns, symbolic verifies the truth.**
> **Contradictions drive learning."**

Not token prediction. Not pattern matching. **Verified computation.**

---

## File Structure

```
veritas/
├── src/
│   ├── symbolic/
│   │   ├── arithmetic.rs    ← Arithmetic bolt-on (NEW)
│   │   ├── expr.rs          ← Symbolic expressions
│   │   └── eval.rs          ← Expression evaluator
│   ├── autograd/            ← Neural network (Spirix)
│   └── encoding/            ← Input compression
└── examples/
    ├── phase1_simple_identity.rs    ← Learn f(x) = x
    ├── phase2_logic_gates.rs        ← Learn AND/OR/XOR
    └── phase3_arithmetic_bolton.rs  ← Arithmetic bolt-on (NEW)
```

---

**Let's fucking go.**
