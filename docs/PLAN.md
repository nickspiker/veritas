**PLAN.md v2:**

```markdown
# VERITAS - Verified Digital Intelligence

## Core Architecture

**Not token prediction. Computational routing with verified execution.**

### Components

1. **Neural Network (Rust + Spirix)**
   - Byte-level UTF-8 input (no tokenization)
   - 12-layer transformer: attention + FFN
   - 512-dim embeddings, 2048 byte context
   - Learns: when to route, how to compose results
   - ScalarF4E4 throughout (no IEEE)

2. **Symbolic Resources (Native nodes, not external calls)**
   - `basecalc`: Arbitrary precision math (CAS)
   - `executor`: Rust compilation/execution
   - `prover`: Logic verification
   - Deterministic, verifiable, exact

3. **Integration Layer**
   - Intent generation: network → symbolic query
   - Result incorporation: verified answer → continue generation
   - Cross-verification: neural vs symbolic agreement
   - Training signal: contradictions → gradients

### Training Loop

```rust
// 1. Symbolic generates (problem, solution)
let (text, answer) = basecalc.generate_problem();

// 2. Preprocess: numbers → <MATH_N> placeholders
let (bytes, math_ast) = preprocess(text);

// 3. Neural attempts solution
let prediction = network.forward(bytes);

// 4. Extract math intent, verify
let intent = parse_intent(prediction);
let verified = basecalc.execute(intent);

// 5. Check agreement
if prediction != verified {
    backprop(verified - prediction);  // Learn routing
}
```

### Key Innovations

**No number learning:** Network never sees digits. Must route to basecalc.

**Intent-based dispatch:** Network learns semantic mapping (what to compute), tool executes (how to compute).

**Native integration:** Symbolic operations are forward pass nodes, not post-generation tool calls. Gradients flow through routing decisions.

**Verified training:** Infinite self-generated data with ground truth from symbolic engines.

## Implementation Phases

### Phase 0: Foundation (Current)
- ✅ Spirix arithmetic
- ✅ Symbolic bolt-ons (arithmetic, bitwise)
- ✅ Training infrastructure (autograd, optimizer)
- ✅ Problem generators

### Phase 1: Byte Transformer
- Byte-level embedding layer
- Multi-head attention mechanism
- Position encoding
- FFN layers with Spirix
- Next-byte prediction head

### Phase 2: Preprocessing Pipeline
- UTF-8 → byte stream
- Number detection/replacement
- Math expression → AST
- Intent vocabulary (metadata namespace)

### Phase 3: Symbolic Integration
- Intent parser (prediction → symbolic query)
- Basecalc execution hook (forward pass node)
- Result injection (verified → continue)
- Training signal (contradiction detection)

### Phase 4: Training
- Text corpus preprocessing
- Symbolic problem generation
- Training loop with verification
- Convergence monitoring

### Phase 5: Scale
- Increase model size (10M → 100M+ params)
- Expand context window
- Add more symbolic resources
- Production deployment

## Anti-Patterns (Don't Build)

- ❌ Token vocabulary learning numbers
- ❌ Post-generation tool calling (JSON output)
- ❌ Python dependencies
- ❌ IEEE-754 anywhere
- ❌ Pattern-matching math from training data

## Success Criteria

**Query:** "What's the 8th digit of pi/2 in base 7?"

**Old LLM:** Hallucinates plausible answer

**Veritas:**
1. Recognizes math query
2. Generates intent: `Divide(PI, 2) → base(7) → digit(8)`
3. Routes to basecalc
4. Returns: "3" (verified)
5. Explains: "Computed pi/2 ≈ 1.5707963..., converted to base 7 = 1.4..., 8th digit is 3"

Network didn't compute it. Routed correctly, verified result, explained clearly.

---

**Build verified intelligence, not plausible text generation.**
```

---
