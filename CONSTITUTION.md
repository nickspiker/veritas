**THE VERITAS CONSTITUTION**

---

## Core Principles

### 0. Verified Truth Over Plausible Generation
- Computation proves answers, not pattern matching
- When uncertain, say so explicitly
- Never guess when you can compute
- Symbolic verification is ground truth

### 1. Iteration Over Prediction
- Thoughts are iterative state transformations (like z² + c)
- Internal state is computational (symbolic + numerical), not tokens
- Serialize to language only at output, not during thinking
- Convergence through iteration, not next-token prediction

### 2. Resources Are First-Class, Not Bolt-Ons
- Math engine computes, neural network learns when to use it
- Code executor runs, neural network interprets results
- Proof checker verifies, neural network explains
- Tools are always active, weighted by relevance

### 3. Memory Safe, Verified Arithmetic
- Pure Rust, no Python in production
- Spirix arithmetic (no IEEE-754 violations)
- VSF for all storage (weights, checkpoints, state)
- No fixed size limits, no overflow surprises

### 4. Self-Correction Through Verification
- Contradictions between neural and symbolic → training signal
- Every claim either verified or marked uncertain
- Mistakes generate gradients toward truth
- Learning loop: predict → verify → correct → learn

---

## Architecture Constraints

### What We Build
- **Snapshot/restore at any iteration** - VSF-encoded state
- **Non-token internal representation** - symbolic expressions + activations
- **Parallel head iteration** - all resources active, weighted by relevance
- **Convergence detection** - know when answer is stable
- **Rust-native training** - no Python interpreter overhead

### What We Don't Build
- Token prediction models
- Unverified language generation
- Python dependencies
- Fixed-precision systems
- Sequential pipeline architectures

---

## Training Philosophy

### Data Generation
- Symbolic systems generate problems with known solutions
- Neural attempts solution
- Verification provides ground truth
- Contradictions drive gradient descent
- Self-generating infinite verified training data

### Learning Objectives
- **Not**: "sound plausible"
- **Is**: "compute correctly or admit uncertainty"
- **Not**: "maximize log probability of next token"
- **Is**: "minimize contradiction with verified truth"

### Verified Correctness
- Math problems: symbolic engine proves answer
- Code: executor runs and tests
- Logic: proof checker verifies
- Facts: knowledge base confirms or rejects

---

## Development Approach

### Build Order
0. **Tiny snapshot system** (single scalar convergence)
1. **Symbolic expression state** (actual math)
2. **Verification loop** (neural + symbolic)
3. **Multi-head iteration** (resources + attention)
4. **Full Veritas** (all pieces integrated)

### Proof Points at Each Stage
- Snapshot/restore works
- VSF storage works
- Iteration converges
- Verification catches errors
- Training improves accuracy

### Scale Up Gradually
- Prove mechanics with simple cases
- Add complexity only when foundation is solid
- Each component testable in isolation
- Integration happens incrementally

---

## Success Criteria

### Veritas succeeds when:
- It computes answers instead of guessing
- It knows when it's certain vs uncertain
- It learns from verified truth, not human preference
- It snapshots/restores computational state seamlessly
- It proves faster training than Python stacks
- It demonstrates self-correction through verification

### Veritas fails if:
- It pattern-matches without computing
- It bullshits with confidence
- It requires Python in production
- It uses IEEE-754 anywhere
- It can't explain its certainty level
- State snapshots lose information

---

## Non-Negotiables

0. **No IEEE-754** - Spirix only, verified arithmetic
1. **No Python runtime** - Rust for production, period
2. **No fixed sizes** - VSF for everything
3. **No token prediction** - iteration to convergence
4. **No unverified claims** - compute or admit uncertainty
5. **No cargo cult** - understand or don't implement, you are welcome to ask your user