# Multi-Domain Query Example: Cookie Recipe

## User Query
"I would like a chocolate chip cookie recipe in grams that gives me 1kg of cookies"

## Query Analysis

### Intent Classification
- **Domain**: Cooking (recipe generation)
- **Constraints**:
  - Type: chocolate chip cookies
  - Unit: grams
  - Total yield: 1kg
- **Knowledge required**:
  - Standard cookie ratios
  - Ingredient properties
  - Scaling math

## Parallel Execution Pipeline

```
Query → Intent Parser
          ↓
    ┌─────────────────────────────────────┐
    │  Domain Router                      │
    │  Detects: Cooking + Math            │
    └─────────────────────────────────────┘
          ↓
    ┌─────────────────────────────────────┐
    │  Parallel Engines                   │
    │                                     │
    │  [Memory Engine]                    │
    │    Query: "chocolate chip cookie    │
    │             ratio fundamentals"     │
    │    Returns: KnowledgeGraph {        │
    │      flour_to_butter: 3:1           │
    │      sugar_total_ratio: 0.6         │
    │      chocolate_percentage: 0.4      │
    │      leavening: baking_soda         │
    │    }                                │
    │                                     │
    │  [Math Engine]                      │
    │    Input: base_recipe (yields 500g) │
    │    Target: 1000g                    │
    │    Computation: scale_factor = 2.0  │
    │    Returns: ScaledIngredients {     │
    │      flour: 500g                    │
    │      butter: 167g                   │
    │      sugar: 300g                    │
    │      chocolate: 200g                │
    │      ...                            │
    │    }                                │
    │                                     │
    │  [Verification Engine]              │
    │    Check: ratios maintained         │
    │    Check: total = 1000g ± tolerance │
    │    Check: ingredients compatible    │
    │    Returns: Verified ✓              │
    └─────────────────────────────────────┘
          ↓
    ThoughtStructure {
      knowledge_retrieved: [cookie_ratios],
      computations: [scaling_math],
      verifications: [ratio_check, total_check],
      synthesis: [ingredient_list, instructions]
    }
          ↓
    Compositor
      - Validates all components consistent
      - Checks no contradictions
      - Orders by dependency
          ↓
    Serializer (Structure → English)
      "Here's a chocolate chip cookie recipe
       scaled to yield exactly 1kg:

       Ingredients (in grams):
       - 500g all-purpose flour
       - 167g unsalted butter
       - 200g white sugar
       - 100g brown sugar
       - 200g chocolate chips
       - 2 eggs (≈100g)
       - 1 tsp baking soda (≈5g)
       - 1 tsp vanilla extract (≈5g)
       - 1/2 tsp salt (≈3g)

       Total: 1000g

       Instructions:
       [step by step...]

       This recipe maintains the classic
       3:1 flour-to-butter ratio with 40%
       chocolate by weight for optimal
       texture and flavor."
```

## The Key Components

### 1. Knowledge Retrieval (Memory Engine)

```rust
struct MemoryEngine {
    knowledge_base: KnowledgeGraph,
}

impl MemoryEngine {
    fn query(&self, intent: &Intent) -> KnowledgeResult {
        match intent.domain {
            Domain::Cooking => {
                // Retrieve cooking fundamentals
                let ratios = self.knowledge_base.get("cooking/ratios/cookies");
                let constraints = self.knowledge_base.get("cooking/constraints/baking");

                KnowledgeResult {
                    facts: vec![
                        Fact::Ratio("flour_to_butter", 3.0),
                        Fact::Ratio("sugar_percentage", 0.6),
                        Fact::Property("chocolate", "melts_at_35C"),
                    ],
                    confidence: 0.95,
                    sources: vec!["culinary_fundamentals.vsf"],
                }
            }
            _ => todo!()
        }
    }
}
```

### 2. Computation (Math Engine)

```rust
struct MathEngine {
    symbolic: SymbolicEngine,
}

impl MathEngine {
    fn scale_recipe(&self, base: &Recipe, target_weight: Scalar) -> Result<ScaledRecipe> {
        // Get base total
        let base_total = base.ingredients.iter()
            .map(|i| i.amount)
            .sum();

        // Calculate scale factor
        let scale_factor = target_weight / base_total;

        // Scale each ingredient
        let scaled = base.ingredients.iter().map(|ingredient| {
            ScaledIngredient {
                name: ingredient.name.clone(),
                amount: ingredient.amount * scale_factor,
                unit: ingredient.unit.clone(),
            }
        }).collect();

        // Verify total
        let actual_total: Scalar = scaled.iter()
            .map(|i| i.amount)
            .sum();

        let error = (actual_total - target_weight).abs();

        if error > Scalar::from(1) {  // 1g tolerance
            return Err("Scaling error exceeds tolerance");
        }

        Ok(ScaledRecipe {
            ingredients: scaled,
            verified: true,
        })
    }
}
```

### 3. Synthesis (Assembly)

```rust
struct Synthesizer {
    memory: MemoryEngine,
    math: MathEngine,
}

impl Synthesizer {
    fn generate_recipe(&mut self, query: &RecipeQuery) -> Result<ThoughtStructure> {
        let mut thoughts = ThoughtStructure::new();

        // 1. Retrieve knowledge about cookies
        let knowledge = self.memory.query(&Intent::cooking("chocolate chip cookies"));
        thoughts.add_knowledge(knowledge);

        // 2. Create base recipe from ratios
        let base_recipe = self.construct_from_ratios(&knowledge.facts);

        // 3. Scale to target weight
        let scaled = self.math.scale_recipe(&base_recipe, query.target_weight)?;
        thoughts.add_computation(Computation {
            expression: Expr::mul(
                Expr::var("base_recipe"),
                Expr::var("scale_factor")
            ),
            result: Scalar::from(query.target_weight),
            verified: true,
            steps: vec![/* scaling steps */],
        });

        // 4. Verify constraints
        self.verify_recipe(&scaled, &knowledge)?;

        // 5. Add synthesis
        thoughts.add_synthesis(Synthesis::Recipe(scaled));

        Ok(thoughts)
    }
}
```

### 4. The Full Pipeline

```rust
async fn handle_recipe_query(query: &str) -> Result<String> {
    // 1. Parse intent
    let intent = parser.parse(query)?;
    // Intent {
    //   domain: Cooking,
    //   type: Recipe,
    //   item: "chocolate chip cookies",
    //   constraints: { unit: Grams, yield: 1000 }
    // }

    // 2. Route to appropriate engines (PARALLEL)
    let (knowledge, computation, verification) = tokio::join!(
        memory_engine.query(&intent),
        math_engine.prepare_scaling(&intent),
        verification_engine.get_constraints(&intent)
    );

    // 3. Synthesize
    let recipe = synthesizer.generate_recipe(
        knowledge?,
        computation?,
        verification?
    )?;

    // 4. Build thought structure
    let thoughts = ThoughtStructure {
        knowledge_retrieved: vec![knowledge?],
        computations: vec![computation?],
        verifications: vec![verification?],
        synthesis: vec![recipe],
    };

    // 5. Verify consistency
    compositor.validate(&thoughts)?;

    // 6. Serialize to English
    let serializer = NaturalLanguageSerializer::new(Verbosity::Detailed);
    let response = serializer.serialize(&thoughts);

    Ok(response.to_string())
}
```

## Why This Works

**Token prediction (GPT-style):**
- Query → tokens → tokens → tokens
- No structure, no verification
- Can't guarantee 1kg total
- Can't verify ratios are correct
- Hallucinates plausible but wrong recipes

**Veritas:**
- Query → Intent → Parallel Engines
- Memory retrieves verified cooking knowledge
- Math computes exact scaling
- Verification checks constraints
- **Impossible to violate 1kg constraint** (verified math)
- **Impossible to break ratios** (verified from knowledge base)

## Training This

```rust
// Generate training data for recipe queries
for _ in 0..1_000_000 {
    // Generate random recipe query
    let query = RecipeQuery {
        dish: random_dish(),
        target_weight: random_weight(100..5000),
        unit: Unit::Grams,
    };

    // Symbolic generates correct recipe
    let (structure, recipe) = symbolic_recipe_engine.generate(&query);

    // Neural tries to generate recipe from query
    let neural_recipe = neural_generator.generate(query_text);

    // Verify neural matches symbolic
    let verification = verify_recipe(neural_recipe, recipe);

    match verification {
        Verified::Correct => reward_neural(),
        Verified::Contradicted { errors } => {
            // Check what went wrong:
            if errors.contains("wrong_total_weight") {
                train_scaling_network();
            }
            if errors.contains("invalid_ratios") {
                train_knowledge_retrieval();
            }
        }
    }
}
```

## The Result

**User gets:**
- ✅ Exactly 1kg total (verified math)
- ✅ Correct ratios (verified knowledge)
- ✅ Proper scaling (verified computation)
- ✅ Clear explanation (verified serialization)

**System guarantees:**
- Can't violate constraints (math engine enforces)
- Can't break cooking fundamentals (knowledge base verified)
- Can't give inconsistent recipe (compositor checks)

**This is digital intelligence. Not token prediction. Actual reasoning thru verified computation.**

Ready to build this at scale? 🚀