# What is an "Idea" in Veritas?

## Token Prediction (GPT-style)

```
"Idea" = Embedding Vector
         ↓
    [0.234, -0.891, 0.445, ..., 0.123]  ← 4096 floats
         ↓
    Linear transform + softmax
         ↓
    Next token probability distribution
         ↓
    Sample token → repeat

No structure. No verification. Just vector math.
```

**Problems:**
- Can't verify correctness
- Can't explain reasoning
- Can't compose ideas logically
- Hallucinates when probabilities mislead

---

## Veritas "Idea" (Structured Graph)

```rust
/// An "idea" is a verified computation graph
struct Idea {
    /// Unique identifier
    id: IdeaId,

    /// What this idea represents
    concept: Concept,

    /// How it was derived
    derivation: DerivationGraph,

    /// Verification status
    verified: bool,

    /// Dependencies on other ideas
    dependencies: Vec<IdeaId>,
}

/// A concept can be mathematical, logical, or semantic
enum Concept {
    /// Mathematical concept
    Mathematical {
        expression: Expr,
        properties: Vec<MathProperty>,
    },

    /// Logical concept
    Logical {
        proposition: Proposition,
        truth_value: Option<bool>,
    },

    /// Semantic concept (meanings, relationships)
    Semantic {
        entity: Entity,
        relations: Vec<Relation>,
    },

    /// Computational concept (algorithms, procedures)
    Computational {
        procedure: Procedure,
        complexity: Complexity,
    },
}

/// How the idea was derived
struct DerivationGraph {
    /// Starting axioms/facts
    axioms: Vec<Axiom>,

    /// Transformation steps
    steps: Vec<DerivationStep>,

    /// Final result
    conclusion: Conclusion,

    /// Proof of validity
    proof: Option<Proof>,
}

/// A single step in deriving an idea
struct DerivationStep {
    /// Input ideas
    inputs: Vec<IdeaId>,

    /// Transformation applied
    transform: Transform,

    /// Output idea
    output: IdeaId,

    /// Justification
    justification: Justification,
}
```

## Example: The Idea "Chocolate Chip Cookie"

### As Token Embedding (GPT-style)
```
"chocolate chip cookie" → [0.234, -0.891, 0.445, ..., 0.123]
                          ↑
                   Learned from text corpus
                   No structure, no verification
```

### As Veritas Idea
```rust
Idea {
    id: IdeaId::new("chocolate_chip_cookie_v1"),

    concept: Concept::Semantic {
        entity: Entity {
            name: "chocolate_chip_cookie",
            category: FoodItem,
            properties: vec![
                Property::Physical("baked_good"),
                Property::Compositional("contains_flour"),
                Property::Compositional("contains_chocolate"),
                Property::Taste("sweet"),
                Property::Texture("crispy_edges_chewy_center"),
            ],
        },
        relations: vec![
            // ISA hierarchy
            Relation::IsA("baked_good"),
            Relation::IsA("dessert"),

            // Composition
            Relation::Contains("flour", Ratio(3.0)),
            Relation::Contains("butter", Ratio(1.0)),
            Relation::Contains("sugar", Ratio(1.2)),
            Relation::Contains("chocolate_chips", Ratio(0.8)),

            // Process
            Relation::RequiresProcess("mixing"),
            Relation::RequiresProcess("baking"),

            // Constraints
            Relation::Constraint("baking_temp", Range(175, 190)),
            Relation::Constraint("baking_time", Range(10, 15)),
        ],
    },

    derivation: DerivationGraph {
        axioms: vec![
            Axiom::From("culinary_knowledge_base"),
            Axiom::From("cooking_chemistry_fundamentals"),
        ],

        steps: vec![
            DerivationStep {
                inputs: vec![
                    IdeaId::new("baking_ratios"),
                    IdeaId::new("butter_cookies"),
                ],
                transform: Transform::Specialize("add_chocolate"),
                output: IdeaId::new("chocolate_chip_cookie_base"),
                justification: Justification::CulinaryTradition,
            },
            DerivationStep {
                inputs: vec![
                    IdeaId::new("chocolate_chip_cookie_base"),
                    IdeaId::new("maillard_reaction"),
                ],
                transform: Transform::AddConstraint("baking_temp"),
                output: IdeaId::new("chocolate_chip_cookie_v1"),
                justification: Justification::ChemicalProcess,
            },
        ],

        conclusion: Conclusion {
            statement: "A chocolate chip cookie is a baked good \
                       following specific ratios and requiring \
                       specific baking conditions",
            verified: true,
        },

        proof: Some(Proof {
            method: ProofMethod::Empirical,
            evidence: vec![
                Evidence::CulinaryText("joy_of_cooking"),
                Evidence::ChemicalAnalysis("maillard_temp_range"),
            ],
        }),
    },

    verified: true,

    dependencies: vec![
        IdeaId::new("baking_ratios"),
        IdeaId::new("butter_cookies"),
        IdeaId::new("maillard_reaction"),
        IdeaId::new("chocolate_properties"),
    ],
}
```

## Key Differences

| Aspect | Token Embedding | Veritas Idea |
|--------|----------------|--------------|
| **Structure** | Dense vector | Explicit graph |
| **Interpretable** | No | Yes |
| **Verifiable** | No | Yes |
| **Composable** | Vector addition (?) | Logical derivation |
| **Traceable** | No | Full provenance |
| **Size** | Fixed (4096 floats) | Variable (as needed) |
| **Storage** | ~16KB per concept | Compressed graph (~1-10KB) |

## Composing Ideas

### Token Prediction
```
embedding("chocolate") + embedding("chip") + embedding("cookie")
    = some vector that hopefully means "chocolate chip cookie"

Can't verify. Can't trace. Just hope.
```

### Veritas
```rust
// Compose ideas explicitly
fn compose_ideas(base: IdeaId, modifier: IdeaId) -> Idea {
    let base_idea = knowledge_base.get(base);
    let modifier_idea = knowledge_base.get(modifier);

    // Apply transformation with justification
    let composed = Idea {
        concept: apply_modifier(base_idea.concept, modifier_idea.concept),
        derivation: DerivationGraph {
            axioms: vec![base, modifier],
            steps: vec![
                DerivationStep {
                    inputs: vec![base, modifier],
                    transform: Transform::Compose,
                    output: new_id(),
                    justification: Justification::Composition,
                }
            ],
            conclusion: verified_composition(),
            proof: Some(composition_proof()),
        },
        verified: true,
        dependencies: vec![base, modifier],
    };

    // Verify composition is valid
    verify_idea(&composed)?;

    composed
}

// Example: "chocolate chip cookie recipe for 1kg"
let cookie = IdeaId::new("chocolate_chip_cookie");
let quantity = IdeaId::new("quantity_1kg");

let recipe = compose_ideas(cookie, quantity);
// recipe now contains verified scaling computation
```

## The Neural Component

**So where does neural fit?**

Neural networks are used for **TWO** specific tasks:

### 1. Understanding (Text → Idea Graph)
```rust
struct NeuralParser {
    model: TransformerModel,
}

impl NeuralParser {
    fn parse(&self, text: &str) -> IdeaGraph {
        // Neural model predicts structure
        let predicted_structure = self.model.forward(text);

        // Structure is VERIFIED before accepting
        let idea = IdeaGraph::from_structure(predicted_structure);

        // Check if idea is valid
        verify_idea(&idea)?;

        idea
    }
}
```

**Training:**
```rust
// Generate training data
let (text, correct_idea_graph) = symbolic_engine.generate();

// Neural tries to parse
let predicted_graph = neural_parser.parse(text);

// Verify against ground truth
if predicted_graph == correct_idea_graph {
    reward();
} else {
    let error = compute_graph_diff(predicted_graph, correct_idea_graph);
    backprop(error);  // Fix the parser
}
```

### 2. Explanation (Idea Graph → Text)
```rust
struct NeuralSerializer {
    model: TransformerModel,
}

impl NeuralSerializer {
    fn serialize(&self, idea: &IdeaGraph) -> String {
        // Neural generates text from structure
        let text = self.model.generate_from_structure(idea);

        // Verify text can be parsed back to same structure
        let reparsed = neural_parser.parse(&text);

        if reparsed != idea {
            // Explanation was unclear, regenerate
            return self.serialize_with_constraints(idea, "be_clearer");
        }

        text
    }
}
```

**Training:**
```rust
// Given verified idea graph
let idea = symbolic_engine.generate_idea();

// Neural generates explanation
let text = neural_serializer.serialize(&idea);

// Parse it back
let reparsed = neural_parser.parse(&text);

// Check roundtrip consistency
if reparsed == idea {
    reward();  // Clear explanation!
} else {
    let error = compute_semantic_loss(reparsed, idea);
    backprop(error);  // Make explanation clearer
}
```

## Storage in VSF

```rust
// Serialize idea to VSF
fn serialize_idea_to_vsf(idea: &Idea) -> Vec<u8> {
    let mut builder = VsfBuilder::new();

    // Store concept
    builder.add_section("concept", vec![
        ("type", VsfType::l(format!("{:?}", idea.concept))),
        ("data", serialize_concept(&idea.concept)),
    ]);

    // Store derivation graph
    builder.add_section("derivation", vec![
        ("steps", serialize_derivation(&idea.derivation)),
    ]);

    // Store dependencies
    builder.add_section("dependencies", vec![
        ("ids", serialize_ids(&idea.dependencies)),
    ]);

    // Sign with symbolic engine key
    let bytes = builder.build().unwrap();
    sign_section(bytes, "concept", &symbolic_key).unwrap()
}
```

**Benefits:**
- ✅ Compact (graph compression)
- ✅ Verified (BLAKE3 hash)
- ✅ Provenance (Ed25519 signature)
- ✅ Optimal encoding (VSF)

## Summary

**Token embeddings:**
- Opaque vectors
- No structure
- No verification
- Hope for the best

**Veritas ideas:**
- Explicit graphs
- Full structure
- Mechanically verified
- Proven correct

**Neural networks in Veritas:**
- Parse text → structure (verified)
- Serialize structure → text (verified)
- NOT used for reasoning itself

**The "idea" is the verified structure. Language is just the interface.**

That's why Veritas can't hallucinate: **Ideas are verified before they exist. Invalid ideas are rejected by the symbolic engine.**
