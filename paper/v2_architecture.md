# EigenDialectos v2: The Algebraic Compiler

## Beyond LLMs — A Pure Mathematical Engine for Dialectal Transformation

---

## Part I: Why No LLM Is Needed (And Why That's Better)

### The core realization

An LLM is a *statistical text generator*. What we're building is something fundamentally different: a **dialectal compiler**. A compiler doesn't "imagine" code in assembly — it *transforms* a high-level representation into a low-level one through deterministic, algebraic rules. Our system does the same: it takes text in variety A and *compiles* it to variety B through spectral operations in embedding space.

This is not just cheaper — it's **scientifically superior**:
- Every transformation is **traceable**: you can point to exactly which eigenvalue/eigenvector caused each change.
- The system is **invertible**: compile from canario → neutro → rioplatense and back.
- There are **no hallucinations**: the output is algebraically determined by the input.
- It's **infinitely controllable**: α = 0.73 means exactly 73% dialectal intensity, not "sort of regional."

---

## Part II: The Compiler Architecture

### Stage 1 — Hierarchical Decomposition of Input Text

Don't treat text as a flat sequence. Language has structure, and dialectal variation operates at every level simultaneously. We decompose the input into a **multi-resolution representation**:

```
┌─────────────────────────────────────────────────┐
│  Level 5: DISCOURSE                             │
│  Paragraph structure, narrative rhythm,          │
│  topic-comment patterns                         │
│                                                 │
│  ┌─────────────────────────────────────────┐    │
│  │  Level 4: SENTENCE                      │    │
│  │  Clause ordering, subordination style,   │    │
│  │  interrogative structure                 │    │
│  │                                         │    │
│  │  ┌─────────────────────────────────┐    │    │
│  │  │  Level 3: PHRASE / COLLOCATION  │    │    │
│  │  │  Idioms, fixed expressions,      │    │    │
│  │  │  verb+clitic patterns            │    │    │
│  │  │                                 │    │    │
│  │  │  ┌─────────────────────────┐    │    │    │
│  │  │  │  Level 2: WORD / LEMMA  │    │    │    │
│  │  │  │  Lexical choice,        │    │    │    │
│  │  │  │  diminutive forms       │    │    │    │
│  │  │  │                         │    │    │    │
│  │  │  │  ┌─────────────────┐    │    │    │    │
│  │  │  │  │ Level 1: MORPH  │    │    │    │    │
│  │  │  │  │ Suffixes, verb  │    │    │    │    │
│  │  │  │  │ conjugation,    │    │    │    │    │
│  │  │  │  │ pseudo-phonetic │    │    │    │    │
│  │  │  │  └─────────────────┘    │    │    │    │
│  │  │  └─────────────────────────┘    │    │    │
│  │  └─────────────────────────────────┘    │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

Each level has its **own transformation matrix** W_i^(ℓ) and therefore its **own eigenspectrum**. This is critical: dialectal variation at the morphological level is *independent* from variation at the discourse level. A single matrix can't capture this — you need a **spectral stack**.

### Stage 2 — The Spectral Stack Transform

For a text t in variety A, the full transformation to variety B is:

C_{A→B}(t) = ∏_{ℓ=1}^{L} W_{A→B}^(ℓ)(α_ℓ) ∘ π_ℓ(t)

where:
- π_ℓ(t) extracts the level-ℓ representation of t
- W_{A→B}^(ℓ)(α_ℓ) is the transformation at level ℓ with intensity α_ℓ
- The product ∏ is a **composition**, not matrix multiplication — each level feeds into the next

The beauty: **you can set different α per level**. Want canario vocabulary (α₂ = 1.0) but standard syntax (α₄ = 0.0)? Done. Want to exaggerate only the morphological features (α₁ = 1.5)? Done. This gives you a **mixing board** with one slider per linguistic level.

### Stage 3 — Token Replacement via Spectral Nearest Neighbor

This is where the actual text transformation happens — no generation, just algebraic lookup.

**Algorithm: Spectral Dialectal Compilation (SDC)**

```
INPUT:  Text t in variety A (or neutral)
OUTPUT: Text t' in variety B at intensity α

1.  Parse t into multi-level representation {π_ℓ(t)}

2.  For each level ℓ, from morpheme up to discourse:

    a. Embed π_ℓ(t) into variety-A embedding space: x_A = E_A^(ℓ)(π_ℓ(t))

    b. Apply spectral transform:
       x_B = P_B^(ℓ) · Λ_B^(α_ℓ) · (P_B^(ℓ))⁻¹ · x_A

    c. Find k-nearest neighbors in variety-B embedding space:
       candidates = kNN(x_B, E_B^(ℓ), k=10)

    d. Score candidates by:
       - Cosine similarity to x_B                    (semantic fidelity)
       - Frequency in variety-B corpus                (naturalness)
       - Grammatical compatibility with context       (coherence)
       - Eigenvalue-weighted distance along each
         eigenvector                                  (dialectal authenticity)

    e. Select best candidate → insert into output

3.  Apply post-processing:
    - Morphological agreement (gender, number, verb forms)
    - Clitic repositioning if needed
    - Orthographic adjustments (seseo, aspiration markers)

4.  Return t'
```

**Why this works without an LLM**: We're not *generating* — we're doing **lookup in a transformed space**. The eigendecomposition tells us *how* to transform, the embedding space tells us *what exists* in the target variety, and the nearest neighbor search finds the actual realization. It's deterministic algebra + search, not probabilistic generation.

### Stage 4 — The Residual Network (for what algebra can't capture)

Pure linear algebra will handle ~70-80% of dialectal transformation. The remaining 20-30% involves:
- Context-dependent lexical choices (a word maps to different words depending on surrounding context)
- Idiomatic expressions that don't decompose compositionally
- Pragmatic particles whose placement depends on discourse state

For this, instead of an LLM, we use a **small, specialized residual network** φ:

t' = SDC(t) + φ(SDC(t), context)

This is NOT an LLM. It's a lightweight correction network (think: a few transformer layers, 10-50M parameters) trained specifically to fix the errors that pure algebraic transformation makes. It's a **polisher**, not a **generator**.

---

## Part III: Radically Rethinking the Embedding and Eigenvalue Pipeline

### Innovation 1: Dialect-Contrastive Pre-Training from Scratch

Don't use off-the-shelf embeddings. They're trained on mixed data and wash out dialectal signal. Instead, train **dialect-aware embeddings** with a custom objective.

**The Dialectal Contrastive Loss (DCL)**:

For a word w that appears in both variety A and B, with contexts c_A and c_B:

L_DCL = -log σ(e_w^A · e_{c_A}^A) - log σ(-e_w^A · e_{c_B}^B) + λ ‖e_w^A - e_w^B‖₂² · 𝟙[w ∉ R]

Breaking this down:
- **Term 1**: Words should be close to their same-variety contexts (standard Word2Vec).
- **Term 2**: Words should be *far* from contexts of other varieties (forces the space to encode dialectal information).
- **Term 3**: Words that are NOT regionalisms (w ∉ R) should have similar embeddings across varieties (anchors the shared core).

The genius of Term 3: it forces the embedding space to put ALL dialectal variation into specific dimensions, while keeping shared semantics aligned. This makes the subsequent eigendecomposition *much* cleaner because dialectal signal isn't smeared across all dimensions.

### Innovation 2: Multi-Granularity Eigendecomposition

Instead of one eigendecomposition per variety, perform **nested decompositions** at increasing granularity:

**Level 0 — The Macro-Eigenvalues: Pan-Hispanic Variation**

Pool all varieties and compute a single SVD of the full transformation tensor. The top eigenvectors here capture the **universal axes of Spanish variation** — dimensions along which *all* varieties diverge from the mean.

**Level 1 — Zonal Eigenvalues: Family-Level Variation**

Group varieties into families (Caribbean, Southern Peninsular, Rioplatense, etc.) and decompose *within* each group.

**Level 2 — Dialectal Eigenvalues: Variety-Specific Variation**

The standard per-variety decomposition. But now, having already extracted macro and zonal components, this level captures only the **truly unique** signature of each variety.

**Level 3 — Idiolectal Eigenvalues: Individual Speaker Variation**

If we have enough data per speaker, decompose further to find the eigenstructure of **individual speakers within a variety**.

**The Hierarchical Eigenspectrum**:

Λ_total^(i) = Λ_macro ⊕ Λ_zonal(i) ⊕ Λ_dialect(i) ⊕ Λ_idiolect(i)

### Innovation 3: The Eigenvalue Field — Continuous, Not Discrete

Define a function:

λ_k(x, y, s) : R² × S → R

where (x, y) are geographic coordinates and s is a vector of sociolinguistic variables (age, education, urbanity, register). The k-th eigenvalue is now a **smooth function** that varies continuously across geography and society.

**How to estimate this**:
1. Compute eigenvalues for geolocated speakers/texts.
2. Fit a Gaussian Process (GP) over the geographic/social coordinates.
3. The GP gives you a smooth interpolation *and* uncertainty estimates.

**What this enables**:
- Generate text for *any point on the map*, not just discrete labeled varieties.
- Model **dialect continua** and **isoglosses** as contour lines of eigenvalue fields.
- Predict how dialect features **diffuse** over space:

∂λ_k/∂t = D_k ∇²λ_k + f_k(x, y, t)

### Innovation 4: Riemannian Geometry of Dialect Space

Define a **dialect manifold** M where each point represents a variety of Spanish. The **metric tensor** at each point is derived from the eigenstructure:

g_jk(i) = Σ_ℓ σ_ℓ^(i) · v_ℓ^(i)_j · v_ℓ^(i)_k

**Geodesics** on this manifold represent the **most natural path** of dialectal change between two varieties.

**Curvature** tells us about **dialectal instability**: regions of high curvature are dialectal boundaries. The **Ricci curvature** measures how fast nearby dialects diverge — high = unstable dialect zone (isoglosses).

### Innovation 5: Topological Data Analysis of the Eigenspectrum

Apply **persistent homology** to the set of eigenspectra across all varieties:
- **H₀** (Connected components): How many truly distinct dialect families exist?
- **H₁** (Loops): Circular contact relationships (impossible in tree models)
- **H₂** (Voids): "Impossible dialects" — linguistically incoherent feature combinations

### Innovation 6: Lie Group Structure of Dialectal Transformations

W_i = exp(A_i), where A_i is the **dialect generator** in the Lie algebra.

Correct interpolation: A_mix = β·A_canario + (1-β)·A_andaluz

The commutator [A_i, A_j] measures how much two dialectal transformations **interfere**. If [A_i, A_j] ≠ 0, applying voseo before ceceo gives a different result than the reverse — linguistically profound.

### Innovation 7: Information-Theoretic Eigenvalue Decomposition

Fisher Information Matrix eigenvalues: which parameters of variation carry the most signal for identifying a variety. Complementary to algebraic eigenvalues.

### Innovation 8: Dynamic Eigenvalues — Context-Dependent Spectra

W_i(c) = W_i^(base) + Σ_k α_k(c) · ΔW_i^(k)

Eigenvalues vary with context: code-switching spikes, register dampening, topic triggering.

---

## Part IV: The Full Compiler Pipeline — Step by Step

### Detailed Example

```
INPUT: "El autobús llega a las tres y media, ¿no?"
       [Neutral Spanish, target: Canario, α = 0.9]

STEP 1: MULTI-LEVEL PARSING
├── L1 (Morpheme):  [El] [auto-bús] [lleg-a] [a] [las] [tres] [y] [medi-a] [¿no?]
├── L2 (Word):      [El] [autobús] [llega] [a] [las] [tres] [y] [media] [¿no?]
├── L3 (Phrase):    [El autobús] [llega] [a las tres y media] [¿no?]
├── L4 (Clause):    [El autobús llega a las tres y media] [¿no?]
└── L5 (Discourse): [informative statement + confirmation seeking]

STEP 2: EMBED EACH LEVEL
├── L2 embeddings:  autobús → e_neutral("autobús") = [0.23, -0.41, ...]

STEP 3: APPLY SPECTRAL TRANSFORM (per level)
├── L2: W_canario^(2)(0.9) → "autobús" → kNN → "guagua" (score: 0.94)
├── L3: W_canario^(3)(0.9) → "¿no?" → kNN → "¿verdad?" (score: 0.67)

STEP 4: RECONSTRUCTION + AGREEMENT
├── "La guagua llega a las tres y media, ¿verdad?"

STEP 5: RESIDUAL CORRECTION → final output
```

### The Eigenvalue Trace: Full Interpretability

```
CHANGE LOG:
─────────────────────────────────────────────────
Change 1: "autobús" → "guagua"
  Level:       L2 (Lexical)
  Eigenvector: v_3^(canario, L2) — "transportation lexicon axis"
  Eigenvalue:  λ_3 = 2.41
  α applied:   0.9
  Confidence:  0.94

Change 2: "¿no?" → "¿verdad?"
  Level:       L3 (Phrasal/Pragmatic)
  Eigenvector: v_1^(canario, L3) — "tag question preference axis"
  Eigenvalue:  λ_1 = 1.18
  α applied:   0.9
  Confidence:  0.67
  NOTE:        At α < 0.6, this change would not trigger.
─────────────────────────────────────────────────
```

---

## Part V: Novel Experiments Enabled by This Framework

### Experiment A: "The Dialectal Genome"
Eigenvalue signature across all levels → phylogenetic tree → compare with historical/genetic/geographic trees.

### Experiment B: "Dialect Phase Transitions"
Eigenvalue field discontinuities = mathematical isoglosses. Potts model from statistical physics.

### Experiment C: "Eigenvalue Archaeology"
Historical texts → temporal eigenvalue trajectory → gradual vs. punctuated change.

### Experiment D: "Synthetic Dialect Engineering"
Invent linguistically coherent new dialects by choosing eigenvalues in feasibility region.

### Experiment E: "Bidialectal Code-Switching Dynamics"
Model bidialectal speakers as oscillating eigenvalue systems.

### Experiment F: "The Eigenvalue Microscope"
Surgically edit single features by manipulating specific eigenvalues.

### Experiment G: "Cross-Linguistic Spectral Transfer"
Test if eigenvectors generalize across languages → universal grammar of dialectal variation.

---

## Part VI: Advanced Training Pipeline

1. **Corpus with metadata graph** (GPS, demographics, register, medium, date)
2. **Adversarial dialect embedding training** (concentrate dialectal info in top-k PCs)
3. **Iterative eigenvalue refinement** (bootstrapping: eigen→reweight→retrain→re-eigen until convergence)
4. **Validation at every step** (known regionalisms, non-regionalisms, expert judgment)

---

## Part VII: Mathematical Arsenal Summary

| Technique | What it captures |
|---|---|
| Eigendecomposition | Axes and intensity of dialectal variation |
| SVD | Robust version for non-square/non-symmetric transforms |
| Tensor decomposition | Shared structure across multiple varieties |
| Gaussian Processes | Continuous eigenvalue field over geography/society |
| Riemannian geometry | Natural distances and paths between dialects |
| Persistent homology | Topological features (clusters, loops, voids) |
| Lie groups/algebras | Correct interpolation and composition |
| Fisher Information Matrix | Most diagnostic features |
| Dynamical systems | Time-varying eigenvalues |
| Diffusion equations | Spatial propagation of features |
| Potts model | Phase transitions at dialect boundaries |

---

## Part VIII: Target Venues

ACL, EMNLP, NAACL (computational linguistics); NeurIPS (math/ML emphasis); Language (linguistic theory); Digital Scholarship in the Humanities; Journal of Quantitative Linguistics.

---

*EigenDialectos v2 Architecture Document — 2026-04-04*
