# Mathematical Framework

Complete mathematical reference for the EigenDialectos framework.

## 1. Notation

| Symbol | Domain | Meaning |
|--------|--------|---------|
| d | N | Embedding dimensionality (300 for subword/word, 768 for sentence) |
| V | N | Vocabulary size |
| m | N | Number of dialect varieties (m = 8) |
| E_i | R^{d x V} | Embedding matrix for dialect i |
| E_ref | R^{d x V} | Reference dialect embedding (Peninsular Standard) |
| W_i | R^{d x d} | Transformation matrix: reference -> dialect i |
| lambda | R+ | Regularisation strength (default 0.01) |
| alpha | R | Dialectal intensity parameter |
| P | C^{d x d} | Eigenvector matrix |
| Lambda | C^{d x d} | Diagonal eigenvalue matrix |
| T | R^{d x d x m} | Multi-dialect tensor |
| H_i | R+ | Spectral entropy of dialect i |
| beta_k | [0,1] | Mixing weight for dialect k (sum = 1) |

## 2. Transformation Matrix Computation

### 2.1 Ridge Regression (default: `lstsq`)

Closed-form solution to the regularised least-squares problem:

```
min_W  ||W E_ref - E_i||_F^2 + lambda ||W||_F^2
```

Solution:

```
W_i = E_i E_ref^T (E_ref E_ref^T + lambda I)^{-1}
```

where `E_ref E_ref^T` is the d x d Gram matrix, and `lambda I` is Tikhonov regularisation.

### 2.2 Orthogonal Procrustes

Constrains W to the orthogonal group O(d):

```
W* = argmin_{W^T W = I}  ||W E_ref - E_i||_F
```

Solution via SVD of the cross-covariance:

```
E_i E_ref^T = U Sigma V^T
W* = U V^T
```

Implementation uses `scipy.linalg.orthogonal_procrustes(E_ref^T, E_i^T)`, returning R such that `E_ref^T R ~ E_i^T`, hence `W = R^T`.

### 2.3 Nuclear-Norm Regularisation

Encourages low-rank solutions via singular value soft-thresholding:

1. Compute unregularised least-squares W_ls
2. SVD: `W_ls = U Sigma V^T`
3. Soft-threshold: `sigma_j <- max(sigma_j - lambda, 0)`
4. Reconstruct: `W = U diag(sigma_thresh) V^T`

## 3. Eigendecomposition

### 3.1 Standard Eigendecomposition

```
W_i = P_i Lambda_i P_i^{-1}
```

where:
- `Lambda_i = diag(lambda_1, ..., lambda_d)` are eigenvalues (generally complex)
- `P_i` contains eigenvectors as columns
- `P_i^{-1}` is computed via `np.linalg.inv` (pseudo-inverse fallback for singular P)

Reconstruction quality check:

```
relative_error = ||P Lambda P^{-1} - W||_F / ||W||_F < 10^{-6}
```

Complex eigenvalues of real W always appear in conjugate pairs.

### 3.2 Singular Value Decomposition

```
W = U Sigma V^T
```

- U, V orthogonal; Sigma diagonal with non-negative entries
- Used for rank analysis and nuclear-norm regularisation

## 4. DIAL: Dialectal Interpolation via Algebraic Linearisation

### 4.1 Core Formula

```
W_i(alpha) = P_i Lambda_i^alpha P_i^{-1}
```

where `Lambda^alpha = diag(lambda_1^alpha, ..., lambda_d^alpha)`.

### 4.2 Complex Power

For `lambda_j = |lambda_j| exp(i theta_j)`:

```
lambda_j^alpha = |lambda_j|^alpha exp(i alpha theta_j)
```

Well-defined for all real alpha when `|lambda_j| > 0`. Zero eigenvalues map to zero for alpha > 0.

### 4.3 Intensity Semantics

| alpha | Interpretation | Matrix |
|-------|---------------|--------|
| 0.0 | Neutral (identity) | W(0) = I |
| 0.0 -- 0.5 | Subtle dialectal colouring | Between I and W |
| 0.5 -- 1.0 | Increasing dialectal strength | Approaching W |
| 1.0 | Full dialect (original W recovered) | W(1) = W |
| 1.0 -- 1.5 | Hyperdialect (exaggerated features) | Beyond W |
| -1.0 | Inverse dialect | W(-1) = W^{-1} |

### 4.4 Properties

- **Continuity:** W(alpha) is continuous in alpha
- **Interpolation:** W(0) = I, W(1) = W
- **Semigroup:** W(alpha) @ W(beta) = W(alpha + beta) from the same decomposition
- **Invertibility:** W(-alpha) = W(alpha)^{-1} when all eigenvalues are non-zero

### 4.5 Embedding Transform

For embedding vector or batch e_in:

```
e_out = W_i(alpha) @ e_in = P_i Lambda_i^alpha P_i^{-1} @ e_in
```

### 4.6 Feasibility Constraints

Before applying DIAL at a given alpha:
- `max(|lambda_j^alpha|) < 100` (prevents explosion)
- `min(|lambda_j^alpha|) > 0.001` (prevents collapse)
- `max / min < 1000` (condition number proxy)

## 5. Spectral Entropy

### 5.1 Definition

```
H_i = -sum_j p_j ln(p_j)
```

where eigenvalue magnitudes are normalised:

```
p_j = |lambda_j| / sum_k |lambda_k|
```

### 5.2 Properties

- H = 0 when all energy on one eigenvalue (rank-1 transform)
- H = ln(d) when all magnitudes equal (maximum complexity)
- Higher entropy implies more independent axes of dialectal variation

### 5.3 Logarithm Bases

- Natural (default): `H = -sum p_j ln(p_j)`
- Base 2: `H = -sum p_j log_2(p_j)` (bits)
- Base 10: `H = -sum p_j log_10(p_j)`

## 6. Distance Metrics

### 6.1 Frobenius Distance

```
d_F(W_a, W_b) = ||W_a - W_b||_F
```

### 6.2 Spectral Distance (Earth Mover's Distance)

Wasserstein-1 distance between normalised eigenvalue magnitude distributions:

```
d_S(spec_a, spec_b) = EMD(p_a, p_b)
```

via `scipy.stats.wasserstein_distance`.

### 6.3 Subspace Distance

Compares top-k eigensubspaces via projection matrix difference:

```
d_sub(P_a, P_b) = ||Q_a Q_a^H - Q_b Q_b^H||_F
```

where Q_a, Q_b are QR-orthonormalised from the first k eigenvector columns.

### 6.4 Entropy Distance

```
d_H(H_a, H_b) = |H_a - H_b|
```

### 6.5 Combined Distance

```
d_combined = (w_F d_F + w_S d_S + w_H d_H) / (w_F + w_S + w_H)
```

Default: w_F = w_S = w_H = 1.

## 7. Dialect Mixing

### 7.1 Linear Mixing

```
W_mix = sum_k beta_k W_k       (sum_k beta_k = 1)
```

### 7.2 Log-Euclidean Mixing

Respects GL(d) Lie group structure:

```
W_mix = expm( sum_k beta_k logm(W_k) )
```

Reduces to geodesic interpolation for two matrices with beta = 0.5.

### 7.3 Eigenvalue-Level Mixing

Using shared eigenvector basis (from first decomposition):

```
lambda_mix = prod_k |lambda_k|^{beta_k} * exp(i sum_k beta_k theta_k)
```

Equivalently in the log domain:

```
log(lambda_mix) = sum_k beta_k log(lambda_k)
```

This is a weighted geometric mean of eigenvalues.

## 8. Tensor Representation

### 8.1 Construction

```
T in R^{d x d x m},    T[:, :, k] = W_k
```

### 8.2 Tucker Decomposition

```
T ~ G x_1 A x_2 B x_3 C
```

- Core tensor G of shape (r1, r2, r3)
- Factor matrices A (d x r1), B (d x r2), C (m x r3)
- Mode-3 factor C reveals dialect groupings in reduced r3-dimensional space

### 8.3 CP Decomposition

```
T ~ sum_{r=1}^{R} w_r (a_r (x) b_r (x) c_r)
```

Each rank-1 component captures a distinct mode of dialectal variation. Core Consistency Diagnostic (CORCONDIA) evaluates appropriate CP rank.

## 9. Algebraic Framework

### 9.1 Operations

| Operation | Formula | Interpretation |
|-----------|---------|---------------|
| Composition | `W_1 @ W_2` | Apply d2 first, then d1 |
| Inversion | `P diag(1/lambda) P^{-1}` | Undo a dialect transform |
| Interpolation | `expm(alpha logm(W))` | Continuous intensity via matrix log |
| Projection | `P_V W P_V` | Restrict to eigensubspace |

### 9.2 Approximate Group Testing

- **Closure:** `W_i @ W_j ~ W_k` for some k
- **Associativity:** `(W_i @ W_j) @ W_k ~ W_i @ (W_j @ W_k)` (always holds numerically)
- **Identity:** exists W_e ~ I
- **Inverse:** `W_i^{-1} @ W_i ~ I`

All checks use Frobenius-norm tolerance.

## 10. Validation Metrics

| Metric | Formula | Application |
|--------|---------|-------------|
| BLEU | Geometric mean of clipped n-gram precisions x BP | Generation quality |
| chrF | Char n-gram F-score (n=1..6, beta=2) | Character-level accuracy |
| Perplexity ratio | PP_target / PP_baseline | Dialect specificity |
| Frobenius error | `||W_true - W_pred||_F / ||W_true||_F` | Matrix reconstruction |
| KL divergence | `sum p_a log(p_a / p_b)` on normalised spectra | Spectral comparison |
| Krippendorff's alpha | Coincidence-matrix agreement | Human evaluation |
