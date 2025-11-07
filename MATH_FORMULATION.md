# Mathematical Formulation of Sudoku Using Linear Algebra

This document provides a detailed mathematical explanation of how we formulate and solve Sudoku puzzles using linear algebra concepts.

## 1. Problem Formulation

### 1.1 Decision Variables

We introduce **729 binary decision variables**:

$$x_{r,c,n} \in \{0, 1\} \quad \text{for } r,c \in \{0,1,...,8\}, n \in \{1,2,...,9\}$$

Where:
- $x_{r,c,n} = 1$ if cell at row $r$, column $c$ contains number $n$
- $x_{r,c,n} = 0$ otherwise

**Variable Indexing**: We map 3D index $(r,c,n)$ to 1D index $i$:
$$i = 81r + 9c + (n-1) \quad \text{where } i \in \{0, 1, ..., 728\}$$

### 1.2 Constraint Formulation

Sudoku rules translate to **324 linear constraints** (all equations equal 1):

#### Type 1: Cell Constraints (81 constraints)
Each cell contains exactly one number:
$$\sum_{n=1}^{9} x_{r,c,n} = 1 \quad \text{for all } r,c \in \{0,...,8\}$$

**Matrix representation**: Constraint index $C_{\text{cell}}(r,c) = 9r + c$

#### Type 2: Row Constraints (81 constraints)  
Each row contains each digit exactly once:
$$\sum_{c=0}^{8} x_{r,c,n} = 1 \quad \text{for all } r \in \{0,...,8\}, n \in \{1,...,9\}$$

**Matrix representation**: Constraint index $C_{\text{row}}(r,n) = 81 + 9r + (n-1)$

#### Type 3: Column Constraints (81 constraints)
Each column contains each digit exactly once:
$$\sum_{r=0}^{8} x_{r,c,n} = 1 \quad \text{for all } c \in \{0,...,8\}, n \in \{1,...,9\}$$

**Matrix representation**: Constraint index $C_{\text{col}}(c,n) = 162 + 9c + (n-1)$

#### Type 4: Box Constraints (81 constraints)
Each 3×3 box contains each digit exactly once:
$$\sum_{r \in B_i} \sum_{c \in B_i} x_{r,c,n} = 1 \quad \text{for all boxes } i \in \{0,...,8\}, n \in \{1,...,9\}$$

Where $B_i$ represents the set of cells in box $i$:
- Box index: $b = 3\lfloor r/3 \rfloor + \lfloor c/3 \rfloor$

**Matrix representation**: Constraint index $C_{\text{box}}(b,n) = 243 + 9b + (n-1)$

## 2. Matrix Formulation (Ax = b)

### 2.1 Constraint Matrix A

**Dimensions**: $A \in \{0,1\}^{324 \times 729}$

Matrix entry $A_{ij}$ is defined as:
$$A_{ij} = 1 \text{ if variable } j \text{ participates in constraint } i$$
$$A_{ij} = 0 \text{ otherwise}$$

Each variable $x_{r,c,n}$ appears in **exactly 4 constraints**:
1. Cell constraint for position $(r,c)$
2. Row constraint for row $r$ and number $n$
3. Column constraint for column $c$ and number $n$  
4. Box constraint for box containing $(r,c)$ and number $n$

**Sparsity**: Each column has exactly 4 ones. Total non-zero entries: $729 \times 4 = 2,916$

### 2.2 Right-Hand Side Vector b

$$b = \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix} \in \mathbb{R}^{324}$$

All constraints must be satisfied exactly once (exact cover problem).

### 2.3 The Linear System

$$Ax = b$$
$$A \in \{0,1\}^{324 \times 729}, \quad x \in \{0,1\}^{729}, \quad b \in \{1\}^{324}$$

**Goal**: Find binary vector $x$ such that $Ax = b$

## 3. Initial Clue Processing

### 3.1 Fixing Variables

For each given clue at position $(r_0, c_0)$ with value $n_0$:

1. **Fix variable**: $x_{r_0,c_0,n_0} = 1$

2. **Eliminate conflicting variables**: Set to 0 all variables that would violate constraints:
   - $x_{r_0,c_0,k} = 0$ for all $k \neq n_0$ (same cell)
   - $x_{r_0,j,n_0} = 0$ for all $j \neq c_0$ (same row)
   - $x_{i,c_0,n_0} = 0$ for all $i \neq r_0$ (same column)
   - $x_{i,j,n_0} = 0$ for all $(i,j)$ in same box as $(r_0,c_0)$ but $(i,j) \neq (r_0,c_0)$

3. **Remove satisfied constraints**: All 4 constraints associated with $x_{r_0,c_0,n_0}$

### 3.2 Matrix Reduction

After processing all clues:
- Remove rows (constraints) that are satisfied
- Remove columns (variables) that are fixed or eliminated
- Result: $A' \in \{0,1\}^{m' \times n'}$ where $m' < 324$, $n' < 729$

## 4. Solution Method

### 4.1 Modified Gaussian Elimination with Backtracking

Since we need **binary solutions** (not general real solutions), we use a hybrid approach:

**Algorithm**:
```
function solve_exact_cover(A, b):
    if A is empty:
        return success (all constraints satisfied)
    
    // Choose constraint with minimum options (MRV heuristic)
    c = constraint with fewest satisfying variables
    
    if no variables satisfy c:
        return failure (backtrack)
    
    for each variable v that satisfies constraint c:
        // Make a choice
        x[v] = 1
        
        // Propagate constraints (row reduction)
        A' = remove all constraints satisfied by v
        A' = remove all conflicting variables
        
        // Recursively solve reduced system
        if solve_exact_cover(A', b'):
            return success
        
        // Backtrack
        x[v] = 0
    
    return failure
```

### 4.2 Constraint Propagation via Row Operations

When we set $x_j = 1$:

1. **Row elimination**: Remove row $i$ if $A_{ij} = 1$ (constraint satisfied)
2. **Column elimination**: Remove column $k$ if:
   - $\exists i$ such that $A_{ij} = 1$ and $A_{ik} = 1$ (conflicting variable)

This is analogous to **Gaussian elimination** but adapted for the exact cover problem structure.

### 4.3 Minimum Remaining Values (MRV) Heuristic

Choose constraint with minimum row sum:
$$c^* = \arg\min_{i} \sum_{j=1}^{n'} A'_{ij}$$

This reduces the branching factor in backtracking.

## 5. Solution Decoding

Once we have solution vector $x^* \in \{0,1\}^{729}$:

**For each $i \in \{0, ..., 728\}$ where $x^*_i = 1$:**

1. Decode variable index to $(r,c,n)$:
   $$n = (i \bmod 9) + 1$$
   $$c = \lfloor i/9 \rfloor \bmod 9$$  
   $$r = \lfloor i/81 \rfloor$$

2. Set board: $\text{board}[r][c] = n$

**Result**: 9×9 Sudoku board with all cells filled

## 6. Complexity Analysis

### 6.1 Space Complexity

- Constraint matrix: $O(324 \times 729) = O(236,196)$ entries
- With sparsity: $O(2,916)$ non-zero entries (can use sparse matrix)
- Solution vector: $O(729)$

**Total**: $O(n^6)$ where $n=9$ (board size)

### 6.2 Time Complexity

- **Matrix construction**: $O(n^6) = O(729 \times 4)$
- **Initial reduction**: $O(k \times n^4)$ where $k$ = number of clues
- **Backtracking search**: 
  - Best case: $O(n^4)$ (constraint propagation solves it)
  - Worst case: $O(n^{e})$ where $e$ = empty cells
  - MRV heuristic significantly reduces branching factor in practice

## 7. Linear Algebra Concepts Demonstrated

### 7.1 Matrix Operations
- Binary matrix construction
- Matrix-vector multiplication check: $Ax = b$
- Row and column operations

### 7.2 Systems of Linear Equations  
- Formulating as $Ax = b$
- Exact cover as special case
- Constraint satisfaction

### 7.3 Row Reduction
- Eliminating satisfied constraints (row operations)
- Variable elimination (column operations)  
- Reduced row echelon form (partial)

### 7.4 Dimensionality Reduction
- Reducing system from 324 constraints to fewer
- Variable space reduction from 729 to fewer
- Constraint propagation via matrix operations

## 8. Advantages of This Approach

1. **Mathematical Elegance**: Clear formulation using standard linear algebra
2. **Constraint Propagation**: Automatic through matrix row reduction  
3. **Generalizability**: Easily extends to variants (16×16 Sudoku, irregular boxes)
4. **Educational Value**: Demonstrates multiple linear algebra concepts
5. **Matrix Tooling**: Can leverage optimized linear algebra libraries (NumPy)

## 9. Implementation Notes

### 9.1 NumPy Integration
```python
import numpy as np

# Constraint matrix (sparse in practice)
A = np.zeros((324, 729), dtype=np.int32)

# Solution vector
x = np.zeros(729, dtype=np.int32)

# Constraint satisfaction check
satisfied = np.all(A @ x == b)
```

### 9.2 Indexing Functions
```python
def var_index(r, c, n):
    """Map (r,c,n) to variable index i"""
    return r * 81 + c * 9 + (n - 1)

def var_to_cell(i):
    """Map variable index i to (r,c,n)"""
    n = (i % 9) + 1
    c = (i // 9) % 9
    r = i // 81
    return r, c, n
```

## 10. Verification

A solution is valid if:

1. **All variables binary**: $x_i \in \{0,1\}$ for all $i$
2. **Constraints satisfied**: $Ax = b$ (all 324 equations hold)
3. **Exactness**: Each constraint satisfied exactly once (not more, not less)

**Matrix verification**:
```python
def verify_solution(A, x, b):
    return np.all(A @ x == b) and np.all((x == 0) | (x == 1))
```

## References

1. Knuth, D. E. (2000). Dancing Links. arXiv:cs/0011047
2. Exact Cover Problem - combinatorial optimization
3. Linear algebra formulation of constraint satisfaction problems
4. Algorithm X and its variants

---

**This formulation transforms Sudoku from a logic puzzle into a system of linear equations, demonstrating the power and versatility of linear algebra in computer science applications.**
