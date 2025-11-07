# Sudoku Solver Using Linear Algebra

A Python-based Sudoku solver that uses **linear algebra concepts** instead of traditional algorithmic approaches. This project demonstrates how constraint satisfaction problems can be formulated and solved using matrix operations, systems of linear equations, and Gaussian elimination techniques.

## 🎯 Project Overview

This project implements a Sudoku solver using linear algebra principles as specified in our Linear Algebra course proposal:
- **Matrix representation** of Sudoku constraints
- **System of linear equations** (Ax = b formulation)
- **Gaussian elimination** and row reduction techniques
- **Exact cover problem** solving with binary variables
- **Matrix-guided search** using constraint analysis

### Traditional CS Approach ❌ vs Our Approach ✅

**Traditional backtracking** uses recursive depth-first search with constraint propagation based on simple rule checking.

**Our linear algebra approach** models Sudoku as:
- **729 binary variables**: x[r,c,n] ∈ {0,1} where x[r,c,n]=1 means cell (r,c) contains number n
- **324 linear constraints**: representing row, column, cell, and box requirements
- **Exact cover matrix**: A binary matrix encoding all constraints
- **Matrix operations**: using NumPy for efficient computation
- **Gaussian elimination**: for constraint propagation and determining forced assignments
- **Matrix analysis**: Using RREF and pivot analysis to guide decisions

**Key Difference**: While traditional backtracking uses simple if-statements to check constraints, our approach formulates the entire problem as a linear system and uses matrix operations (Gaussian elimination, row reduction, constraint matrix analysis) to both propagate constraints and guide the search process.

## 🧮 Mathematical Foundation

### Variable Representation
Each Sudoku puzzle has **729 binary decision variables**:
- For each cell (r, c) and number n ∈ {1,2,...,9}
- Variable x[r,c,n] = 1 if cell (r,c) contains n, else 0
- Total: 9 rows × 9 columns × 9 numbers = 729 variables

### Constraint Matrix (A)
The constraint matrix A is **324 × 729** with four types of constraints:

1. **Cell Constraints** (81 constraints):
   - Each cell must contain exactly one number
   - ∑(n=1 to 9) x[r,c,n] = 1 for each cell (r,c)

2. **Row Constraints** (81 constraints):
   - Each row must contain each digit 1-9 exactly once
   - ∑(c=0 to 8) x[r,c,n] = 1 for each row r and number n

3. **Column Constraints** (81 constraints):
   - Each column must contain each digit 1-9 exactly once
   - ∑(r=0 to 8) x[r,c,n] = 1 for each column c and number n

4. **Box Constraints** (81 constraints):
   - Each 3×3 box must contain each digit 1-9 exactly once
   - ∑(cells in box) x[r,c,n] = 1 for each box and number n

### Exact Cover Formulation
The Sudoku puzzle becomes: **Find binary vector x such that Ax = b**
- A: 324×729 binary constraint matrix
- x: 729×1 binary solution vector (variables to solve for)
- b: 324×1 vector of ones (all constraints must be satisfied exactly once)

## 🚀 Features

### Linear Algebra Solver (`linear_algebra_solver.py`)
- ✅ **Matrix-based constraint representation**
- ✅ **Binary variable encoding** (729 variables)
- ✅ **Exact cover problem** formulation
- ✅ **Gaussian elimination** with backtracking for binary constraints
- ✅ **Minimum Remaining Values (MRV)** heuristic for efficiency
- ✅ **Constraint propagation** via matrix row reduction

### GUI Application (`sudoku_solver.py`)
- ✅ Interactive 9×9 grid for manual input
- ✅ **Generate** random puzzles with transformations
- ✅ **Solve** using linear algebra approach
- ✅ **Save/Load** puzzles from files
- ✅ Visual distinction between given clues and solver-filled cells
- ✅ Colored highlighting for better UX

## 📦 Installation

### Prerequisites
- Python 3.7+ (tested on Python 3.14)
- NumPy (for matrix operations)

### Setup
```powershell
# Clone the repository
git clone https://github.com/JanKennethGerona/Linear_Proj.git
cd Linear_Proj

# Install dependencies
pip install numpy

# Or create a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy
```

## 🎮 Usage

### GUI Mode (Default)
```powershell
python sudoku_solver.py
```
Or simply:
```powershell
python sudoku_solver.py --gui
```

**GUI Controls:**
- **Blank**: Start with an empty grid
- **Generate**: Create a random valid puzzle
- **Load**: Open a saved puzzle from `Saved/unsolved/`
- **Solve**: Solve using linear algebra (solver-filled cells appear in blue)
- **Save**: Save the current board to `Saved/solved/` or `Saved/unsolved/`

### CLI Mode
Solve a puzzle from command line:
```powershell
# Using puzzle string (81 characters, 0 or . for empty)
python sudoku_solver.py --puzzle "530070000600195000098000060800060003400803001700020006060000280000419005000080079"

# Using puzzle file
python sudoku_solver.py --file puzzle.txt
```

## 🧪 Testing

Run the verification tests:
```powershell
# Basic test
python test_solver.py

# Comprehensive validation (tests easy, medium, hard puzzles)
python verify_solver.py
```

Expected output:
```
✓ Solver returned success
✓✓ VALID SOLUTION: All constraints satisfied!
```

## 📊 How It Works (Step-by-Step)

### 1. Build Constraint Matrix
```python
A = build_constraint_matrix()  # Returns 324×729 binary matrix
```
Each column represents variable x[r,c,n], each row represents a constraint.

### 2. Encode Initial Puzzle
```python
# For given clues, fix corresponding variables to 1
# Reduce matrix by removing satisfied constraints and conflicting variables
A_reduced, b_reduced, var_mapping = apply_initial_clues(A, b, board)
```

### 3. Solve Exact Cover Problem
```python
# Use modified Gaussian elimination with backtracking
# Applies MRV heuristic: choose constraint with fewest satisfying variables
solution = gaussian_elimination_with_backtracking(A_reduced, b_reduced, var_mapping)
```

### 4. Decode Solution
```python
# Convert binary solution vector back to 9×9 Sudoku board
solved_board = decode_solution(solution)
```

## 📈 Algorithm Complexity

- **Space Complexity**: O(324 × 729) = O(236,196) for constraint matrix
- **Time Complexity**: 
  - Best case: O(n) for highly constrained puzzles (constraint propagation solves it)
  - Worst case: O(9^k) where k is number of empty cells (with MRV heuristic improvement)
  - Matrix operations: O(m × n) for row reduction steps

## 🎓 Linear Algebra Concepts Applied

This project demonstrates the following Linear Algebra concepts from our course:

1. **Matrices and Matrix Operations**
   - Binary constraint matrix representation
   - Matrix-vector multiplication
   - Sparse matrix techniques

2. **Systems of Linear Equations**
   - Formulating Sudoku as Ax = b
   - Exact cover problem as special case
   - Binary constraint satisfaction

3. **Gaussian Elimination & Row Reduction**
   - Modified for binary variables
   - Forward elimination to simplify system
   - Backtracking for constraint satisfaction

4. **Linear Transformations**
   - Variable elimination via row operations
   - Constraint propagation through matrix reduction

## 📁 Project Structure

```
Linear_Proj/
├── sudoku_solver.py           # Main GUI application
├── linear_algebra_solver.py   # Core linear algebra solver implementation
├── test_solver.py             # Basic testing script
├── verify_solver.py           # Comprehensive validation tests
├── README.md                  # This file
└── Saved/                     # Directory for saved puzzles
    ├── solved/                # Completed puzzles
    └── unsolved/              # Partial puzzles
```

## 🔍 Key Implementation Files

### `linear_algebra_solver.py`
Contains the `LinearAlgebraSudokuSolver` class with methods:
- `build_constraint_matrix()`: Constructs the 324×729 binary matrix A
- `apply_initial_clues()`: Reduces system based on given numbers
- `gaussian_elimination_with_backtracking()`: Solves the exact cover problem
- `solve_sudoku()`: Main solving pipeline

### `sudoku_solver.py`
- Imports the linear algebra solver
- Provides GUI using tkinter
- Handles puzzle I/O, randomization, and visualization
- The `solve()` function now calls `solve_with_linear_algebra()`

## 🎨 Visual Features

- **Alternating box colors**: Light blue and white backgrounds for 3×3 boxes
- **Solver highlighting**: Blue text for solver-filled numbers
- **Black text**: Original clues or manually entered numbers
- **Button colors**: 
  - Blue "Solve" button (matches solver color)
  - Light green "Save" button
  - Standard colors for other controls

## 🐛 Known Limitations

- For extremely sparse puzzles (very few clues), the backtracking component may take longer
- NumPy is required (not pure Python standard library)
- Binary constraint satisfaction still requires some backtracking (not purely Gaussian elimination due to integer constraints)

## 🤝 Contributing

This is an academic project for Linear Algebra course. Suggestions and improvements are welcome!

## 📚 References

- Project proposal: See attached course documentation
- Linear algebra formulation based on exact cover problem
- Constraint satisfaction via matrix operations
- Algorithm X and Dancing Links (Knuth) adapted for matrix representation

## 👥 Authors

- **JanKennethGerona** (GitHub: @JanKennethGerona)
- Course: Linear Algebra
- Date: November 2025

## 📝 License

This project is created for educational purposes as part of a Linear Algebra course project.

---

**Note**: This implementation prioritizes demonstrating linear algebra concepts over raw performance. A pure backtracking algorithm may be faster in practice, but wouldn't demonstrate the mathematical modeling and matrix operations that are the focus of this project.
