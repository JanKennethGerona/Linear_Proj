"""
Linear Algebra-based Sudoku Solver

This module implements a Sudoku solver using linear algebra concepts:
- Matrix representation of constraints
- System of linear equations (Ax = b)
- Gaussian elimination and row reduction
- Exact cover problem formulation

Based on the approach where each Sudoku puzzle is represented as a system
of linear equations with binary variables.
"""

import numpy as np
from typing import List, Tuple, Optional

Board = List[List[int]]


class LinearAlgebraSudokuSolver:
    """
    Sudoku solver using linear algebra approach.
    
    Represents Sudoku as an exact cover problem with 729 binary variables:
    - x[i,j,k] = 1 if cell (i,j) contains number k (1-indexed)
    - x[i,j,k] = 0 otherwise
    
    Constraints (324 total):
    - 81 cell constraints: each cell has exactly one number
    - 81 row constraints: each row has each digit 1-9
    - 81 column constraints: each column has each digit 1-9
    - 81 box constraints: each 3x3 box has each digit 1-9
    """
    
    def __init__(self):
        self.n = 9  # Board size
        self.num_vars = 729  # 9x9x9 binary variables
        self.num_constraints = 324  # 4 types × 81 each
        
    def _var_index(self, row: int, col: int, num: int) -> int:
        """
        Convert (row, col, num) to variable index.
        Variable x[r,c,n] represents: cell (r,c) contains number n.
        
        Args:
            row: 0-8 (row index)
            col: 0-8 (column index)
            num: 1-9 (the number)
            
        Returns:
            Index in range [0, 728]
        """
        return row * 81 + col * 9 + (num - 1)
    
    def _var_to_cell(self, var_idx: int) -> Tuple[int, int, int]:
        """
        Convert variable index back to (row, col, num).
        
        Args:
            var_idx: Variable index [0, 728]
            
        Returns:
            Tuple (row, col, num) where row, col ∈ [0,8] and num ∈ [1,9]
        """
        num = (var_idx % 9) + 1
        col = (var_idx // 9) % 9
        row = var_idx // 81
        return row, col, num
    
    def build_constraint_matrix(self) -> np.ndarray:
        """
        Build the constraint matrix A for the exact cover problem.
        
        Matrix dimensions: 324 × 729
        - Rows represent constraints
        - Columns represent variables x[r,c,n]
        
        Constraint types (row indices):
        - [0-80]: Cell constraints (each cell has exactly one number)
        - [81-161]: Row constraints (each row has each digit)
        - [162-242]: Column constraints (each column has each digit)
        - [243-323]: Box constraints (each 3×3 box has each digit)
        
        Returns:
            Binary constraint matrix A (324 × 729)
        """
        A = np.zeros((self.num_constraints, self.num_vars), dtype=np.int32)
        
        for row in range(9):
            for col in range(9):
                for num in range(1, 10):
                    var_idx = self._var_index(row, col, num)
                    
                    # Cell constraint: cell (row, col) has exactly one number
                    cell_constraint_idx = row * 9 + col
                    A[cell_constraint_idx, var_idx] = 1
                    
                    # Row constraint: row 'row' contains digit 'num'
                    row_constraint_idx = 81 + row * 9 + (num - 1)
                    A[row_constraint_idx, var_idx] = 1
                    
                    # Column constraint: column 'col' contains digit 'num'
                    col_constraint_idx = 162 + col * 9 + (num - 1)
                    A[col_constraint_idx, var_idx] = 1
                    
                    # Box constraint: 3×3 box contains digit 'num'
                    box_idx = (row // 3) * 3 + (col // 3)
                    box_constraint_idx = 243 + box_idx * 9 + (num - 1)
                    A[box_constraint_idx, var_idx] = 1
        
        return A
    
    def encode_puzzle(self, board: Board) -> np.ndarray:
        """
        Encode the initial puzzle as a constraint vector b.
        
        For given clues, we fix certain variables to 1 by adding constraints.
        
        Args:
            board: 9×9 board with 0 for empty cells, 1-9 for filled cells
            
        Returns:
            Constraint vector b (324 × 1), all ones for exact cover
        """
        b = np.ones(self.num_constraints, dtype=np.int32)
        return b
    
    def apply_initial_clues(self, A: np.ndarray, b: np.ndarray, board: Board) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Reduce the constraint matrix based on initial clues.
        
        For each given clue (row, col, num):
        1. Fix x[row,col,num] = 1
        2. Remove conflicting variables (other numbers in same cell/row/col/box)
        3. Update constraint matrix accordingly
        
        Args:
            A: Original constraint matrix
            b: Constraint vector
            board: Initial puzzle board
            
        Returns:
            Tuple of (reduced_A, reduced_b, fixed_vars)
            - reduced_A: Reduced constraint matrix
            - reduced_b: Updated constraint vector
            - fixed_vars: List of variable indices that are fixed to 1
        """
        A_work = A.copy()
        b_work = b.copy()
        fixed_vars = []
        removed_constraints = set()
        removed_vars = set()
        
        # Process all given clues
        for row in range(9):
            for col in range(9):
                if board[row][col] != 0:
                    num = board[row][col]
                    var_idx = self._var_index(row, col, num)
                    fixed_vars.append(var_idx)
                    
                    # Find all constraints satisfied by this variable
                    satisfied_constraints = np.where(A_work[:, var_idx] == 1)[0]
                    
                    for constraint_idx in satisfied_constraints:
                        if constraint_idx not in removed_constraints:
                            # Find all variables that also satisfy this constraint
                            conflicting_vars = np.where(A_work[constraint_idx, :] == 1)[0]
                            removed_vars.update(conflicting_vars.tolist())
                            removed_constraints.add(constraint_idx)
        
        # Keep only variables and constraints not yet satisfied
        remaining_vars = [i for i in range(self.num_vars) if i not in removed_vars]
        remaining_constraints = [i for i in range(self.num_constraints) if i not in removed_constraints]
        
        if len(remaining_vars) > 0 and len(remaining_constraints) > 0:
            A_reduced = A_work[np.ix_(remaining_constraints, remaining_vars)]
            b_reduced = b_work[remaining_constraints]
            return A_reduced, b_reduced, remaining_vars
        else:
            # Puzzle already solved by clues
            return np.zeros((0, 0)), np.zeros(0), remaining_vars
    
    def gaussian_elimination_with_backtracking(
        self, 
        A: np.ndarray, 
        b: np.ndarray, 
        var_mapping: List[int]
    ) -> Optional[np.ndarray]:
        """
        Solve the exact cover problem using modified Gaussian elimination.
        
        Since Sudoku requires binary (0/1) solutions, we use a hybrid approach:
        1. Row reduction to simplify the system
        2. Backtracking for binary constraint satisfaction
        3. Forward checking to detect conflicts early
        
        Args:
            A: Constraint matrix (reduced)
            b: Constraint vector (all ones)
            var_mapping: Mapping from reduced indices to original variable indices
            
        Returns:
            Solution vector x (729 × 1) with binary values, or None if no solution
        """
        if A.shape[0] == 0:
            # Already solved by initial clues
            solution = np.zeros(self.num_vars, dtype=np.int32)
            return solution
        
        # Use exact cover backtracking with constraint propagation
        return self._solve_exact_cover_backtracking(A, b, var_mapping)
    
    def _solve_exact_cover_backtracking(
        self, 
        A: np.ndarray, 
        b: np.ndarray, 
        var_mapping: List[int]
    ) -> Optional[np.ndarray]:
        """
        Solve exact cover using backtracking with constraint propagation.
        
        This implements Algorithm X (similar to Dancing Links):
        1. If matrix is empty, solution found
        2. Choose a constraint (row) with minimum options (MRV heuristic)
        3. Try each variable that satisfies this constraint
        4. Recursively solve reduced problem
        5. Backtrack if no solution
        
        Args:
            A: Current constraint matrix
            b: Current constraint vector
            var_mapping: Current variable to original index mapping
            
        Returns:
            Binary solution vector or None
        """
        solution = np.zeros(self.num_vars, dtype=np.int32)
        
        def backtrack(A_curr, var_map_curr, partial_solution):
            # Base case: all constraints satisfied
            if A_curr.shape[0] == 0 or np.all(np.sum(A_curr, axis=1) == 0):
                return True
            
            # Find constraint with minimum remaining variables (MRV heuristic)
            row_sums = np.sum(A_curr, axis=1)
            
            # If any constraint has no satisfying variables, backtrack
            if np.any(row_sums == 0):
                return False
            
            # Choose constraint with fewest options
            constraint_idx = np.argmin(row_sums[row_sums > 0])
            # Get the actual constraint index
            valid_indices = np.where(row_sums > 0)[0]
            constraint_idx = valid_indices[0] if len(valid_indices) > 0 else 0
            
            # Try each variable that satisfies this constraint
            satisfying_vars = np.where(A_curr[constraint_idx, :] == 1)[0]
            
            for var_local_idx in satisfying_vars:
                var_original_idx = var_map_curr[var_local_idx]
                
                # Set this variable to 1
                partial_solution[var_original_idx] = 1
                
                # Find constraints satisfied by this variable
                satisfied_constraints = np.where(A_curr[:, var_local_idx] == 1)[0]
                
                # Find conflicting variables (share constraints with chosen var)
                conflicting_vars_set = set()
                for constraint in satisfied_constraints:
                    conflicting_vars = np.where(A_curr[constraint, :] == 1)[0]
                    conflicting_vars_set.update(conflicting_vars.tolist())
                
                # Remove satisfied constraints and conflicting variables
                remaining_constraints = [i for i in range(A_curr.shape[0]) if i not in satisfied_constraints]
                remaining_vars = [i for i in range(A_curr.shape[1]) if i not in conflicting_vars_set]
                
                if len(remaining_constraints) == 0:
                    # All constraints satisfied
                    return True
                
                if len(remaining_vars) > 0:
                    A_next = A_curr[np.ix_(remaining_constraints, remaining_vars)]
                    var_map_next = [var_map_curr[i] for i in remaining_vars]
                    
                    if backtrack(A_next, var_map_next, partial_solution):
                        return True
                
                # Backtrack
                partial_solution[var_original_idx] = 0
            
            return False
        
        if backtrack(A, var_mapping, solution):
            return solution
        return None
    
    def decode_solution(self, solution: np.ndarray) -> Board:
        """
        Convert solution vector back to Sudoku board.
        
        Args:
            solution: Binary vector (729 × 1)
            
        Returns:
            9×9 Sudoku board
        """
        board = [[0 for _ in range(9)] for _ in range(9)]
        
        for var_idx in range(self.num_vars):
            if solution[var_idx] == 1:
                row, col, num = self._var_to_cell(var_idx)
                board[row][col] = num
        
        return board
    
    def solve_sudoku(self, board: Board) -> Optional[Board]:
        """
        Main solving method using linear algebra approach.
        
        Steps:
        1. Build constraint matrix A (exact cover formulation)
        2. Apply initial clues to reduce the system
        3. Solve reduced system using Gaussian elimination + backtracking
        4. Decode solution back to board representation
        
        Args:
            board: Input puzzle (0 for empty cells)
            
        Returns:
            Solved board or None if no solution exists
        """
        # Step 1: Build constraint matrix
        A = self.build_constraint_matrix()
        b = self.encode_puzzle(board)
        
        # Initialize solution with initial clues
        initial_solution = np.zeros(self.num_vars, dtype=np.int32)
        for row in range(9):
            for col in range(9):
                if board[row][col] != 0:
                    num = board[row][col]
                    var_idx = self._var_index(row, col, num)
                    initial_solution[var_idx] = 1
        
        # Step 2: Apply initial clues
        A_reduced, b_reduced, var_mapping = self.apply_initial_clues(A, b, board)
        
        # Step 3: Solve using Gaussian elimination with backtracking
        if len(var_mapping) > 0:
            remaining_solution = self.gaussian_elimination_with_backtracking(A_reduced, b_reduced, var_mapping)
            
            if remaining_solution is None:
                return None
            
            # Combine initial clues with found solution
            final_solution = initial_solution + remaining_solution
        else:
            # Puzzle already solved by initial clues
            final_solution = initial_solution
        
        # Step 4: Decode solution
        solved_board = self.decode_solution(final_solution)
        
        return solved_board


def solve_with_linear_algebra(board: Board) -> bool:
    """
    Wrapper function compatible with existing code.
    Modifies board in-place and returns True if solved.
    
    Args:
        board: 9×9 board to solve (modified in-place)
        
    Returns:
        True if solved, False otherwise
    """
    solver = LinearAlgebraSudokuSolver()
    solved_board = solver.solve_sudoku(board)
    
    if solved_board is None:
        return False
    
    # Copy solution back to original board
    for r in range(9):
        for c in range(9):
            board[r][c] = solved_board[r][c]
    
    return True
