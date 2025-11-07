from typing import List, Optional, Tuple
import argparse
import sys
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
import random
import os
from datetime import datetime

Board = List[List[int]]


def parse_puzzle(puzzle_str: str) -> Board:
    cleaned = puzzle_str.strip().replace("\n", "").replace(" ", "")
    if len(cleaned) != 81:
        raise ValueError("Puzzle must contain exactly 81 characters (use 0 or . for empty).")
    board: Board = []
    for i in range(9):
        row = []
        for j in range(9):
            ch = cleaned[i * 9 + j]
            if ch in ".0":
                row.append(0)
            elif ch.isdigit() and 1 <= int(ch) <= 9:
                row.append(int(ch))
            else:
                raise ValueError(f"Invalid character '{ch}' in puzzle.")
        board.append(row)
    return board


def board_to_string(board: Board) -> str:
    lines = []
    for i, row in enumerate(board):
        lines.append(" ".join(str(n) if n != 0 else "." for n in row))
    return "\n".join(lines)


def is_valid(board: Board, row: int, col: int, num: int) -> bool:
    # Check row and column
    for k in range(9):
        if board[row][k] == num or board[k][col] == num:
            return False
    # Check 3x3 box
    box_r = (row // 3) * 3
    box_c = (col // 3) * 3
    for r in range(box_r, box_r + 3):
        for c in range(box_c, box_c + 3):
            if board[r][c] == num:
                return False
    return True


def find_empty_mrv(board: Board) -> Optional[Tuple[int, int, List[int]]]:
    best_cell: Optional[Tuple[int, int, List[int]]] = None
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                candidates = [n for n in range(1, 10) if is_valid(board, r, c, n)]
                if not candidates:
                    return (r, c, [])  # dead end
                if best_cell is None or len(candidates) < len(best_cell[2]):
                    best_cell = (r, c, candidates)
                    if len(candidates) == 1:
                        return best_cell
    return best_cell


def solve(board: Board) -> bool:
    found = find_empty_mrv(board)
    if found is None:
        return True  # solved
    row, col, candidates = found
    if not candidates:
        return False
    for num in candidates:
        board[row][col] = num
        if solve(board):
            return True
        board[row][col] = 0
    return False


def load_puzzle_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------ New: Puzzle randomization helpers ------------------

def permute_digits(board: Board) -> Board:
    perm = random.sample(range(1, 10), 9)
    mapping = {i + 1: perm[i] for i in range(9)}
    return [[mapping[val] if val != 0 else 0 for val in row] for row in board]


def shuffle_rows_within_bands(board: Board) -> Board:
    new = [row[:] for row in board]
    for band in range(0, 9, 3):
        rows = board[band:band + 3]
        perm = random.sample([0, 1, 2], 3)
        for i, p in enumerate(perm):
            new[band + i] = rows[p][:]
    return new


def shuffle_cols_within_stacks(board: Board) -> Board:
    # operate on columns by transposing, shuffling rows within bands of transposed, then transpose back
    trans = [list(col) for col in zip(*board)]
    trans = shuffle_rows_within_bands(trans)
    return [list(col) for col in zip(*trans)]


def swap_row_bands(board: Board) -> Board:
    bands = [board[i:i + 3] for i in range(0, 9, 3)]
    perm = random.sample([0, 1, 2], 3)
    new = []
    for p in perm:
        new.extend([row[:] for row in bands[p]])
    return new


def swap_col_stacks(board: Board) -> Board:
    # operate on columns by transposing, swapping bands, then transpose back
    trans = [list(col) for col in zip(*board)]
    trans = swap_row_bands(trans)
    return [list(col) for col in zip(*trans)]


def maybe_transpose(board: Board) -> Board:
    if random.choice((True, False)):
        return [list(col) for col in zip(*board)]
    return board


def randomize_puzzle(board: Board) -> Board:
    b = [row[:] for row in board]
    # apply a sequence of valid transformations that preserve solvability
    b = permute_digits(b)
    b = maybe_transpose(b)
    b = shuffle_rows_within_bands(b)
    b = shuffle_cols_within_stacks(b)
    b = swap_row_bands(b)
    b = swap_col_stacks(b)
    return b


# ------------------ Save/IO helpers ------------------
def is_board_valid_filled(board: Board) -> bool:
    """Return True if board has no duplicate in any row/col/box. Assumes board is fully filled (no zeros) or checks only non-zero cells."""
    # rows
    for r in range(9):
        seen = set()
        for c in range(9):
            v = board[r][c]
            if v == 0:
                continue
            if v in seen:
                return False
            seen.add(v)
    # cols
    for c in range(9):
        seen = set()
        for r in range(9):
            v = board[r][c]
            if v == 0:
                continue
            if v in seen:
                return False
            seen.add(v)
    # boxes
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            seen = set()
            for r in range(br, br + 3):
                for c in range(bc, bc + 3):
                    v = board[r][c]
                    if v == 0:
                        continue
                    if v in seen:
                        return False
                    seen.add(v)
    return True


def save_board_to_file(board: Board, solved: bool) -> str:
    """Save board to Saved/solved or Saved/unsolved and return path."""
    # Use __file__ contextually within a running script; for a standalone
    # environment, os.getcwd() might be more appropriate, but sticking to
    # the original file's logic structure.
    base_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "Saved")
    sub = "solved" if solved else "unsolved"
    target_dir = os.path.join(base_dir, sub)
    os.makedirs(target_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ensure unique filename
    filename = f"sudoku_{timestamp}_{sub}.txt"
    path = os.path.join(target_dir, filename)
    # if file exists, append a counter
    counter = 1
    orig_path = path
    while os.path.exists(path):
        # NOTE: This logic for finding a unique name is slightly flawed if path is not a simple string
        # but will work for the simple filename constructed above.
        path = orig_path.replace(".txt", f"_{counter}.txt")
        counter += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(board_to_string(board))
    return path

# ------------------ End Save/IO helpers ------------------

# ------------------ End randomization helpers ------------------

# ------------------ New: Tkinter GUI integration ------------------

class SudokuGUI:
    def __init__(self, root: tk.Tk, initial_board: Optional[Board] = None):
        self.root = root
        self.root.title("Sudoku Solver")
        self.cells = [[None for _ in range(9)] for _ in range(9)]
        # Use a flat background color for the main frame to better show highlights
        self.frame = tk.Frame(root, padx=10, pady=10) 
        self.frame.pack()

        # store mask of which cells were filled before a solve (True = user/sample filled)
        self._pre_solve_filled: Optional[List[List[bool]]] = None
        self._filled_fg = "black"
        self._solver_fg = "#65B1FC"

        # Instruction label so user knows they can input manually
        instr = tk.Label(self.root, text="Click a cell and type 1-9. Press Solve to solve.", fg="black")
        instr.pack(pady=(0, 6))

        self._build_grid()
        self._build_controls()
        # focus first cell for quick manual input
        self.root.after(50, lambda: self.cells[0][0].focus_set())

        if initial_board:
            self.set_board_to_entries(initial_board)

    def _build_grid(self):
        font = ("Helvetica", 18)
        # ------------------- MODIFIED BLOCK START -------------------
        # Define colors for alternating 3x3 boxes
        # Using a subtle light blue hex code for highlighting
        color_highlight = "#E6F0FF"
        color_default = "white" # Explicitly setting white for non-highlighted cells

        for r in range(9):
            for c in range(9):
                # Check if the 3x3 box should be highlighted to create a checkerboard pattern
                # based on 3x3 blocks. We alternate based on the sum of row-band index and col-stack index.
                is_highlighted_box = ((r // 3) + (c // 3)) % 2 == 1
                box_bg_color = color_highlight if is_highlighted_box else color_default

                e = tk.Entry(
                    self.frame, 
                    width=2, 
                    font=font, 
                    justify="center", 
                    fg=self._filled_fg, 
                    # Apply alternating background color
                    bg=box_bg_color,
                    selectbackground=color_highlight, # Use highlight color for selection too
                    selectforeground="black"
                )
                # ------------------- MODIFIED BLOCK END -------------------
                
                # Use slightly larger padding for the 3x3 box borders for better visual separation
                e.grid(
                    row=r, column=c, 
                    padx=(1 if c % 3 != 0 else 4), 
                    pady=(1 if r % 3 != 0 else 4)
                )
                
                # restrict input to single digit or empty
                e.bind("<Key>", self._on_key)
                # select contents when focused to make replacing easy
                e.bind("<FocusIn>", lambda ev, w=e: w.select_range(0, tk.END))
                self.cells[r][c] = e

    def _build_controls(self):
        # Two-row button layout: top row = Solve, Generate, Save; bottom row = Clear, Blank
        top_ctrl = tk.Frame(self.root, pady=4)
        top_ctrl.pack()
        blank_btn = tk.Button(top_ctrl, text="Blank", width=12, command=self.on_blank)
        blank_btn.grid(row=0, column=0, padx=8)
        load_btn = tk.Button(top_ctrl, text="Generate", width=12, command=self.on_load_sample)
        load_btn.grid(row=0, column=1, padx=4)
        load_saved_btn = tk.Button(top_ctrl, text="Load", width=12, command=self.on_load_saved)
        load_saved_btn.grid(row=0, column=3, padx=4)
        

        bottom_ctrl = tk.Frame(self.root, pady=4)
        bottom_ctrl.pack()
        # Blank button only (Clear removed per user request)
        # Use the same blue as solver-filled cells for the Solve button background
        # and a very light green for the Save button to make them distinct but subtle.
        solve_btn = tk.Button(
            bottom_ctrl,
            text="Solve",
            width=12,
            command=self.on_solve,
            bg=self._solver_fg,
            fg="black",
            activebackground="#8EC4FA",
        )
        solve_btn.grid(row=0, column=0, padx=4)
        save_btn = tk.Button(
            bottom_ctrl,
            text="Save",
            width=12,
            command=self.on_save,
            bg="#E6FFE6",
            fg="black",
            activebackground="#CFFFD0",
        )
        save_btn.grid(row=0, column=2, padx=4)
        

        self.status = tk.Label(self.root, text="", fg="blue")
        self.status.pack()

        self.solve_btn = solve_btn
        self.load_btn = load_btn
        self.save_btn = save_btn
        self.load_saved_btn = load_saved_btn
        self.blank_btn = blank_btn

    def _on_key(self, event):
        # allow digits 1-9, BackSpace, Delete, Tab, Left/Right arrows, Return
        # Prevent typing into generated cells even if their state was not properly set
        try:
            widget = event.widget
            for r in range(9):
                for c in range(9):
                    if self.cells[r][c] is widget:
                        if self._generated_mask[r][c]:
                            return "break"
                        break
        except Exception:
            pass
        if event.keysym in ("BackSpace", "Delete", "Tab", "Left", "Right", "Up", "Down", "Return"):
            return
        if event.char and event.char in "123456789":
            # let Entry handle normal input (one digit). After insertion, remove extra chars.
            widget = event.widget
            self.root.after(1, lambda w=widget: self._truncate_entry(w))
            return
        # disallow other keys
        return "break"

    def _truncate_entry(self, widget):
        try:
            # If this widget is a generated cell, prevent any changes
            for r in range(9):
                for c in range(9):
                    if self.cells[r][c] is widget and self._generated_mask[r][c]:
                        widget.delete(0, tk.END)
                        return
        except Exception:
            pass
        v = widget.get()
        if len(v) > 1:
            widget.delete(0, tk.END)
            widget.insert(0, v[-1])
        if v == ".":
            widget.delete(0, tk.END)

    def get_board_from_entries(self) -> Board:
        board: Board = [[0]*9 for _ in range(9)]
        for r in range(9):
            for c in range(9):
                v = self.cells[r][c].get().strip()
                if v in ("", "."):
                    board[r][c] = 0
                elif v.isdigit() and 1 <= int(v) <= 9:
                    board[r][c] = int(v)
                else:
                    raise ValueError(f"Invalid entry at {r+1},{c+1}: {v!r}")
        return board

    def set_board_to_entries(self, board: Board):
        for r in range(9):
            for c in range(9):
                val = board[r][c]
                ent = self.cells[r][c]
                ent.config(fg=self._filled_fg)
                if val == 0:
                    ent.delete(0, tk.END)
                else:
                    ent.delete(0, tk.END)
                    ent.insert(0, str(val))
        # enforce generated mask after populating entries so generated cells remain non-editable
        try:
            self._apply_generated_mask()
        except Exception:
            pass

    def set_board_with_highlight(self, board: Board, prefilled_mask: List[List[bool]]):
        """
        Update the Entry widgets with board values. Cells that were empty before
        solving (prefilled_mask == False) are shown in solver color (blue).
        """
        for r in range(9):
            for c in range(9):
                val = board[r][c]
                ent = self.cells[r][c]
                ent.config(fg=self._filled_fg)  # default
                ent.config(state="normal")
                ent.delete(0, tk.END)
                if val != 0:
                    ent.insert(0, str(val))
                    if not prefilled_mask[r][c]:
                        ent.config(fg=self._solver_fg)
                # leave empty cells blank (shouldn't happen after successful solve)

    def set_widgets_state(self, state: str):
        for r in range(9):
            for c in range(9):
                try:
                    self.cells[r][c].config(state=state)
                except Exception:
                    pass
        self.solve_btn.config(state=state)
        self.load_btn.config(state=state)
        # allow Save when widgets are enabled; when disabling UI keep Save disabled too
        self.save_btn.config(state=state)
        try:
            self.load_saved_btn.config(state=state)
        except Exception:
            pass
        try:
            self.blank_btn.config(state=state)
        except Exception:
            pass

        # If re-enabling widgets, re-apply generated mask so generated cells stay non-editable
        if state == "normal":
            try:
                self._apply_generated_mask()
            except Exception:
                pass

    # Clear button and handler removed per user request

    def on_save(self):
        """Save current board to Saved/solved or Saved/unsolved depending on whether it's solved."""
        try:
            board = self.get_board_from_entries()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return

        # determine if board is considered solved (no zeros and valid)
        has_zero = any(board[r][c] == 0 for r in range(9) for c in range(9))
        if not has_zero and is_board_valid_filled(board):
            solved_flag = True
        else:
            solved_flag = False

        try:
            path = save_board_to_file(board, solved_flag)
        except Exception as e:
            messagebox.showerror("Save failed", f"Failed to save file: {e}")
            return

        # This will fail outside of a script environment (like in this sandbox)
        # Fallback to just the path
        try:
             rel = os.path.relpath(path, os.path.dirname(os.path.abspath(sys.argv[0])))
        except ValueError:
             rel = path
             
        self.status.config(text=f"Saved to {rel}")
        messagebox.showinfo("Saved", f"Board saved to:\n{path}")

    def on_load_sample(self):
        sample = (
            "530070000"
            "600195000"
            "098000060"
            "800060003"
            "400803001"
            "700020006"
            "060000280"
            "000419005"
            "000080079"
        )
        board = parse_puzzle(sample)
        board = randomize_puzzle(board)
        self.set_board_to_entries(board)
        # mark generated cells and make them non-editable; blanks remain editable
        self._generated_mask = [[board[r][c] != 0 for c in range(9)] for r in range(9)]
        self._apply_generated_mask()
        self.status.config(text="Random puzzle generated")

    def on_load_saved(self):
        """Open a file dialog to choose a puzzle from Saved/unsolved and load it into the grid."""
        try:
            script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        except Exception:
            script_dir = os.getcwd()

        unsolved_dir = os.path.join(script_dir, "Saved", "unsolved")
        try:
            os.makedirs(unsolved_dir, exist_ok=True)
        except Exception:
            pass

        path = filedialog.askopenfilename(
            initialdir=unsolved_dir,
            title="Select unsolved puzzle",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            puzzle_input = load_puzzle_from_file(path)
            board = parse_puzzle(puzzle_input)
        except Exception as e:
            messagebox.showerror("Load failed", f"Failed to load puzzle: {e}")
            return

        # Load board and mark non-empty cells as generated (non-editable)
        self.set_board_to_entries(board)
        self._generated_mask = [[board[r][c] != 0 for c in range(9)] for r in range(9)]
        self._apply_generated_mask()
        try:
            rel = os.path.relpath(path, script_dir)
        except Exception:
            rel = path
        self.status.config(text=f"Loaded {rel}")

    def on_blank(self):
        # create an empty board and clear generated mask so everything is editable
        board = [[0 for _ in range(9)] for _ in range(9)]
        self._generated_mask = [[False for _ in range(9)] for _ in range(9)]
        self.set_board_to_entries(board)
        self._apply_generated_mask()
        self.status.config(text="Blank board created")

    def on_solve(self):
        try:
            board = self.get_board_from_entries()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        # build pre-solve mask: True for cells that already had values
        self._pre_solve_filled = [[board[r][c] != 0 for c in range(9)] for r in range(9)]

        # disable widgets while solving
        self.set_widgets_state("disabled")
        self.status.config(text="Solving...")
        thread = threading.Thread(target=self._worker_solve, args=(board,), daemon=True)
        thread.start()

    def _worker_solve(self, board: Board):
        success = solve(board)
        # schedule UI update back on main thread
        self.root.after(0, lambda: self._on_solve_finished(success, board))

    def _on_solve_finished(self, success: bool, board: Board):
        if success:
            # update entries and color solver-filled cells blue
            if self._pre_solve_filled is None:
                # fallback: treat all non-zero as solver values
                prefilled = [[board[r][c] != 0 for c in range(9)] for r in range(9)]
            else:
                prefilled = self._pre_solve_filled
            # ensure entries are editable while updating
            for r in range(9):
                for c in range(9):
                    self.cells[r][c].config(state="normal")
            self.set_board_with_highlight(board, prefilled)
            self.status.config(text="Solved")
        else:
            messagebox.showinfo("No solution", "No solution found for the given puzzle.")
            self.status.config(text="No solution")
        self.set_widgets_state("normal")

# ------------------ End GUI code ------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a Sudoku solver on a provided puzzle.")
    parser.add_argument("--puzzle", "-p", help="81-char puzzle string (use 0 or . for empty).")
    parser.add_argument("--file", "-f", help="Path to a text file containing the puzzle.")
    parser.add_argument("--gui", "-g", action="store_true", help="Launch the tkinter GUI.")
    args = parser.parse_args(argv)

    # Launch GUI if user explicitly asked for it OR if no puzzle/file was provided.
    if args.gui or (not args.puzzle and not args.file):
        root = tk.Tk()
        # if puzzle or file provided, try to pre-load
        initial = None
        try:
            if args.file:
                puzzle_input = load_puzzle_from_file(args.file)
                initial = parse_puzzle(puzzle_input)
            elif args.puzzle:
                initial = parse_puzzle(args.puzzle)
        except Exception as e:
            messagebox.showwarning("Load puzzle", f"Failed to load puzzle: {e}")
            initial = None
        app = SudokuGUI(root, initial_board=initial)
        root.mainloop()
        return 0

    if args.file:
        try:
            puzzle_input = load_puzzle_from_file(args.file)
        except Exception as e:
            print(f"Failed to read file: {e}", file=sys.stderr)
            return 2
    elif args.puzzle:
        puzzle_input = args.puzzle
    else:
        # Sample puzzle (medium difficulty). Use 0 or . for empties.
        puzzle_input = (
            "530070000"
            "600195000"
            "098000060"
            "800060003"
            "400803001"
            "700020006"
            "060000280"
            "000419005"
            "000080079"
        )

    try:
        board = parse_puzzle(puzzle_input)
    except ValueError as e:
        print(f"Invalid puzzle: {e}", file=sys.stderr)
        return 2

    print("Input puzzle:")
    print(board_to_string(board))
    print("\nSolving...\n")

    if solve(board):
        print("Solved puzzle:")
        print(board_to_string(board))
        return 0
    else:
        print("No solution found.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # The original code raises SystemExit in __main__
    # To run in a sandbox, we wrap the call.
    try:
        sys.exit(main())
    except SystemExit:
        pass # Allow SystemExit to pass through