from colorama import init, Fore, Back, Style
import shutil
import sys
import time
from typing import List, Dict, Any, Optional

# Initialize colorama for cross-platform color support
init()

class BaseVisualizer:
    """Base class for SAT solver visualization"""
    
    def __init__(self):
        self.step_count = 0
        self.depth = 0
        self.term_width, self.term_height = shutil.get_terminal_size()
    
    def color_print(self, text: str, color: str = Fore.WHITE, style: str = Style.NORMAL, end='\n'):
        """Helper function to print colored text"""
        print(f"{style}{color}{text}{Style.RESET_ALL}", end=end)
    
    def center_text(self, text: str) -> str:
        """Center text based on terminal width"""
        return text.center(self.term_width)
    
    def create_box(self, text: str, padding: int = 1) -> List[str]:
        """Create a box around text with given padding"""
        lines = text.split('\n')
        width = max(len(line) for line in lines) + 2 * padding
        
        top = '┌' + '─' * width + '┐'
        bottom = '└' + '─' * width + '┘'
        middle = [f'│{line.center(width)}│' for line in lines]
        
        return [top, *middle, bottom]
    
    def print_rule_box(self, rule_name: str, description: str, color: str = Fore.YELLOW):
        """Print a rule in a formatted box"""
        box = self.create_box(f"{rule_name}\n{description}")
        for line in box:
            self.color_print(self.center_text(line), color)
    
    def print_progress_bar(self, current: int, total: int, width: int = 40):
        """Print a progress bar showing algorithm progress"""
        percentage = current / total
        filled = int(width * percentage)
        bar = '█' * filled + '░' * (width - filled)
        self.color_print(f"\rProgress: |{bar}| {percentage:.1%}", Fore.CYAN, end='')
    
    def print_step_counter(self, current: int, depth: int, extra_info: Optional[str] = None):
        """Print step counter in the top-right corner"""
        # Save cursor position
        print("\033[s", end='')
        
        # Prepare counter text
        counter_text = f"Step: {current} | Depth: {depth}"
        if extra_info:
            counter_text += f" | {extra_info}"
        
        # Calculate box dimensions
        box_width = len(counter_text) + 2  # Add 2 for padding
        start_col = self.term_width - box_width - 2  # Subtract 2 for box borders
        start_row = 2  # Start the box at row 2 instead of row 0
        
        # Box drawing characters
        top = f"╭{'─' * box_width}╮"
        middle = f"│ {counter_text} │"
        bottom = f"╰{'─' * box_width}╯"
        
        # Move to positions and print each line
        # Add start_row to each row position
        print(f"\033[{start_row};{start_col}H", end='')
        self.color_print(top, Fore.CYAN)
        print(f"\033[{start_row + 1};{start_col}H", end='')
        self.color_print(middle, Fore.CYAN)
        print(f"\033[{start_row + 2};{start_col}H", end='')
        self.color_print(bottom, Fore.CYAN)
        
        # Restore cursor position
        print("\033[u", end='')
        sys.stdout.flush()  
    
    def clear_screen(self):
        """Clear the screen and move cursor to top"""
        print("\033[2J\033[H", end='')
    
    def print_welcome(self, title: str):
        """Print welcome message in a box"""
        self.clear_screen()
        welcome_box = self.create_box(title)
        for line in welcome_box:
            self.color_print(self.center_text(line), Fore.CYAN, Style.BRIGHT)
    
    def print_legend(self):
        """Print color legend"""
        self.color_print("\nColor Legend:", Fore.WHITE, Style.BRIGHT)
        self.color_print("✓ GREEN: Success, TRUE assignments", Fore.GREEN)
        self.color_print("✗ RED: Failure, FALSE assignments", Fore.RED)
        self.color_print("→ BLUE: Decisions and branching", Fore.BLUE)
        self.color_print("• YELLOW: Formulas and clauses", Fore.YELLOW)
        self.color_print("• MAGENTA: Variable assignments", Fore.MAGENTA)
    
    def get_parameters(self, default_n: int, default_ratio: float) -> tuple[int, float]:
        """Get problem parameters from user"""
        self.color_print("\n" + "="*self.term_width, Fore.CYAN)
        self.color_print("\nConfiguration:", Fore.CYAN, Style.BRIGHT)
        
        self.color_print(f"\nUse default parameters (n={default_n}, ratio={default_ratio})? [Y/n]: ", 
                        Fore.CYAN, end='')
        use_default = input().lower() != 'n'
        
        if use_default:
            return default_n, default_ratio
        
        while True:
            try:
                self.color_print("\nEnter number of variables (3-10 recommended): ", 
                               Fore.CYAN, end='')
                n = int(input())
                if n < 1:
                    self.color_print("❌ Number of variables must be positive", Fore.RED)
                    continue
                if n > 10:
                    self.color_print("⚠️  Warning: Large values may be hard to follow", 
                                   Fore.YELLOW)
                break
            except ValueError:
                self.color_print("❌ Please enter a valid number", Fore.RED)
        
        while True:
            try:
                self.color_print("Enter clause/variable ratio (2.0-5.0 recommended): ", 
                               Fore.CYAN, end='')
                ratio = float(input())
                if ratio <= 0:
                    self.color_print("❌ Ratio must be positive", Fore.RED)
                    continue
                break
            except ValueError:
                self.color_print("❌ Please enter a valid number", Fore.RED)
        
        return n, ratio
    
    def print_statistics(self, stats: List[tuple[str, Any]]):
        """Print statistics in a formatted box"""
        stats_box = self.create_box("Algorithm Statistics")
        for line in stats_box:
            self.color_print(self.center_text(line), Fore.CYAN, Style.BRIGHT)
        
        max_label_length = max(len(label) for label, _ in stats)
        for label, value in stats:
            centered_stat = self.center_text(f"{label:{max_label_length}}: {value}")
            self.color_print(centered_stat, Fore.CYAN)