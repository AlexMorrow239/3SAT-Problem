from SAT import RandomSATSolver, Formula
from Utilities import RandomFormulaGenerator
from BaseVisualizer import BaseVisualizer
import time
import sys
from colorama import Fore, Style
import random

class RandomWalkVisualizer(RandomSATSolver, BaseVisualizer):
    """Random Walk Solver with interactive visualization"""
    
    def __init__(self):
        RandomSATSolver.__init__(self, max_flips=100, max_tries=10, debug=True)
        BaseVisualizer.__init__(self)
    
    def solve(self, formula: Formula) -> dict:
        """Override solve method to reset step counter"""
        self.step_count = 0
        self.total_flips = 0
        return super().solve(formula)
    
    def wait_for_user(self, message: str = "\nPress Enter to continue..."):
        """Wait for user to press Enter"""
        self.color_print(message, Fore.CYAN, end='')
        input()
    
    def _random_walk(self, formula: Formula, assignment: dict) -> dict:
        """Modified Random Walk implementation with visualization"""
        self.step_count += 1
        current_assignment = assignment.copy()
        prev_unsat_count = float('inf')
        
        # Clear screen
        self.clear_screen()
        
        # Print current state
        self.color_print("\n" + "="*50, Fore.CYAN)
        self.color_print(f"Try {self.step_count} - Starting new random walk", Fore.CYAN, Style.BRIGHT)
        self.color_print("="*50, Fore.CYAN)
        
        # Print initial assignment
        self.color_print("\nInitial random assignment:", Fore.YELLOW, Style.BRIGHT)
        for var, val in sorted(current_assignment.items()):
            color = Fore.GREEN if val else Fore.RED
            self.color_print(f"  x{var} = {str(val)}", color)
        
        self.wait_for_user()
        
        # Random walk loop
        for flip in range(self.max_flips):
            self.total_flips += 1
            self.stats.increment("total_flips")
            
            # Clear screen and update state
            self.clear_screen()
            
            # Print current state
            self.color_print("\n" + "="*50, Fore.CYAN)
            self.color_print(f"Try {self.step_count} - Flip {flip + 1}/{self.max_flips}", 
                           Fore.CYAN, Style.BRIGHT)
            self.color_print("="*50, Fore.CYAN)
            
            # Check if solution found
            if self.verify_solution(formula, current_assignment):
                self.color_print("\n✨ Success! Found satisfying assignment.", 
                               Fore.GREEN, Style.BRIGHT)
                self.wait_for_user()
                return current_assignment
            
            # Find unsatisfied clauses
            unsatisfied = [
                clause for clause in formula.clauses
                if not self._verify_clause(clause, current_assignment)
            ]
            unsat_count = len(unsatisfied)
            
            # Print unsatisfied clauses
            self.color_print("\nUnsatisfied clauses:", Fore.YELLOW, Style.BRIGHT)
            for clause in unsatisfied:
                self.color_print(f"  {clause}", Fore.RED)
            
            self.color_print(f"\nTotal unsatisfied clauses: {unsat_count}", 
                           Fore.YELLOW, Style.BRIGHT)
            
            # Track improvement
            if unsat_count < prev_unsat_count:
                self.stats.increment("successful_flips")
                self.color_print("\n↗ Improvement found!", Fore.GREEN)
            elif unsat_count > prev_unsat_count:
                self.color_print("\n↘ Solution quality decreased", Fore.RED)
                self.stats.increment("local_minima")
            else:
                self.color_print("\n→ No change in solution quality", Fore.BLUE)
            
            prev_unsat_count = unsat_count
            
            # Choose random unsatisfied clause
            chosen_clause = random.choice(unsatisfied)
            chosen_literal = random.choice(chosen_clause.literals)
            
            self.color_print(f"\nChosen unsatisfied clause: {chosen_clause}", 
                           Fore.MAGENTA, Style.BRIGHT)
            self.color_print(f"Will flip variable: x{chosen_literal.variable}", 
                           Fore.MAGENTA, Style.BRIGHT)
            
            # Show current assignment
            self.color_print("\nCurrent assignment:", Fore.YELLOW)
            for var, val in sorted(current_assignment.items()):
                if var == chosen_literal.variable:
                    self.color_print(f"  x{var} = {str(val)} → {str(not val)}", 
                                   Fore.MAGENTA, Style.BRIGHT)
                else:
                    color = Fore.GREEN if val else Fore.RED
                    self.color_print(f"  x{var} = {str(val)}", color)
            
            self.wait_for_user()
            
            # Flip the chosen variable
            current_assignment[chosen_literal.variable] = not current_assignment[chosen_literal.variable]
        
        self.color_print("\n❌ Maximum flips reached without finding solution.", 
                        Fore.RED, Style.BRIGHT)
        self.wait_for_user()
        return None

def main():
    visualizer = RandomWalkVisualizer()
    
    # Print welcome and explanation
    visualizer.print_welcome("Welcome to the Random Walk SAT Algorithm Visualizer")
    
    # Print algorithm description
    visualizer.color_print("\nAlgorithm Steps:", Fore.WHITE, Style.BRIGHT)
    visualizer.print_rule_box("1. Random Assignment",
                            "Start with random values for all variables")
    visualizer.print_rule_box("2. Evaluate Solution",
                            "Count number of unsatisfied clauses")
    visualizer.print_rule_box("3. Random Selection",
                            "Choose random unsatisfied clause and variable")
    visualizer.print_rule_box("4. Flip Variable",
                            "Change chosen variable's value and repeat")
    
    # Print color legend
    visualizer.print_legend()
    
    # Get parameters
    n, ratio = visualizer.get_parameters(3, 4.0)
    
    # Generate formula with animation
    visualizer.color_print("\nGenerating random 3SAT formula...", Fore.BLUE)
    for i in range(5):
        visualizer.print_progress_bar(i, 4)
        time.sleep(0.1)
    visualizer.print_progress_bar(4, 4)
    print()
    
    # Generate and solve
    generator = RandomFormulaGenerator()
    formula = generator.generate(n, int(n * ratio))
    
    # Split the formula string into lines if it's too long
    formula_lines = []
    current_line = "Formula with {} variables:".format(formula.num_variables)
    formula_lines.append(current_line)
    
    # Handle each clause separately
    clauses = [str(clause) for clause in formula.clauses]
    current_line = clauses[0]
    for clause in clauses[1:]:
        if len(current_line + " ∧ " + clause) > (visualizer.term_width - 20):
            formula_lines.append(current_line)
            current_line = clause
        else:
            current_line += " ∧ " + clause
    formula_lines.append(current_line)
    
    # Create and display the box with the formatted formula
    formula_box = visualizer.create_box("\n".join(formula_lines))
    for line in formula_box:
        visualizer.color_print(visualizer.center_text(line), Fore.YELLOW)
    
    visualizer.color_print("\nPress Enter to start Random Walk solution process...", 
                          Fore.CYAN, end='')
    input()
    
    # Solve formula
    solution = visualizer.solve(formula)
    
    # Print final result
    result_box = visualizer.create_box("Final Result")
    for line in result_box:
        visualizer.color_print(visualizer.center_text(line), Fore.CYAN, Style.BRIGHT)
    
    if solution:
        success_box = visualizer.create_box("✅ Formula is SATISFIABLE")
        for line in success_box:
            visualizer.color_print(visualizer.center_text(line), Fore.GREEN, Style.BRIGHT)
        
        visualizer.color_print("\nSatisfying assignment:", Fore.GREEN)
        for var, val in sorted(solution.items()):
            color = Fore.GREEN if val else Fore.RED
            visualizer.color_print(visualizer.center_text(f"x{var} = {str(val)}"), color)
    else:
        failure_box = visualizer.create_box("❌ Maximum attempts reached - No solution found")
        for line in failure_box:
            visualizer.color_print(visualizer.center_text(line), Fore.RED, Style.BRIGHT)
    
    # Print statistics
    stats = [
        ("Total tries", visualizer.step_count),
        ("Total flips", visualizer.total_flips),
        ("Successful flips", visualizer.stats.stats['successful_flips'].value),
        ("Local minima", visualizer.stats.stats['local_minima'].value),
    ]
    
    visualizer.print_statistics(stats)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
        sys.exit(0)