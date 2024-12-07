from SAT import DPLLSolver, Formula
from Utilities import RandomFormulaGenerator, FormulaSimplifier
from BaseVisualizer import BaseVisualizer
import time
import sys
from colorama import Fore, Style, init

class DPLLVisualizer(DPLLSolver, BaseVisualizer):
    """DPLL Solver with interactive visualization"""
    
    def __init__(self):
        DPLLSolver.__init__(self, debug=True)
        BaseVisualizer.__init__(self)
    
    def solve(self, formula: Formula) -> dict:
        """Override solve method to reset step counter"""
        self.step_count = 0
        self.depth = 0
        return super().solve(formula)
    
    def wait_for_user(self, message: str = "\nPress Enter to continue..."):
        """Wait for user to press Enter"""
        self.color_print(message, Fore.CYAN, end='')
        input()
    
    def _dpll(self, formula: Formula, assignments: dict) -> dict:
        """Modified DPLL implementation with visualization"""
        self.step_count += 1
        self.depth += 1
        current_depth = self.depth
        
        # Clear screen and print step counter
        self.clear_screen()
        self.print_step_counter(self.step_count, current_depth)
        
        # Print current state
        self.color_print("\n" + "="*50, Fore.CYAN)
        self.color_print(f"Step {self.step_count} (Depth {current_depth}):", Fore.CYAN, Style.BRIGHT)
        self.color_print("="*50, Fore.CYAN)
        
        # Print current formula
        self.color_print("\nCurrent formula:", Fore.YELLOW, Style.BRIGHT)
        if not formula.clauses:
            self.color_print("  [Empty - all clauses satisfied]", Fore.GREEN)
        else:
            for clause in formula.clauses:
                self.color_print(f"  {clause}", Fore.YELLOW)
        
        # Print current assignments
        self.color_print("\nCurrent assignments:", Fore.MAGENTA, Style.BRIGHT)
        if not assignments:
            self.color_print("  [None]", Fore.MAGENTA)
        else:
            for var, val in sorted(assignments.items()):
                color = Fore.GREEN if val else Fore.RED
                self.color_print(f"  x{var} = {str(val)}", color)
        
        self.wait_for_user("\nPress Enter to analyze formula...")
        
        # Base cases
        if not formula.clauses:
            self.color_print("\n✨ Success! Found satisfying assignment.", Fore.GREEN, Style.BRIGHT)
            self.wait_for_user()
            result = self._complete_assignment(assignments, formula.num_variables)
            self.depth -= 1
            return result
        
        if any(not clause.literals for clause in formula.clauses):
            self.color_print("\n❌ Found empty clause - this branch fails.", Fore.RED, Style.BRIGHT)
            self.wait_for_user()
            self.stats.increment("backtracks")
            self.depth -= 1
            return None
        
        # Unit propagation
        unit_clause = next((clause for clause in formula.clauses if clause.is_unit()), None)
        if unit_clause:
            lit = unit_clause.literals[0]
            self.color_print("\nChecking for unit clauses...", Fore.BLUE)
            self.color_print(f"🔍 Found unit clause: {unit_clause}", Fore.GREEN, Style.BRIGHT)
            self.color_print(f"Applying unit propagation: Setting {lit}", Fore.GREEN)
            self.wait_for_user()
            
            self.stats.increment("unit_propagations")
            new_assignments = assignments.copy()
            new_assignments[lit.variable] = lit.is_positive
            
            simplified = FormulaSimplifier.simplify_formula(formula, new_assignments)
            self.stats.append("clause_sizes", len(simplified.clauses))
            
            return self._dpll(simplified, new_assignments)
        
        self.color_print("\nNo unit clauses found.", Fore.BLUE)
        
        # Pure literal elimination
        pure_literal = self._find_pure_literal(formula)
        if pure_literal:
            self.color_print("\nChecking for pure literals...", Fore.BLUE)
            self.color_print(f"🔍 Found pure literal: {pure_literal}", Fore.GREEN, Style.BRIGHT)
            self.color_print(f"Applying pure literal elimination: Setting {pure_literal}", Fore.GREEN)
            self.wait_for_user()
            
            self.stats.increment("pure_literals")
            new_assignments = assignments.copy()
            new_assignments[pure_literal.variable] = pure_literal.is_positive
            
            return self._dpll(
                FormulaSimplifier.simplify_formula(formula, new_assignments),
                new_assignments
            )
        
        self.color_print("\nNo pure literals found.", Fore.BLUE)
        
        # Two-clause rule
        two_clause_assignment = self._apply_two_clause_rule(formula)
        if two_clause_assignment:
            var, value = two_clause_assignment
            self.color_print("\nChecking for two-literal clauses...", Fore.BLUE)
            self.color_print(f"🔍 Found two-literal clause", Fore.GREEN, Style.BRIGHT)
            self.color_print(f"Applying two-clause rule: Setting x{var} = {value}", Fore.GREEN)
            self.wait_for_user()
            
            new_assignments = assignments.copy()
            new_assignments[var] = value
            
            return self._dpll(
                FormulaSimplifier.simplify_formula(formula, new_assignments),
                new_assignments
            )
        
        self.color_print("\nNo two-literal clauses found.", Fore.BLUE)
        
        # Variable selection
        self.color_print("\nNo simplification rules apply. Choosing variable for branching...", Fore.BLUE)
        var = self._choose_next_variable(formula)
        self.color_print(f"🎯 Choosing variable x{var} (most frequent)", Fore.BLUE, Style.BRIGHT)
        self.wait_for_user()
        
        self.stats.append("variable_frequencies", var)
        
        # Try assignments
        for value in [True, False]:
            self.color_print(f"\n↪ Trying x{var} = {value}", Fore.BLUE)
            self.wait_for_user()
            
            new_assignments = assignments.copy()
            new_assignments[var] = value
            result = self._dpll(
                FormulaSimplifier.simplify_formula(formula, new_assignments),
                new_assignments
            )
            if result is not None:
                self.depth -= 1
                return result
        
        self.color_print(f"\n↩ Both assignments for x{var} failed, backtracking...", Fore.RED)
        self.wait_for_user()
        self.stats.increment("backtracks")
        self.depth -= 1
        return None
def main():
    visualizer = DPLLVisualizer()
    
    # Print welcome and explanation
    visualizer.print_welcome("Welcome to the DPLL Algorithm Visualizer")
    
    # Print rules
    visualizer.color_print("\nThe algorithm applies these rules in sequence:", Fore.WHITE, Style.BRIGHT)
    visualizer.print_rule_box("1. Unit Propagation",
                            "Assigns values to variables that appear alone in a clause")
    visualizer.print_rule_box("2. Pure Literal Elimination",
                            "Assigns values to variables that appear with only one polarity")
    visualizer.print_rule_box("3. Two-Clause Rule",
                            "Makes assignments based on clauses with exactly two literals")
    visualizer.print_rule_box("4. Most Frequent Variable",
                            "Chooses the most frequently occurring unassigned variable")
    
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
        # Check if adding the clause would make the line too long
        if len(current_line + " ∧ " + clause) > (visualizer.term_width - 20):  # Leave some margin
            formula_lines.append(current_line)
            current_line = clause
        else:
            current_line += " ∧ " + clause
    formula_lines.append(current_line)
    
    # Create and display the box with the formatted formula
    formula_box = visualizer.create_box("\n".join(formula_lines))
    for line in formula_box:
        visualizer.color_print(visualizer.center_text(line), Fore.YELLOW)
    
    visualizer.color_print("\nPress Enter to start DPLL solution process...", Fore.CYAN, end='')
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
        failure_box = visualizer.create_box("❌ Formula is UNSATISFIABLE")
        for line in failure_box:
            visualizer.color_print(visualizer.center_text(line), Fore.RED, Style.BRIGHT)
    
    # Print statistics
    stats = [
        ("Total steps", visualizer.step_count),
        ("Unit propagations", visualizer.stats.stats['unit_propagations'].value),
        ("Pure literals", visualizer.stats.stats['pure_literals'].value),
        ("2-clause rules", visualizer.stats.stats['two_clause_rules'].value),
        ("Backtracks", visualizer.stats.stats['backtracks'].value),
    ]
    
    visualizer.print_statistics(stats)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
        sys.exit(0)