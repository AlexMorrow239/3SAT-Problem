from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional
import random
import time
from collections import Counter
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.DEBUG, 
                   format='%(levelname)s - %(message)s')
logger = logging.getLogger('SATSolver')

# Disable matplotlib debug logs
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.INFO)

@dataclass
class Literal:
    variable: int
    is_positive: bool
    
    def __str__(self):
        return f"x{self.variable}" if self.is_positive else f"¬x{self.variable}"

@dataclass
class Clause:
    literals: List[Literal]

    def __str__(self):
        return f"({' ∨ '.join(str(lit) for lit in self.literals)})"

@dataclass
class Formula:
    clauses: List[Clause]
    num_variables: int

    def __str__(self):
        return f"Formula with {self.num_variables} variables:\n" + \
               f"{' ∧ '.join(str(clause) for clause in self.clauses)}"

class SATSolver(ABC):
    """Abstract base class for SAT solvers."""
    
    def __init__(self, debug):
        self.debug = debug
        self.reset_stats()

    def reset_stats(self):
        """Reset solver statistics. Override in subclasses to add specific stats."""
        self.stats = {}
    
    @abstractmethod
    def solve(self, formula: Formula) -> Optional[Dict[int, bool]]:
        """
        Solve the SAT formula and return an assignment if satisfiable.
        Returns None if unsatisfiable.
        """
        pass
    
    def _verify_assignment(self, formula: Formula, assignment: Dict[int, bool]) -> bool:
        """Verify if an assignment satisfies the formula."""
        for clause in formula.clauses:
            clause_satisfied = False
            for lit in clause.literals:
                if lit.is_positive == assignment[lit.variable]:
                    clause_satisfied = True
                    break
            if not clause_satisfied:
                return False
        return True

    def _complete_assignment(self, partial: Dict[int, bool], num_vars: int) -> Dict[int, bool]:
        """Complete a partial assignment by setting unassigned variables to True."""
        complete = partial.copy()
        for var in range(1, num_vars + 1):
            if var not in complete:
                complete[var] = True
        return complete

class DPLLSolver(SATSolver):
    """DPLL-based SAT solver implementation."""
    def __init__(self, debug=False):
        super().__init__(debug)
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            'backtracks': 0,
            'pure_literals': 0,
            'unit_clauses': 0,
            'depth': 0 if self.debug else None
        }

    def solve(self, formula: Formula) -> Dict[int, bool]:
        """Main DPLL solving method."""

        if self.debug: logger.debug("Starting DPLL Solver")

        self.reset_stats()
        result = self._dpll(formula, {})
        return result

    def _dpll(self, formula: Formula, assignments: Dict[int, bool]) -> Optional[Dict[int, bool]]:
        """
        Modified DPLL to return assignments directly instead of bool.
        Returns None if unsatisfiable, otherwise returns complete assignment dict.
        """
        if self.debug:
            self.stats['depth'] += 1
            indent = "  " * self.stats['depth']
            logger.debug(f"{indent}{'='*40}")
            logger.debug(f"{indent}DPLL at depth {self.stats['depth']}")
            logger.debug(f"{indent}Current formula:")
            for i, clause in enumerate(formula.clauses, 1):
                logger.debug(f"{indent}  Clause {i}: {clause}")
            logger.debug(f"{indent}Current assignments: {assignments}")
        
        # Base cases
        if not formula.clauses:
            if self.debug:
                logger.debug(f"{indent}SUCCESS: Empty formula - all clauses satisfied")
                logger.debug(f"{indent}Completing partial assignment...")
            
            complete_assignment = assignments.copy()
            for var in range(1, formula.num_variables + 1):
                if var not in complete_assignment:
                    complete_assignment[var] = True
                    if self.debug:
                        logger.debug(f"{indent}  Setting unassigned var x{var}=True")
            
            if self.debug:
                logger.debug(f"{indent}Final complete assignment: {complete_assignment}")
            return complete_assignment

        if any(not clause.literals for clause in formula.clauses):
            self.stats['backtracks'] += 1
            if self.debug:
                logger.debug(f"{indent}FAIL: Empty clause found - contradiction")
                logger.debug(f"{indent}Backtracking... (total backtracks: {self.stats['backtracks']})")
            return None

        # Try unit clauses first (most constrained)
        for clause in formula.clauses:
            if len(clause.literals) == 1:
                lit = clause.literals[0]
                if self.debug:
                    logger.debug(f"{indent}Found unit clause: {clause}")
                    logger.debug(f"{indent}Applying unit propagation: x{lit.variable}={lit.is_positive}")
                
                new_assignments = assignments.copy()
                new_assignments[lit.variable] = lit.is_positive
                self.stats['unit_clauses'] += 1
                
                if self.debug:
                    logger.debug(f"{indent}Simplifying formula with unit assignment...")
                
                simplified = self._simplify_formula(formula, new_assignments)
                
                if self.debug:
                    logger.debug(f"{indent}After simplification:")
                    for i, cl in enumerate(simplified.clauses, 1):
                        logger.debug(f"{indent}  Clause {i}: {cl}")
                
                result = self._dpll(simplified, new_assignments)
                if result is not None:
                    return result
                
                if self.debug:
                    logger.debug(f"{indent}Unit propagation branch failed, backtracking...")
                return None

        # Choose most frequent unassigned variable
        var = None
        max_freq = -1
        var_counts = Counter()
        for clause in formula.clauses:
            for lit in clause.literals:
                if lit.variable not in assignments:
                    var_counts[lit.variable] += 1
                    if var_counts[lit.variable] > max_freq:
                        var = lit.variable
                        max_freq = var_counts[lit.variable]

        if var is None:  # All variables assigned
            if self.debug:
                logger.debug(f"{indent}All variables assigned, verifying solution...")
            is_sat = self._verify_formula(formula, assignments)
            if self.debug:
                if is_sat:
                    logger.debug(f"{indent}SUCCESS: Valid solution found!")
                else:
                    logger.debug(f"{indent}FAIL: Invalid solution")
            return assignments if is_sat else None

        if self.debug:
            logger.debug(f"{indent}Selected branching variable x{var} (frequency: {max_freq})")
            logger.debug(f"{indent}Variable frequencies: {dict(var_counts)}")

        # Try True first
        if self.debug:
            logger.debug(f"{indent}Trying branch x{var}=True")
        
        new_assignments = assignments.copy()
        new_assignments[var] = True
        result = self._dpll(
            self._simplify_formula(formula, new_assignments),
            new_assignments
        )
        if result is not None:
            if self.debug:
                logger.debug(f"{indent}Found solution in True branch!")
            return result

        # Try False
        if self.debug:
            logger.debug(f"{indent}True branch failed, trying x{var}=False")
        
        new_assignments = assignments.copy()
        new_assignments[var] = False
        self.stats['backtracks'] += 1
        
        if self.debug:
            logger.debug(f"{indent}Backtracking... (total backtracks: {self.stats['backtracks']})")
        
        return self._dpll(
            self._simplify_formula(formula, new_assignments),
            new_assignments
        )
    
    def _simplify_formula(self, formula: Formula, assignments: Dict[int, bool]) -> Formula:
        """Simplify formula based on partial assignment."""
        new_clauses = []
        
        for clause in formula.clauses:
            # Keep track if clause is satisfied by any literal
            clause_satisfied = False
            new_literals = []
            
            for lit in clause.literals:
                if lit.variable in assignments:
                    # Check if this literal satisfies the clause
                    if lit.is_positive == assignments[lit.variable]:
                        clause_satisfied = True
                        break
                else:
                    new_literals.append(lit)
            
            if not clause_satisfied and new_literals:  # Only add if not satisfied and not empty
                new_clauses.append(Clause(new_literals))
            elif not clause_satisfied and not new_literals:  # Empty clause = contradiction
                return Formula([Clause([])], formula.num_variables)
                
        return Formula(new_clauses, formula.num_variables)

    def _verify_formula(self, formula: Formula, assignment: Dict[int, bool]) -> bool:
        """Verify if assignment satisfies the formula."""
        for clause in formula.clauses:
            clause_satisfied = False
            for lit in clause.literals:
                if lit.variable in assignment:
                    if lit.is_positive == assignment[lit.variable]:
                        clause_satisfied = True
                        break
            if not clause_satisfied:
                return False
        return True
        
class ExhaustiveSATSolver(SATSolver):
    def __init__(self, debug=False):
        super().__init__(debug)
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            'nodes_visited': 0,
            'solutions_found': 0
        }
        
    def solve(self, formula: Formula) -> Dict[int, bool]:
        """Main solving method for exhaustive search."""
        if self.debug:
            logger.debug("Starting Exhaustive solver")
            logger.debug(f"Initial formula: {formula}")
        
        self.assignment = {}
        self.reset_stats()
        
        # Start recursive search with variable 1
        result = self._exhaustive_search(formula, 1)
        
        if self.debug:
            logger.debug("Final Statistics:")
            logger.debug(f"Nodes visited: {self.stats['nodes_visited']}")
            logger.debug(f"Solutions found: {self.stats['solutions_found']}")
            logger.debug(f"Solution found: {result}")
            if result: logger.debug(f"Final assignment: {self.assignment}")
        
        return self.assignment if result else None

    def _exhaustive_search(self, formula: Formula, current_var: int) -> bool:
        """
        Recursive exhaustive search function.
        Tries all possible assignments for variables starting from current_var.
        """
        self.stats['nodes_visited'] += 1
        if self.debug:
            logger.debug(f"Examining variable x{current_var}")
            logger.debug(f"Current partial assignment: {self.assignment}")
        
        # If we've assigned all variables, check if formula is satisfied
        if current_var > formula.num_variables:
            is_satisfied = self._evaluate_formula(formula)
            if is_satisfied:
                self.stats['solutions_found'] += 1
            return is_satisfied
        
        # Try assigning False first
        self.assignment[current_var] = False
        if self.debug: logger.debug(f"Trying x{current_var}=False")
        
        if self._exhaustive_search(formula, current_var + 1):
            return True
        
        # Try assigning True next
        self.assignment[current_var] = True
        if self. debug: logger.debug(f"Trying x{current_var}=True")
        
        if self._exhaustive_search(formula, current_var + 1):
            return True
        
        # Neither assignment worked
        del self.assignment[current_var]
        return False

    def _evaluate_formula(self, formula: Formula) -> bool:
        """
        Evaluate if the formula is satisfied under the current complete assignment.
        """
        for clause in formula.clauses:
            clause_satisfied = False
            for literal in clause.literals:
                # A clause is satisfied if any literal evaluates to True
                literal_value = self.assignment[literal.variable]
                if literal.is_positive:
                    clause_satisfied = clause_satisfied or literal_value
                else:
                    clause_satisfied = clause_satisfied or not literal_value
                
                if clause_satisfied:
                    break
            
            # If any clause is false, the whole formula is false
            if not clause_satisfied:
                return False
        
        # All clauses were satisfied
        return True
    
def compare_solvers(n: int, ratio: float, results: dict):
    """Compare DPLL and Exhaustive search on the same formula. Store results in dict."""
    generator = RandomFormulaGenerator()
    formula = generator.generate(n, int(n * ratio))
    
    # Test DPLL
    dpll_solver = DPLLSolver()
    dpll_start = time.time()
    dpll_result = dpll_solver.solve(formula)
    dpll_time = time.time() - dpll_start
    
    # Test Exhaustive
    exhaustive_solver = ExhaustiveSATSolver()
    exhaustive_start = time.time()
    exhaustive_result = exhaustive_solver.solve(formula)
    exhaustive_time = time.time() - exhaustive_start
    
    # Store results
    key = (n, ratio)
    if key not in results:
        results[key] = []
    
    results[key].append({
        'dpll_time': dpll_time * 1000,  # Convert to ms
        'exhaustive_time': exhaustive_time * 1000,
        'is_sat': dpll_result is not None,
        'backtracks': dpll_solver.stats['backtracks'],
        'nodes_visited': exhaustive_solver.stats['nodes_visited']
    })

class RandomFormulaGenerator:
    @staticmethod
    def generate(num_variables: int, num_clauses: int) -> Formula:
        """Generate random 3SAT formula."""
        clauses = []
        for _ in range(num_clauses):
            # Select 3 distinct variables
            vars = random.sample(range(1, num_variables + 1), 3)
            # Randomly decide polarity
            literals = [
                Literal(var, random.choice([True, False]))
                for var in vars
            ]
            clauses.append(Clause(literals))
        return Formula(clauses, num_variables)

def plot_results(results: dict):
    """Create visualizations focusing on satisfiability and time complexity analysis."""
    # Create DataFrame from results
    data_list = []
    for (n, ratio), trials in results.items():
        for trial in trials:
            data_list.append({
                'n': n,
                'ratio': ratio,
                **trial
            })
    df = pd.DataFrame(data_list)
    
    # 1. Detailed Satisfiability Analysis
    plt.figure(figsize=(12, 6))
    
    # Main satisfiability plot
    for n in sorted(df['n'].unique()):
        subset = df[df['n'] == n]
        sat_prob = subset.groupby('ratio')['is_sat'].mean() * 100  # Convert to percentage
        plt.plot(sat_prob.index, sat_prob.values, 'o-', label=f'n={n}', linewidth=2)
    
    plt.axvline(x=4.26, color='r', linestyle='--', label='Theoretical Phase Transition')
    plt.xlabel('Clause-to-Variable Ratio (m/n)')
    plt.ylabel('Satisfiability Rate (%)')
    plt.title('Phase Transition Analysis in Random 3-SAT')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # 2. Time Complexity Analysis
    plt.figure(figsize=(12, 6))
    
    # Create variable size vs time plot for different ratios
    ratio_groups = [3.0, 4.26, 5.5]  # Below, at, and above phase transition
    for ratio in ratio_groups:
        times = []
        sizes = []
        for n in sorted(df['n'].unique()):
            subset = df[(df['n'] == n) & (np.abs(df['ratio'] - ratio) < 0.01)]
            if not subset.empty:
                mean_dpll_time = subset['dpll_time'].mean()
                times.append(mean_dpll_time)
                sizes.append(n)
        
        plt.plot(sizes, times, 'o-', label=f'ratio={ratio}', linewidth=2)
    
    plt.xlabel('Number of Variables (n)')
    plt.ylabel('Average Solving Time (ms)')
    plt.title('DPLL Time Complexity Analysis')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # 3. Statistical Analysis
    print("\nDetailed Satisfiability Analysis:")
    print("================================")
    
    # Calculate satisfiability rates
    sat_rates = df.groupby(['n', 'ratio'])['is_sat'].agg(['mean', 'count']).reset_index()
    sat_rates['mean'] = sat_rates['mean'] * 100  # Convert to percentage
    
    # Create a pivot table for satisfiability rates
    sat_pivot = sat_rates.pivot_table(
        values='mean',
        index='ratio',
        columns='n',
        aggfunc='first'
    )
    
    print("\nSatisfiability Rates (%):")
    print(sat_pivot.round(1))
    
    # Calculate and print median solving times
    time_pivot = pd.pivot_table(
        df,
        values='dpll_time',
        index='ratio',
        columns='n',
        aggfunc='median'
    )
    
    print("\nMedian Solving Times (ms):")
    print(time_pivot.round(2))

def main():
    print("3SAT Solver Comparison - DPLL vs Exhaustive Search")
    print("=" * 50)

    # More granular ratios around phase transition
    var_sizes = [5, 10, 15, 20]
    ratios = [3.0, 3.5, 4.0, 4.1, 4.26, 4.4, 4.6, 5.0, 5.5]
    trials = 10
    
    results = {}
    for n in var_sizes:
        print(f"\nTesting with n = {n}")
        print("-" * 30)
        
        for ratio in ratios:
            print(f"\nm/n = {ratio:.2f}")
            for _ in range(trials):
                compare_solvers(n, ratio, results)
    
    plot_results(results)

if __name__ == "__main__":
    main()
