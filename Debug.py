from SAT import Formula, RandomFormulaGenerator, ExhaustiveSATSolver, DPLLSolver
from typing import Dict
import time

def verify_assignment(formula: Formula, assignment: Dict[int, bool]) -> bool:
    """Verify if an assignment satisfies the formula."""
    for i, clause in enumerate(formula.clauses):
        clause_satisfied = False
        for literal in clause.literals:
            if literal.is_positive:
                clause_satisfied |= assignment[literal.variable]
            else:
                clause_satisfied |= not assignment[literal.variable]
        if not clause_satisfied:
            print(f"Clause {i + 1} not satisfied: {clause}")
            print(f"Relevant assignments:")
            for lit in clause.literals:
                print(f"  {lit}: {assignment[lit.variable]}")
            return False
    return True

def debug_solver(n: int, ratio: float, num_trials: int = 1):
    """
    Debug SAT solvers with detailed output for a specific problem size.
    
    Args:
        n: Number of variables
        ratio: Clause to variable ratio (m/n)
        num_trials: Number of test cases to run
    """
    print(f"\nDebugging SAT Solvers")
    print(f"Configuration: n={n}, m/n={ratio:.2f}, trials={num_trials}")
    print("=" * 60)
    
    generator = RandomFormulaGenerator()
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        print("-" * 30)
        
        # Generate and display formula
        num_clauses = int(n * ratio)
        formula = generator.generate(n, num_clauses)
        print("\nFormula Details:")
        print(f"Variables: {n}")
        print(f"Clauses: {num_clauses}")
        print("Formula:")
        for i, clause in enumerate(formula.clauses, 1):
            print(f"  Clause {i}: {clause}")
        
        # Run solvers with timing
        print("\nSolver Analysis:")
        print("-" * 20)
        
        # DPLL Analysis
        dpll = DPLLSolver(debug=True)
        dpll_start = time.time()
        dpll_result = dpll.solve(formula)
        dpll_time = (time.time() - dpll_start) * 1000  # Convert to ms
        
        print("\nDPLL Results:")
        print(f"Time: {dpll_time:.2f}ms")
        print(f"Result: {'Satisfiable' if dpll_result else 'Unsatisfiable'}")
        if dpll_result:
            print("Assignment:", end=" ")
            sorted_vars = sorted(dpll_result.keys())
            print(", ".join(f"x{var}={'T' if dpll_result[var] else 'F'}" for var in sorted_vars))
            print(f"Valid: {verify_assignment(formula, dpll_result)}")
        print(f"Statistics:")
        for stat, value in dpll.stats.items():
            print(f"  {stat}: {value}")
        
        # Exhaustive Search Analysis
        exhaustive = ExhaustiveSATSolver(debug=True)
        exhaustive_start = time.time()
        exhaustive_result = exhaustive.solve(formula)
        exhaustive_time = (time.time() - exhaustive_start) * 1000  # Convert to ms
        
        print("\nExhaustive Search Results:")
        print(f"Time: {exhaustive_time:.2f}ms")
        print(f"Result: {'Satisfiable' if exhaustive_result else 'Unsatisfiable'}")
        if exhaustive_result:
            print("Assignment:", end=" ")
            sorted_vars = sorted(exhaustive_result.keys())
            print(", ".join(f"x{var}={'T' if exhaustive_result[var] else 'F'}" for var in sorted_vars))
            print(f"Valid: {verify_assignment(formula, exhaustive_result)}")
        print(f"Statistics:")
        for stat, value in exhaustive.stats.items():
            print(f"  {stat}: {value}")
        
        # Consistency Check
        print("\nConsistency Analysis:")
        results_match = (dpll_result is None) == (exhaustive_result is None)
        print(f"Results match: {'✓' if results_match else '✗'}")
        
        if not results_match:
            print("\n⚠️ SOLVER MISMATCH DETECTED!")
            if dpll_result is None and exhaustive_result is not None:
                print("Issue: DPLL failed to find a valid solution")
                print("Valid solution from exhaustive search:")
                print(exhaustive_result)
            elif dpll_result is not None and exhaustive_result is None:
                print("Issue: DPLL found a solution but exhaustive search claims unsatisfiable")
                print("DPLL solution verification:")
                verify_assignment(formula, dpll_result)

if __name__ == "__main__":
    debug_solver(n=5, ratio=4.26, num_trials=1)