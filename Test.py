from SAT import Literal, Clause, Formula, SATSolver


def test_solver():
    # Create a simple unsatisfiable formula: (x₁) ∧ (¬x₁)
    formula1 = Formula([
        Clause([Literal(1, True)]),
        Clause([Literal(1, False)])
    ], 1)
    
    # Create a simple satisfiable formula: (x₁ ∨ x₂) ∧ (¬x₁ ∨ x₂)
    formula2 = Formula([
        Clause([Literal(1, True), Literal(2, True)]),
        Clause([Literal(1, False), Literal(2, True)])
    ], 2)
    
    solver = SATSolver(debug=True)
    
    print("\nTesting unsatisfiable formula:")
    print(formula1)
    result1 = solver.solve_dpll(formula1)
    print(f"Result (should be None): {result1}")
    print(f"Stats: {solver.stats}")
    
    print("\nTesting satisfiable formula:")
    print(formula2)
    result2 = solver.solve_dpll(formula2)
    print(f"Result (should have x₂=True): {result2}")
    print(f"Stats: {solver.stats}")

    
if __name__ == '__main__':
    test_solver()