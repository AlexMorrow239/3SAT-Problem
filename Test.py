from SAT import Literal, Clause, Formula, DPLLSolver


def test_solver():
    # Create a simple unsatisfiable formula: (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ ¬x2 ∨ ¬x3)
    clause1_assignments = [True, True, True]
    clause2_assignments = [False, True, True]
    clause3_assignments = [True, False, True]
    clause4_assignments = [True, True, False]
    clause5_assignments = [False, False, True]
    clause6_assignments = [False, True, False]
    clause7_assignments = [True, False, False]
    clause8_assignments = [False, False, False]



    formula1 = Formula([
        Clause([Literal(x, assignment) for x, assignment in enumerate(clause1_assignments, 1)]),
        Clause([Literal(x, assignment) for x, assignment in enumerate(clause2_assignments, 1)]),
        Clause([Literal(x, assignment) for x, assignment in enumerate(clause3_assignments, 1)]),
        Clause([Literal(x, assignment) for x, assignment in enumerate(clause4_assignments, 1)]),
        Clause([Literal(x, assignment) for x, assignment in enumerate(clause5_assignments, 1)]),
        Clause([Literal(x, assignment) for x, assignment in enumerate(clause6_assignments, 1)]),
        Clause([Literal(x, assignment) for x, assignment in enumerate(clause7_assignments, 1)]),
        Clause([Literal(x, assignment) for x, assignment in enumerate(clause8_assignments, 1)]),

    ], 1)
    
    # Create a simple satisfiable formula: 
    clause1_assignments = [False, True, False]
    clause2_assignments = [True, False, True]
    formula2 = Formula([
        Clause([Literal(x, True) for x in range(1, 4)]),
        Clause([Literal(x, assignment) for x, assignment in enumerate(clause1_assignments, 1)]),
        Clause([Literal(x, assignment) for x, assignment in enumerate(clause2_assignments, 1)]
        )
    ], 2)
    
    solver = DPLLSolver(debug=True)
    
    print("\nTesting unsatisfiable formula:")
    print(formula1)
    result1 = solver.solve(formula1)
    print(f"Result (should be None): {result1}")
    print(f"Stats: {solver.stats}")
    
    print("\nTesting satisfiable formula:")
    print(formula2)
    result2 = solver.solve(formula2)
    print(f"Result (should have 1: True, 2: True): {result2}")
    print(f"Stats: {solver.stats}")

    
if __name__ == '__main__':
    test_solver()