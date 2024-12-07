from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
from enum import Enum
import random
import logging
from SolverStats import create_solver_statistics
from Utilities import Literal, Clause, Formula, FormulaSimplifier
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger('SATSolver')

class SolverStrategy(Enum):
    """Enum for different solver strategies"""
    DPLL = "dpll"
    RANDOM = "random"
    EXHAUSTIVE = "exhaustive"

class SATSolver(ABC):
    """Abstract base class for SAT solvers"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging based on debug setting"""
        self.logger = logging.getLogger(f'SATSolver.{self.__class__.__name__}')
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
    
    @abstractmethod
    def solve(self, formula: Formula) -> Optional[Dict[int, bool]]:
        """
        Solve the SAT formula and return an assignment if satisfiable.
        Returns None if unsatisfiable.
        """
        pass
    
    def verify_solution(self, formula: Formula, assignment: Dict[int, bool]) -> bool:
        """Verify if an assignment satisfies the formula"""
        if not assignment:
            return False
            
        formula_vars = formula.get_all_variables()
        if not formula_vars.issubset(assignment.keys()):
            self.logger.warning("Incomplete assignment provided")
            return False
        
        for clause in formula.clauses:
            if not self._verify_clause(clause, assignment):
                return False
        
        return True
    
    def _verify_clause(self, clause: Clause, assignment: Dict[int, bool]) -> bool:
        """Verify if a clause is satisfied by the assignment"""
        return any(
            (lit.is_positive == assignment[lit.variable])
            for lit in clause.literals
        )

class DPLLSolver(SATSolver):
    """DPLL-based SAT solver implementation"""
    
    def __init__(self, debug: bool = False):
        super().__init__(debug)
        self.stats = create_solver_statistics("dpll")
        self._current_depth = 0
    
    def solve(self, formula: Formula) -> Optional[Dict[int, bool]]:
        """Solve using the DPLL algorithm"""
        self.stats.reset()
        self.stats.start_timer()  # Start timing
        try:
            result = self._dpll(formula, {})
            if result is not None:
                self.stats.successful_solves.value += 1
            else:
                self.stats.failed_solves.value += 1
            return result
        finally:
            self.stats.stop_timer()  # Stop timing in finally block to ensure it happens
    
    def _dpll(self, formula: Formula, assignments: Dict[int, bool]) -> Optional[Dict[int, bool]]:
        """Core DPLL recursive implementation"""
        if self.debug:
            self.logger.debug(f"DPLL called with {len(assignments)} assignments")
        
        self._current_depth += 1
        self.stats.append("decision_depths", self._current_depth)
        
        # Base cases
        if not formula.clauses:
            self._current_depth -= 1
            return self._complete_assignment(assignments, formula.num_variables)
        
        if any(not clause.literals for clause in formula.clauses):
            self.stats.increment("backtracks")
            self._current_depth -= 1
            return None
        
        # Unit propagation
        unit_clause = next((clause for clause in formula.clauses if clause.is_unit()), None)
        if unit_clause:
            self.stats.increment("unit_propagations")
            lit = unit_clause.literals[0]
            new_assignments = assignments.copy()
            new_assignments[lit.variable] = lit.is_positive
            
            simplified_formula = FormulaSimplifier.simplify_formula(formula, new_assignments)
            self.stats.append("clause_sizes", len(simplified_formula.clauses))
            
            return self._dpll(simplified_formula, new_assignments)
        
        # Pure literal elimination
        pure_literal = self._find_pure_literal(formula)
        if pure_literal:
            self.stats.increment("pure_literals")
            new_assignments = assignments.copy()
            new_assignments[pure_literal.variable] = pure_literal.is_positive
            
            return self._dpll(
                FormulaSimplifier.simplify_formula(formula, new_assignments),
                new_assignments
            )

        # Two-clause rule
        two_clause_result = self._apply_two_clause_rule(formula)
        if two_clause_result:
            self.stats.increment("two_clause_applications")
            var, value = two_clause_result
            new_assignments = assignments.copy()
            new_assignments[var] = value
            
            return self._dpll(
                FormulaSimplifier.simplify_formula(formula, new_assignments),
                new_assignments
            )


        # Variable selection
        var = self._choose_next_variable(formula)
        self.stats.append("variable_frequencies", var)
        
        # Try assignments
        for value in [True, False]:
            new_assignments = assignments.copy()
            new_assignments[var] = value
            result = self._dpll(
                FormulaSimplifier.simplify_formula(formula, new_assignments),
                new_assignments
            )
            if result is not None:
                self._current_depth -= 1
                return result
        
        self.stats.increment("backtracks")
        self._current_depth -= 1
        return None

    def _find_pure_literal(self, formula: Formula) -> Optional[Literal]:
        """Find a pure literal in the formula"""
        positive_vars = set()
        negative_vars = set()
        
        for clause in formula.clauses:
            for lit in clause.literals:
                if lit.is_positive:
                    positive_vars.add(lit.variable)
                else:
                    negative_vars.add(lit.variable)
        
        pure_positive = positive_vars - negative_vars
        if pure_positive:
            return Literal(min(pure_positive), True)
            
        pure_negative = negative_vars - positive_vars
        if pure_negative:
            return Literal(min(pure_negative), False)
            
        return None

    def _apply_two_clause_rule(self, formula: Formula) -> Optional[Tuple[int, bool]]:
        """
        Apply the two-clause rule to find forced assignments.
        Returns (variable, value) tuple if a forced assignment is found, None otherwise.
        
        The two-clause rule looks for:
        1. (x ∨ y) ∧ (x ∨ ¬y) → x must be True
        2. (¬x ∨ y) ∧ (¬x ∨ ¬y) → x must be False
        """
        # First, collect all binary clauses
        binary_clauses = [clause for clause in formula.clauses if len(clause.literals) == 2]
        
        # Create a map of variable to its clauses for efficient lookup
        var_to_clauses = defaultdict(list)
        for clause in binary_clauses:
            for lit in clause.literals:
                var_to_clauses[lit.variable].append(clause)
        
        # Check each variable
        for var in var_to_clauses:
            # Get all binary clauses containing this variable
            relevant_clauses = var_to_clauses[var]
            if len(relevant_clauses) < 2:
                continue
                
            # Look for clauses that force an assignment
            pos_clauses = []  # Clauses where var appears positively
            neg_clauses = []  # Clauses where var appears negatively
            
            for clause in relevant_clauses:
                for lit in clause.literals:
                    if lit.variable == var:
                        if lit.is_positive:
                            pos_clauses.append(clause)
                        else:
                            neg_clauses.append(clause)
            
            # Check for forcing patterns
            force_true = self._check_forcing_pattern(pos_clauses, var, True)
            force_false = self._check_forcing_pattern(neg_clauses, var, False)
            
            if force_true:
                return (var, True)
            if force_false:
                return (var, False)
        
        return None
    
    def _check_forcing_pattern(self, clauses: List[Clause], var: int, is_positive: bool) -> bool:
        """
        Check if a set of clauses forces a variable assignment.
        Returns True if the variable is forced to the specified polarity.
        """
        if len(clauses) < 2:
            return False
            
        # For each pair of clauses
        for i, clause1 in enumerate(clauses):
            for clause2 in clauses[i + 1:]:
                # Get the other literals in each clause
                other_lit1 = next(lit for lit in clause1.literals if lit.variable != var)
                other_lit2 = next(lit for lit in clause2.literals if lit.variable != var)
                
                # Check if they form a forcing pattern
                if (other_lit1.variable == other_lit2.variable and 
                    other_lit1.is_positive != other_lit2.is_positive):
                    return True
        
        return False

    def _choose_next_variable(self, formula: Formula) -> int:
        """Choose next variable based on frequency"""
        frequencies = formula.calculate_variable_frequencies()
        return max(frequencies.items(), key=lambda x: x[1])[0]
    
    def _complete_assignment(self, partial: Dict[int, bool], num_vars: int) -> Dict[int, bool]:
        """Complete a partial assignment with True values"""
        complete = partial.copy()
        for var in range(1, num_vars + 1):
            if var not in complete:
                complete[var] = True
        self.stats.variable_assignments.value.append(complete)
        return complete

class RandomSATSolver(SATSolver):
    """Implementation of Random Walk SAT solver"""
    
    def __init__(self, max_flips: int = 100, max_tries: int = 100, debug: bool = False):
        super().__init__(debug)
        self.max_flips = max_flips
        self.max_tries = max_tries
        self.stats = create_solver_statistics("random")
    
    def solve(self, formula: Formula) -> Optional[Dict[int, bool]]:
        """Solve using random walk strategy"""
        self.stats.reset()
        self.stats.start_timer()
        
        try:
            for _ in range(self.max_tries):
                self.stats.increment("restart_count")
                assignment = self._generate_random_assignment(formula.num_variables)
                result = self._random_walk(formula, assignment)
                
                if result is not None:
                    self.stats.successful_solves.value += 1
                    return result
            
            self.stats.failed_solves.value += 1
            return None
        finally:
            self.stats.stop_timer()
    
    def _random_walk(self, formula: Formula, assignment: Dict[int, bool]) -> Optional[Dict[int, bool]]:
        """Perform random walk from initial assignment"""
        current_assignment = assignment.copy()
        prev_unsat_count = float('inf')
        
        for _ in range(self.max_flips):
            self.stats.increment("total_flips")
            
            if self.verify_solution(formula, current_assignment):
                return current_assignment
            
            # Count unsatisfied clauses
            unsatisfied = [
                clause for clause in formula.clauses
                if not self._verify_clause(clause, current_assignment)
            ]
            unsat_count = len(unsatisfied)
            
            self.stats.append("unsatisfied_clauses", unsat_count)
            
            if unsat_count < prev_unsat_count:
                self.stats.increment("successful_flips")
                self.stats.append("flip_improvements", prev_unsat_count - unsat_count)
            elif unsat_count > prev_unsat_count:
                self.stats.increment("local_minima")
            
            prev_unsat_count = unsat_count
            
            if not unsatisfied:
                return current_assignment
            
            # Flip a random variable from a random unsatisfied clause
            clause = random.choice(unsatisfied)
            literal = random.choice(clause.literals)
            current_assignment[literal.variable] = not current_assignment[literal.variable]
        
        return None
    
    def _generate_random_assignment(self, num_variables: int) -> Dict[int, bool]:
        """Generate random initial assignment"""
        assignment = {
            var: random.choice([True, False])
            for var in range(1, num_variables + 1)
        }
        self.stats.variable_assignments.value.append(assignment)
        return assignment

class ExhaustiveSATSolver(SATSolver):
    """Implementation of Exhaustive Search SAT solver"""
    
    def __init__(self, debug: bool = False):
        super().__init__(debug)
        self.stats = create_solver_statistics("exhaustive")
        self._current_depth = 0
    
    def solve(self, formula: Formula) -> Optional[Dict[int, bool]]:
        self.stats.reset()
        self.stats.start_timer()
        try:
            self.current_assignment: Dict[int, bool] = {}
            if self._exhaustive_search(formula, 1):
                self.stats.successful_solves.value += 1
                return self.current_assignment
            self.stats.failed_solves.value += 1
            return None
        finally:
            self.stats.stop_timer()
            
    def _exhaustive_search(self, formula: Formula, current_var: int) -> bool:
        """Recursive exhaustive search implementation"""
        self.stats.increment("nodes_visited")
        self._current_depth += 1
        self.stats.append("branch_depths", self._current_depth)
        
        if current_var > formula.num_variables:
            self.stats.increment("assignments_tested")
            is_sat = self.verify_solution(formula, self.current_assignment)
            if is_sat:
                self.stats.append("satisfying_depths", self._current_depth)
                self.stats.variable_assignments.value.append(self.current_assignment.copy())
            self._current_depth -= 1
            return is_sat
        
        # Try both assignments for current variable
        for value in [False, True]:
            self.current_assignment[current_var] = value
            self.stats.increment("partial_validations")
            
            if self._exhaustive_search(formula, current_var + 1):
                self._current_depth -= 1
                return True
        
        del self.current_assignment[current_var]
        self._current_depth -= 1
        return False