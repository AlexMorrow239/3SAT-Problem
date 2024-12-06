from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from enum import Enum
import random
import logging
from functools import wraps
import numpy as np
from collections import Counter, defaultdict
from SolverStats import create_solver_statistics

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


@dataclass(frozen=True)
class Literal:
    """Immutable representation of a literal in a Boolean formula"""
    variable: int
    is_positive: bool

    def __str__(self) -> str:
        return f"x{self.variable}" if self.is_positive else f"¬x{self.variable}"

    def negate(self) -> 'Literal':
        """Return a new literal with opposite polarity"""
        return Literal(self.variable, not self.is_positive)

@dataclass
class Clause:
    """Representation of a clause in a Boolean formula"""
    literals: List[Literal] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"({' ∨ '.join(str(lit) for lit in self.literals)})"
    
    def is_unit(self) -> bool:
        """Check if this is a unit clause (contains only one literal)"""
        return len(self.literals) == 1
    
    def get_variables(self) -> Set[int]:
        """Return set of variables in this clause"""
        return {lit.variable for lit in self.literals}

@dataclass
class Formula:
    """Representation of a Boolean formula in CNF"""
    clauses: List[Clause]
    num_variables: int
    
    def __str__(self) -> str:
        return (f"Formula with {self.num_variables} variables:\n"
                f"{' ∧ '.join(str(clause) for clause in self.clauses)}")
    
    def get_all_variables(self) -> Set[int]:
        """Return set of all variables in the formula"""
        variables = set()
        for clause in self.clauses:
            variables.update(clause.get_variables())
        return variables
    
    def calculate_variable_frequencies(self) -> Counter:
        """Calculate frequency of each variable in the formula"""
        frequencies = Counter()
        for clause in self.clauses:
            for literal in clause.literals:
                frequencies[literal.variable] += 1
        return frequencies

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

class FormulaSimplifier:
    """Utility class for formula simplification operations"""
    
    @staticmethod
    def simplify_formula(formula: Formula, assignment: Dict[int, bool]) -> Formula:
        """Simplify formula based on partial assignment"""
        new_clauses = []
        
        for clause in formula.clauses:
            new_clause = FormulaSimplifier._simplify_clause(clause, assignment)
            if new_clause is None:  # Clause is satisfied
                continue
            if not new_clause.literals:  # Empty clause (contradiction)
                return Formula([Clause()], formula.num_variables)
            new_clauses.append(new_clause)
        
        return Formula(new_clauses, formula.num_variables)
    
    @staticmethod
    def _simplify_clause(clause: Clause, assignment: Dict[int, bool]) -> Optional[Clause]:
        """
        Simplify a clause based on partial assignment.
        Returns None if clause is satisfied, new simplified clause otherwise.
        """
        new_literals = []
        for lit in clause.literals:
            if lit.variable in assignment:
                if lit.is_positive == assignment[lit.variable]:
                    return None  # Clause is satisfied
            else:
                new_literals.append(lit)
        return Clause(new_literals)

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

class RandomFormulaGenerator:
    """Generator for random 3SAT formulas"""
    
    @staticmethod
    def generate(num_variables: int, num_clauses: int, seed: Optional[int] = None) -> Formula:
        """Generate random 3SAT formula with given parameters"""
        if seed is not None:
            random.seed(seed)
            
        if num_variables < 3:
            raise ValueError("Number of variables must be at least 3 for 3SAT")
        
        clauses = []
        for _ in range(num_clauses):
            # Select 3 distinct variables
            vars_selected = random.sample(range(1, num_variables + 1), 3)
            # Randomly decide polarity for each variable
            literals = [
                Literal(var, random.choice([True, False]))
                for var in vars_selected
            ]
            clauses.append(Clause(literals))
            
        return Formula(clauses, num_variables)
    
    @staticmethod
    def generate_phase_transition_formulas(
        num_variables: int,
        ratio_range: Tuple[float, float],
        num_ratios: int,
        formulas_per_ratio: int
    ) -> Dict[float, List[Formula]]:
        """
        Generate multiple formulas around the phase transition point
        
        Args:
            num_variables: Number of variables in each formula
            ratio_range: (min_ratio, max_ratio) for clause/variable ratio
            num_ratios: Number of different ratios to test
            formulas_per_ratio: Number of formulas to generate for each ratio
            
        Returns:
            Dictionary mapping ratios to lists of formulas
        """
        ratios = np.linspace(ratio_range[0], ratio_range[1], num_ratios)
        formulas = defaultdict(list)
        
        for ratio in ratios:
            num_clauses = int(num_variables * ratio)
            for _ in range(formulas_per_ratio):
                formula = RandomFormulaGenerator.generate(num_variables, num_clauses)
                formulas[ratio].append(formula)
                
        return formulas

class SATAnalyzer:
    """Utility class for analyzing SAT solver performance"""
    
    def __init__(self):
        self.results: Dict[Tuple[int, float], List[Dict[str, Any]]] = defaultdict(list)
    
    def run_analysis(
        self,
        var_sizes: List[int],
        ratios: List[float],
        trials_per_config: int,
        solvers: List[SATSolver]
    ) -> None:
        """Run comprehensive analysis with multiple solvers"""
        for n in var_sizes:
            for ratio in ratios:
                for _ in range(trials_per_config):
                    self._run_single_trial(n, ratio, solvers)
    
    def _run_single_trial(
        self,
        n: int,
        ratio: float,
        solvers: List[SATSolver]
    ) -> None:
        """Run a single trial with all solvers"""
        generator = RandomFormulaGenerator()
        formula = generator.generate(n, int(n * ratio))
        
        results = {}
        for solver in solvers:
            solver_name = solver.__class__.__name__
            
            # Just get the result, time is tracked in stats
            result = solver.solve(formula)
            
            # Get timing from stats
            results[f"{solver_name}_time"] = solver.stats.solving_time_ms.value
            results[f"{solver_name}_sat"] = result is not None
            
            # Get other solver statistics
            for stat_name, stat in solver.stats.stats.items():
                results[f"{solver_name}_{stat_name}"] = stat.value
            
            self.results[(n, ratio)].append(results) 

    def _get_solver_stats(self, solver: SATSolver) -> Dict[str, Any]:
        """Extract relevant statistics from solver"""
        solver_name = solver.__class__.__name__
        stats = {}
        
        for stat_name, value in vars(solver.stats).items():
            if not stat_name.startswith('_'):
                stats[f"{solver_name}_{stat_name}"] = value
        
        return stats

# Example usage
if __name__ == "__main__":
    # Create solvers
    dpll_solver = DPLLSolver()
    random_solver = RandomSATSolver()
    exhaustive_solver = ExhaustiveSATSolver()
    
    # Create analyzer
    analyzer = SATAnalyzer()
    
    # Run analysis
    var_sizes = [5, 10, 15]
    ratios = [3.0, 4.0, 4.26, 4.5, 5.0]
    analyzer.run_analysis(
        var_sizes=var_sizes,
        ratios=ratios,
        trials_per_config=5,
        solvers=[dpll_solver, random_solver, exhaustive_solver]
    )