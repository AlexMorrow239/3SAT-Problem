from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter
from collections import defaultdict
import random
import numpy as np

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
        """Generate multiple formulas around the phase transition point"""
        ratios = np.linspace(ratio_range[0], ratio_range[1], num_ratios)
        formulas = defaultdict(list)
        
        for ratio in ratios:
            num_clauses = int(num_variables * ratio)
            for _ in range(formulas_per_ratio):
                formula = RandomFormulaGenerator.generate(num_variables, num_clauses)
                formulas[ratio].append(formula)
                
        return formulas