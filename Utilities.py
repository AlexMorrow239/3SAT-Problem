import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from numbers import Real
from typing import (
    Dict,
    Final,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np


class FormulaError(Exception):
    """Custom exception class for formula-related errors"""

    pass


class LiteralPolarity(Enum):
    """Enumeration for literal polarity"""

    POSITIVE = auto()
    NEGATIVE = auto()


class ClauseState(Enum):
    """Enumeration for clause evaluation states"""

    SATISFIED = auto()
    UNSATISFIED = auto()
    UNDETERMINED = auto()


NumericType = Union[int, float]
Assignment = Dict[int, bool]
VariableSet = Set[int]


@dataclass(frozen=True)
class Literal:
    """Immutable representation of a literal in a Boolean formula"""

    variable: int
    is_positive: bool

    def __post_init__(self) -> None:
        """Validate literal after initialization"""
        if not isinstance(self.variable, int):
            raise FormulaError(
                f"Variable must be an integer, got {type(self.variable)}"
            )
        if self.variable < 1:
            raise FormulaError(f"Variable must be positive, got {self.variable}")
        if not isinstance(self.is_positive, bool):
            raise FormulaError(
                f"is_positive must be a boolean, got {type(self.is_positive)}"
            )

    def __str__(self) -> str:
        return f"x{self.variable}" if self.is_positive else f"¬x{self.variable}"

    def negate(self) -> "Literal":
        """Return a new literal with opposite polarity"""
        return Literal(self.variable, not self.is_positive)

    @property
    def polarity(self) -> LiteralPolarity:
        """Get the polarity of the literal"""
        return (
            LiteralPolarity.POSITIVE if self.is_positive else LiteralPolarity.NEGATIVE
        )


@dataclass
class Clause:
    """Representation of a clause in a Boolean formula"""

    literals: List[Literal] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate clause after initialization"""
        if not isinstance(self.literals, list):
            raise FormulaError(f"Literals must be a list, got {type(self.literals)}")
        for literal in self.literals:
            if not isinstance(literal, Literal):
                raise FormulaError(
                    f"Each literal must be a Literal instance, got {type(literal)}"
                )

    def __str__(self) -> str:
        return f"({' ∨ '.join(str(lit) for lit in self.literals)})"

    def __len__(self) -> int:
        return len(self.literals)

    def __iter__(self) -> Iterator[Literal]:
        return iter(self.literals)

    def is_unit(self) -> bool:
        """Check if this is a unit clause (contains only one literal)"""
        return len(self.literals) == 1

    def get_variables(self) -> VariableSet:
        """Return set of variables in this clause"""
        return {lit.variable for lit in self.literals}

    def evaluate(self, assignment: Assignment) -> Optional[bool]:
        """
        Evaluate clause under a partial assignment.
        Returns None if clause cannot be evaluated yet.
        """
        unassigned: bool = False
        for literal in self.literals:
            if literal.variable in assignment:
                if literal.is_positive == assignment[literal.variable]:
                    return True
            else:
                unassigned = True
        return None if unassigned else False


@dataclass
class Formula:
    """Representation of a Boolean formula in CNF"""

    clauses: List[Clause]
    num_variables: int

    def __post_init__(self) -> None:
        """Validate formula after initialization"""
        if not isinstance(self.clauses, list):
            raise FormulaError(f"Clauses must be a list, got {type(self.clauses)}")
        for clause in self.clauses:
            if not isinstance(clause, Clause):
                raise FormulaError(
                    f"Each clause must be a Clause instance, got {type(clause)}"
                )

        if not isinstance(self.num_variables, int):
            raise FormulaError(
                f"Number of variables must be an integer, got {type(self.num_variables)}"
            )
        if self.num_variables < 1:
            raise FormulaError(
                f"Number of variables must be positive, got {self.num_variables}"
            )

        all_vars = self.get_all_variables()
        if all_vars and max(all_vars) > self.num_variables:
            raise FormulaError(
                f"Formula contains variable(s) beyond declared number: {max(all_vars)} > {self.num_variables}"
            )

    def __str__(self) -> str:
        return (
            f"Formula with {self.num_variables} variables:\n"
            f"{' ∧ '.join(str(clause) for clause in self.clauses)}"
        )

    def __len__(self) -> int:
        return len(self.clauses)

    def __iter__(self) -> Iterator[Clause]:
        return iter(self.clauses)

    def get_all_variables(self) -> VariableSet:
        """Return set of all variables in the formula"""
        return {lit.variable for clause in self.clauses for lit in clause.literals}

    def calculate_variable_frequencies(self) -> Counter:
        """Calculate frequency of each variable in the formula"""
        return Counter(
            lit.variable for clause in self.clauses for lit in clause.literals
        )

    def evaluate(self, assignment: Assignment) -> Optional[bool]:
        """
        Evaluate formula under a partial assignment.
        Returns None if formula cannot be evaluated yet.
        """
        results: List[Optional[bool]] = [
            clause.evaluate(assignment) for clause in self.clauses
        ]
        if None in results:
            return None
        return all(cast(bool, result) for result in results)


class FormulaSimplifier:
    """Utility class for formula simplification operations"""

    @staticmethod
    def simplify_formula(formula: Formula, assignment: Assignment) -> Formula:
        """Simplify formula based on partial assignment"""
        if not isinstance(formula, Formula):
            raise FormulaError(f"Expected Formula instance, got {type(formula)}")
        if not isinstance(assignment, dict):
            raise FormulaError(
                f"Assignment must be a dictionary, got {type(assignment)}"
            )

        for var, val in assignment.items():
            if not isinstance(var, int):
                raise FormulaError(f"Assignment keys must be integers, got {type(var)}")
            if not isinstance(val, bool):
                raise FormulaError(
                    f"Assignment values must be booleans, got {type(val)}"
                )
            if var < 1:
                raise FormulaError(f"Variable numbers must be positive, got {var}")
            if var > formula.num_variables:
                raise FormulaError(
                    f"Assignment contains variable {var} beyond formula's declared number {formula.num_variables}"
                )

        try:
            new_clauses: List[Clause] = []
            for clause in formula.clauses:
                new_clause = FormulaSimplifier._simplify_clause(clause, assignment)
                if new_clause is None:  # Clause is satisfied
                    continue
                if not new_clause.literals:  # Empty clause (contradiction)
                    return Formula([Clause()], formula.num_variables)
                new_clauses.append(new_clause)

            return Formula(new_clauses, formula.num_variables)
        except Exception as e:
            raise FormulaError(f"Error during formula simplification: {str(e)}")

    @staticmethod
    def _simplify_clause(clause: Clause, assignment: Assignment) -> Optional[Clause]:
        """Simplify a clause based on partial assignment"""
        if not isinstance(clause, Clause):
            raise FormulaError(f"Expected Clause instance, got {type(clause)}")

        new_literals: List[Literal] = []
        for lit in clause.literals:
            if lit.variable in assignment:
                if lit.is_positive == assignment[lit.variable]:
                    return None  # Clause is satisfied
            else:
                new_literals.append(lit)
        return Clause(new_literals)


class RandomFormulaGenerator:
    """Generator for random 3SAT formulas"""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize generator with optional seed"""
        self.rng: random.Random = random.Random(seed)

    def generate(self, num_variables: int, num_clauses: int) -> Formula:
        """Generate random 3SAT formula with given parameters"""
        if not isinstance(num_variables, int) or num_variables < 3:
            raise FormulaError(
                f"Number of variables must be an integer ≥ 3, got {num_variables}"
            )
        if not isinstance(num_clauses, int) or num_clauses < 1:
            raise FormulaError(
                f"Number of clauses must be a positive integer, got {num_clauses}"
            )

        try:
            clauses: List[Clause] = []
            for _ in range(num_clauses):
                # Select 3 distinct variables
                vars_selected: List[int] = self.rng.sample(
                    range(1, num_variables + 1), 3
                )
                # Randomly decide polarity for each variable
                literals: List[Literal] = [
                    Literal(var, self.rng.choice([True, False]))
                    for var in vars_selected
                ]
                clauses.append(Clause(literals))

            return Formula(clauses, num_variables)
        except Exception as e:
            raise FormulaError(f"Error generating formula: {str(e)}")

    def generate_phase_transition_formulas(
        self,
        num_variables: int,
        ratio_range: Tuple[float, float],
        num_ratios: int,
        formulas_per_ratio: int,
    ) -> Dict[float, List[Formula]]:
        """Generate multiple formulas around the phase transition point"""
        if not isinstance(num_variables, int) or num_variables < 3:
            raise FormulaError(
                f"Number of variables must be an integer ≥ 3, got {num_variables}"
            )
        if not isinstance(ratio_range, tuple) or len(ratio_range) != 2:
            raise FormulaError("ratio_range must be a tuple of (min, max)")
        if not all(isinstance(r, Real) for r in ratio_range):
            raise FormulaError("Ratio values must be numeric")
        if ratio_range[0] >= ratio_range[1]:
            raise FormulaError(
                f"Invalid ratio range: {ratio_range[0]} >= {ratio_range[1]}"
            )
        if not isinstance(num_ratios, int) or num_ratios < 1:
            raise FormulaError(
                f"num_ratios must be a positive integer, got {num_ratios}"
            )
        if not isinstance(formulas_per_ratio, int) or formulas_per_ratio < 1:
            raise FormulaError(
                f"formulas_per_ratio must be a positive integer, got {formulas_per_ratio}"
            )

        try:
            ratios: np.ndarray = np.linspace(ratio_range[0], ratio_range[1], num_ratios)
            formulas: Dict[float, List[Formula]] = defaultdict(list)

            for ratio in ratios:
                num_clauses = int(num_variables * ratio)
                for _ in range(formulas_per_ratio):
                    formula = self.generate(num_variables, num_clauses)
                    formulas[float(ratio)].append(formula)

            return formulas
        except Exception as e:
            raise FormulaError(f"Error generating phase transition formulas: {str(e)}")
