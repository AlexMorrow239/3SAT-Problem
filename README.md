# Analysis Report: 3SAT Solver Implementation and Results

## Executive Summary

This report analyzes the implementation and performance of different 3SAT solver approaches, focusing on the DPLL (Davis-Putnam-Logemann-Loveland) algorithm and exhaustive search. The analysis examines solver behavior across various problem sizes and clause-to-variable ratios, with particular attention to the phase transition phenomenon around the critical ratio of 4.26.

## Code Implementation Overview

### 1. Base Data Structures

```python
@dataclass
class Literal:
    variable: int
    is_positive: bool

@dataclass
class Clause:
    literals: List[Literal]

@dataclass
class Formula:
    clauses: List[Clause]
    num_variables: int
```

These foundational classes provide the basic building blocks for representing 3SAT formulas:

- `Literal`: Represents a variable or its negation
- `Clause`: Groups literals in disjunctive (OR) relationships
- `Formula`: Combines clauses in conjunctive (AND) relationships

### 2. Abstract Solver Base Class

```python
class SATSolver(ABC):
    @abstractmethod
    def solve(self, formula: Formula) -> Optional[Dict[int, bool]]
    
    def _verify_assignment(self, formula: Formula, assignment: Dict[int, bool]) -> bool
    def _complete_assignment(self, partial: Dict[int, bool], num_vars: int) -> Dict[int, bool]
```

The abstract base class defines:

- Common interface for all solvers
- Shared utility methods for assignment verification
- Statistics tracking infrastructure

## Solver Implementations

### 1. DPLL Solver

```python
class DPLLSolver(SATSolver):
    def solve(self, formula: Formula) -> Dict[int, bool]
    def _dpll(self, formula: Formula, assignments: Dict[int, bool]) -> Optional[Dict[int, bool]]
    def _simplify_formula(self, formula: Formula, assignments: Dict[int, bool]) -> Formula
```

Key components:

1. **Main Solving Logic**
   - Implements recursive DPLL algorithm
   - Maintains assignment state
   - Handles backtracking

2. **Formula Simplification**
   - Removes satisfied clauses
   - Updates remaining clauses
   - Detects contradictions

3. **Variable Selection**
   - Frequency-based heuristic
   - Unit clause detection
   - Pure literal elimination

### 2. Exhaustive Solver

```python
class ExhaustiveSATSolver(SATSolver):
    def solve(self, formula: Formula) -> Dict[int, bool]
    def _exhaustive_search(self, formula: Formula, current_var: int) -> bool
    def _evaluate_formula(self, formula: Formula) -> bool
```

Components:

1. **Search Implementation**
   - Systematic variable assignment
   - Complete space exploration
   - Early termination on solution

2. **Formula Evaluation**
   - Direct clause evaluation
   - Complete assignment verification
   - Statistics collection

## Analysis Framework

### 1. Random Formula Generator

```python
class RandomFormulaGenerator:
    @staticmethod
    def generate(num_variables: int, num_clauses: int) -> Formula
```

Features:

- Uniform clause distribution
- Variable distinctness enforcement
- Configurable formula size

### 2. Analysis Tools

#### Performance Analysis

```python
def compare_solvers(n: int, ratio: float, results: dict):
    # Timing and comparison logic
    # Results collection
```

Components:

- Timing measurements
- Statistics collection
- Result verification

#### Visualization

```python
def plot_results(results: dict):
    # Satisfiability analysis
    # Time complexity plots
    # Statistical summaries
```

Features:

- Phase transition visualization
- Performance comparison plots
- Statistical analysis

## Testing Infrastructure

### 1. Debug Tools

```python
def debug_solver(n: int, ratio: float, num_trials: int = 1):
    # Detailed solver output
    # Step-by-step tracking
    # Result verification
```

Components:

1. **Verification Functions**
   - Assignment checking
   - Result consistency
   - Performance monitoring

2. **Logging System**

   ```python
   logging.basicConfig(level=logging.DEBUG, 
                      format='%(levelname)s - %(message)s')
   ```

   - Detailed execution tracking
   - Performance profiling
   - Error detection

### 2. Test Cases

```python
def test_solver():
    # Simple satisfiable/unsatisfiable cases
    # Edge case testing
    # Performance benchmarks
```

## File Organization

1. **Main Implementation Files**
   - `SAT.py`: Core solver implementations
   - `Test.py`: Basic test cases
   - `Debug.py`: Debugging utilities

2. **Analysis Files**
   - `Full Analysis.py`: Comprehensive analysis
   - `SimpleAnalysis.py`: Basic performance testing

## Code Flow

1. **Formula Generation**

   ```python
   generator = RandomFormulaGenerator()
   formula = generator.generate(n, int(n * ratio))
   ```

2. **Solver Execution**

   ```python
   solver = DPLLSolver()
   result = solver.solve(formula)
   ```

3. **Result Analysis**

   ```python
   compare_solvers(n, ratio, results)
   plot_results(results)
   ```

## Analysis Methodology

The analysis framework employed several key approaches:

1. **Parameter Space Exploration**
   - Variable sizes (n):

        DPLL and Exhastive search are both run for small n, only DPLL is run for larger values of n because exhastive search preforms significantly worse asymptotically.

   - Clause-to-variable ratios:

        Ensure testing of ratios both before and after the critical point of 4.26.

   - Multiple trials per configuration:

        I included multiple trials at each configuration to analyze the satisfiability rates. Additionally, this improves the statistical significance of my other metrics by taking the mean values instead of absolute value from a single trial at each configuration.

2. **Performance Metrics**
   - Solution time (milliseconds)
   - Satisfiability rate
   - Backtrack count (DPLL)
   - Node visits (Exhaustive)

        Backtrack count and Node visits helps highlight the disparity in asymptotic time complexity between the 2 algorithms for solving this problem

3. **Visualization Methods**
   - Phase transition analysis plots
   - Time complexity analysis
   - Satisfiability rate heatmaps

## Key Findings

### 1. Phase Transition Behavior

The analysis confirms the theoretical phase transition at m/n ≈ 4.26:

- Clear transition from mostly satisfiable to mostly unsatisfiable

- Sharp drop in satisfiability probability around the critical ratio
- Maximum solving time complexity near the transition point

### 2. Algorithm Performance

DPLL shows significant advantages over exhaustive search:

- Orders of magnitude faster for all problem sizes

- More pronounced advantage as problem size increases
- Particularly effective for satisfiable instances

### 3. Scaling Characteristics

Both algorithms exhibit expected scaling patterns:

- Exponential time growth with problem size

- More pronounced scaling near phase transition
- DPLL maintains better scaling characteristics throughout

### 4. Statistical Reliability

The implementation shows robust behavior:

- Consistent results across multiple trials

- Agreement between DPLL and exhaustive search
- Expected variance patterns in solving times

## Conclusion

The implementation successfully demonstrates the key characteristics of 3SAT solving, particularly the phase transition phenomenon and the effectiveness of the DPLL algorithm. The analysis framework provides valuable insights into solver behavior and performance characteristics, while also validating theoretical predictions about phase transitions in random 3SAT instances.

The results confirm that the DPLL implementation is both correct and efficient, showing expected behavior across all tested parameters. The phase transition analysis aligns well with theoretical predictions, and the performance characteristics match expected complexity bounds.

Future work should focus on implementing modern SAT solving techniques such as clause learning and non-chronological backtracking to further improve performance on larger instances.
