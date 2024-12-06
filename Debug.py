import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Type
from dataclasses import dataclass
import time
from SAT import DPLLSolver, RandomSATSolver, ExhaustiveSATSolver, RandomFormulaGenerator

@dataclass
class DebugResult:
    """Store results from a single debug test"""
    n: int
    ratio: float
    solver_name: str
    solving_time: float
    is_correct: bool
    is_satisfiable: bool
    solver_stats: Dict[str, Any]

class DebugVisualizer:
    def __init__(self):
        self.results: List[DebugResult] = []
        self.palette = sns.color_palette("husl", 3)
        plt.style.use('seaborn-v0_8')
    
    def add_result(self, result: DebugResult):
        """Add a debug result to the collection"""
        self.results.append(result)
    
    def plot_results(self):
        """Generate comprehensive visualization of debug results"""
        if not self.results:
            print("No results to visualize")
            return
        
        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame([vars(r) for r in self.results])
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Success Rate Heatmap
        plt.subplot(2, 2, 1)
        self._plot_success_rate_heatmap(df)
        
        # 2. Solving Time Distribution
        plt.subplot(2, 2, 2)
        self._plot_time_distribution(df)
        
        # 3. Problem Areas Highlight
        plt.subplot(2, 2, 3)
        self._plot_problem_areas(df)
        
        # 4. Statistics Summary
        plt.subplot(2, 2, 4)
        self._plot_stats_summary(df)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed mismatch report
        self._print_mismatch_report(df)
    
    def _plot_success_rate_heatmap(self, df: pd.DataFrame):
        """Plot heatmap of success rates by n and ratio"""
        success_rates = df.pivot_table(
            values='is_correct',
            index='ratio',
            columns='n',
            aggfunc='mean'
        )
        
        sns.heatmap(
            success_rates,
            annot=True,
            fmt='.2%',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Success Rate'}
        )
        plt.title('Success Rate by Problem Size')
        plt.xlabel('Number of Variables (n)')
        plt.ylabel('Clause/Variable Ratio')
    
    def _plot_time_distribution(self, df: pd.DataFrame):
        """Plot solving time distribution with error highlighting"""
        sns.boxplot(
            x='n',
            y='solving_time',
            hue='is_correct',
            data=df,
            palette=['red', 'green']
        )
        plt.title('Solving Time Distribution')
        plt.xlabel('Number of Variables (n)')
        plt.ylabel('Time (ms)')
        plt.legend(title='Correct', labels=['No', 'Yes'])
    
    def _plot_problem_areas(self, df: pd.DataFrame):
        """Scatter plot highlighting problem areas"""
        plt.scatter(
            df[df['is_correct']]['n'],
            df[df['is_correct']]['ratio'],
            c='green',
            alpha=0.5,
            label='Correct'
        )
        plt.scatter(
            df[~df['is_correct']]['n'],
            df[~df['is_correct']]['ratio'],
            c='red',
            alpha=0.5,
            label='Incorrect'
        )
        
        plt.title('Problem Areas')
        plt.xlabel('Number of Variables (n)')
        plt.ylabel('Clause/Variable Ratio')
        plt.legend()
        
        # Add annotations for incorrect results
        for _, row in df[~df['is_correct']].iterrows():
            plt.annotate(
                f"n={row['n']}\nr={row['ratio']:.1f}",
                (row['n'], row['ratio']),
                xytext=(5, 5),
                textcoords='offset points'
            )
    
    def _plot_stats_summary(self, df: pd.DataFrame):
        """Plot summary statistics"""
        stats = {
            'Total Tests': len(df),
            'Correct Results': df['is_correct'].sum(),
            'Incorrect Results': (~df['is_correct']).sum(),
            'Satisfiable Cases': df['is_satisfiable'].sum(),
            'Avg Time (ms)': df['solving_time'].mean()
        }
        
        y_pos = range(len(stats))
        plt.barh(y_pos, list(stats.values()))
        plt.yticks(y_pos, list(stats.keys()))
        plt.title('Summary Statistics')
        
        # Add value labels
        for i, v in enumerate(stats.values()):
            plt.text(v, i, f' {v:.1f}' if isinstance(v, float) else f' {v}')
    
    def _print_mismatch_report(self, df: pd.DataFrame):
        """Print detailed report of mismatches"""
        mismatches = df[~df['is_correct']]
        
        if len(mismatches) == 0:
            print("\n✅ No mismatches detected across all test cases!")
            return
        
        print("\n⚠️ Mismatch Report")
        print("=" * 60)
        print(f"Total mismatches: {len(mismatches)}")
        
        # Group mismatches by parameters
        param_groups = mismatches.groupby(['n', 'ratio'])
        
        print("\nMismatches by parameters:")
        for (n, ratio), group in param_groups:
            print(f"\nParameters: n={n}, ratio={ratio:.2f}")
            print(f"Number of failures: {len(group)}")
            print(f"Average solving time: {group['solving_time'].mean():.2f}ms")
            
            # Calculate failure rate for these parameters
            total_cases = len(df[(df['n'] == n) & (df['ratio'] == ratio)])
            failure_rate = len(group) / total_cases * 100
            print(f"Failure rate: {failure_rate:.1f}%")

def get_solver_class() -> Type:
    """Get the solver class based on user input."""
    print("\nAvailable solvers:")
    print("1. DPLL")
    print("2. Random Walk")
    print("3. Exhaustive Search")
    
    while True:
        choice = input("\nSelect solver to debug (1-3): ")
        if choice == "1":
            return DPLLSolver
        elif choice == "2":
            return RandomSATSolver
        elif choice == "3":
            return ExhaustiveSATSolver
        else:
            print("Invalid choice. Please select 1-3.")

def get_problem_parameters() -> tuple:
    """Get problem parameters from user."""
    # Get number of variables
    while True:
        n = input("\nEnter number of variables (3-20 recommended): ")
        try:
            n = int(n)
            if n >= 3:
                break
            print("Number must be at least 3.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get clause/variable ratio
    while True:
        ratio = input("Enter clause/variable ratio (3.0-5.0 recommended): ")
        try:
            ratio = float(ratio)
            if ratio > 0:
                break
            print("Ratio must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get number of test cases
    while True:
        num_tests = input("Enter number of test cases (1-10 recommended): ")
        try:
            num_tests = int(num_tests)
            if num_tests > 0:
                break
            print("Number must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    return n, ratio, num_tests

def debug_solver_with_viz(solver_class: Type, n: int, ratio: float, num_tests: int) -> None:
    """Enhanced debug_solver function that uses the visualizer"""
    visualizer = DebugVisualizer()
    
    # Create solvers
    test_solver = solver_class(debug=True)
    verification_solver = ExhaustiveSATSolver(debug=False)
    generator = RandomFormulaGenerator()
    
    for test in range(num_tests):
        # Generate and solve formula
        formula = generator.generate(n, int(n * ratio))
        
        # Test solver
        solver_start = time.time()
        test_result = test_solver.solve(formula)
        solver_time = (time.time() - solver_start) * 1000
        
        # Verify against exhaustive search
        verification_result = verification_solver.solve(formula)
        
        # Record result
        result = DebugResult(
            n=n,
            ratio=ratio,
            solver_name=solver_class.__name__,
            solving_time=solver_time,
            is_correct=(test_result is None) == (verification_result is None),
            is_satisfiable=verification_result is not None,
            solver_stats=test_solver.stats.__dict__
        )
        visualizer.add_result(result)
    
    # Generate visualizations
    visualizer.plot_results()

def main():
    print("SAT Solver Debugger with Visualization")
    print("=" * 60)
    
    # Get solver and parameters
    solver_class = get_solver_class()
    n, ratio, num_tests = get_problem_parameters()
    
    # Run debug session with visualization
    debug_solver_with_viz(solver_class, n, ratio, num_tests)

if __name__ == "__main__":
    main()