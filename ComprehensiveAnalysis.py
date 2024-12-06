import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from SAT import (
    DPLLSolver, RandomFormulaGenerator, ExhaustiveSATSolver,
    RandomSATSolver
)

class ComprehensiveSATAnalyzer:
    """Enhanced comprehensive analyzer for SAT solver analysis"""
    
    def __init__(self):
        self.results = defaultdict(list)
        self.solvers = {
            'dpll': DPLLSolver(),
            'exhaustive': ExhaustiveSATSolver(),
            'random': RandomSATSolver(max_flips=100, max_tries=10)  # Reduced for performance
        }
        
        # Configure plot style
        plt.style.use('seaborn-v0_8')
        self.colors = sns.color_palette("husl", 8)
    
    def compare_solvers(
        self, 
        n: int, 
        ratio: float, 
        results: dict,
        solvers_to_run: List[str]
    ) -> None:
        """Compare specified solvers on the same formula"""
        generator = RandomFormulaGenerator()
        formula = generator.generate(n, int(n * ratio))
        
        trial_results = {
            'n': n,
            'ratio': ratio
        }
        
        # Run specified solvers
        for solver_name in solvers_to_run:
            solver = self.solvers[solver_name]
            result = solver.solve(formula)
            
            # Add basic metrics
            trial_results[f'{solver_name}_result'] = result is not None
            trial_results[f'{solver_name}_time'] = solver.stats.solving_time_ms.value
            
            # Add solver-specific metrics
            if solver_name == 'dpll':
                trial_results.update({
                    'backtracks': solver.stats.stats['backtracks'].value,
                    'unit_propagations': solver.stats.stats['unit_propagations'].value,
                    'pure_literals': solver.stats.stats['pure_literals'].value,
                    'decision_depths_avg': np.mean(solver.stats.stats['decision_depths'].value) 
                        if solver.stats.stats['decision_depths'].value else 0
                })
            elif solver_name == 'random':
                trial_results.update({
                    'total_flips': solver.stats.stats['total_flips'].value,
                    'successful_flips': solver.stats.stats['successful_flips'].value,
                    'restart_count': solver.stats.stats['restart_count'].value
                })
            elif solver_name == 'exhaustive':
                trial_results.update({
                    'nodes_visited': solver.stats.stats['nodes_visited'].value,
                    'assignments_tested': solver.stats.stats['assignments_tested'].value,
                    'partial_validations': solver.stats.stats['partial_validations'].value
                })
        
        # Store results
        results[len(results)] = trial_results
    
    def plot_results(self, results: dict) -> None:
        """Create comprehensive visualizations of solver performance"""
        # Convert results to DataFrame
        df = pd.DataFrame.from_dict(results, orient='index')
        
        self._plot_phase_transition(df)
        self._plot_time_complexity(df)
        self._plot_solver_comparison(df)
        self._plot_heatmaps(df)
        self._print_statistical_analysis(df)
    
    def _plot_phase_transition(self, df: pd.DataFrame) -> None:
        """Plot phase transition analysis"""
        plt.figure(figsize=(12, 6))
        
        # Select distinct n values
        n_values = sorted(df['n'].unique())
        
        for n in n_values:
            subset = df[df['n'] == n]
            sat_prob = subset.groupby('ratio')['dpll_result'].mean() * 100
            plt.plot(sat_prob.index, sat_prob.values, 'o-', 
                    label=f'n={n}', linewidth=2)
        
        plt.axvline(x=4.26, color='r', linestyle='--', 
                   label='Theoretical Phase Transition')
        plt.xlabel('Clause-to-Variable Ratio (m/n)')
        plt.ylabel('Satisfiability Rate (%)')
        plt.title('Phase Transition Analysis')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _plot_time_complexity(self, df: pd.DataFrame) -> None:
        """Plot time complexity analysis"""
        plt.figure(figsize=(12, 6))
        
        # Debug print
        print("\nDebug: Available columns:", df.columns.tolist())
        print("Debug: Number of rows:", len(df))
        
        # Create separate dataframes for each solver
        solver_dfs = {}
        for solver in ['dpll', 'random', 'exhaustive']:
            time_col = f'{solver}_time'
            if time_col in df.columns:
                solver_df = df[df[time_col].notna()]
                if not solver_df.empty:
                    solver_dfs[solver] = solver_df
                    # Debug print
                    print(f"\nDebug: {solver} solver data:")
                    print(f"Number of rows: {len(solver_df)}")
                    print(f"Unique n values: {sorted(solver_df['n'].unique())}")
                    print(f"Unique ratio values: {sorted(solver_df['ratio'].unique())}")
        
        # Plot for each solver
        for solver, solver_df in solver_dfs.items():
            time_col = f'{solver}_time'
            ratios = sorted(solver_df['ratio'].unique())
            
            for ratio in ratios:
                ratio_df = solver_df[solver_df['ratio'] == ratio]
                # Debug print
                print(f"\nDebug: {solver} at ratio {ratio}:")
                print("Data points:", list(zip(ratio_df['n'], ratio_df[time_col])))
                
                avg_times = ratio_df.groupby('n')[time_col].mean()
                if not avg_times.empty:  # Only plot if we have data
                    plt.plot(avg_times.index, avg_times.values, 'o-',
                            label=f'{solver.upper()}, ratio={ratio}',
                            alpha=0.7)
        
        plt.xlabel('Number of Variables (n)')
        plt.ylabel('Average Solving Time (ms)')
        plt.title('Solver Time Complexity Analysis')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    def _plot_solver_comparison(self, df: pd.DataFrame) -> None:
        """Plot solver performance comparison"""
        plt.figure(figsize=(12, 6))
        
        # Create separate dataframes for each solver
        solver_dfs = {}
        for solver in ['dpll', 'random', 'exhaustive']:
            time_col = f'{solver}_time'
            if time_col in df.columns:
                solver_df = df[df[time_col].notna()]
                if not solver_df.empty:
                    solver_dfs[solver] = solver_df
        
        # Plot for each solver
        for solver, solver_df in solver_dfs.items():
            time_col = f'{solver}_time'
            median_times = solver_df.groupby('n')[time_col].median()
            plt.plot(median_times.index, median_times.values, 'o-',
                    label=f'{solver.upper()}',
                    alpha=0.7)
        
        plt.xlabel('Number of Variables (n)')
        plt.ylabel('Median Solving Time (ms)')
        plt.title('Solver Performance Comparison')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _plot_heatmaps(self, df: pd.DataFrame) -> None:
        """Plot comprehensive heatmap analysis"""
        metrics = {
            'dpll_result': 'Satisfiability Rate (%)',
            'backtracks': 'Average Backtracks',
            'unit_propagations': 'Average Unit Propagations',
            'decision_depths_avg': 'Average Decision Depth'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx, (metric, title) in enumerate(metrics.items()):
            if metric not in df.columns:
                continue
                
            pivot = df.pivot_table(
                values=metric,
                index='n',
                columns='ratio',
                aggfunc='mean'
            )
            
            if metric == 'dpll_result':
                pivot *= 100  # Convert to percentage
            
            sns.heatmap(pivot, ax=axes[idx], 
                    cmap='RdYlBu_r',
                    annot=True, fmt='.1f',
                    cbar_kws={'label': title})
            
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Clause-to-Variable Ratio (m/n)')
            axes[idx].set_ylabel('Number of Variables (n)')
        
        plt.tight_layout()
        plt.show()
    
    def _print_statistical_analysis(self, df: pd.DataFrame) -> None:
        """Print comprehensive statistical analysis"""
        print("\nDetailed Statistical Analysis")
        print("=" * 80)
        
        # Satisfiability analysis
        print("\nSatisfiability Analysis:")
        sat_pivot = pd.pivot_table(
            df,
            values='dpll_result',
            index='ratio',
            columns='n',
            aggfunc=lambda x: f"{np.mean(x)*100:.1f}%"
        )
        print(sat_pivot)
        
        # Solver performance analysis
        for solver in ['dpll', 'random', 'exhaustive']:
            time_col = f'{solver}_time'
            if time_col not in df.columns:
                continue
            
            solver_df = df[df[time_col].notna()]
            if solver_df.empty:
                continue
                
            print(f"\n{solver.upper()} Solver Statistics:")
            time_stats = pd.pivot_table(
                solver_df,
                values=time_col,
                index='ratio',
                columns='n',
                aggfunc=['mean', 'median', 'std']
            )
            print("\nMean solving times (ms):")
            print(time_stats['mean'].round(2))
            print("\nMedian solving times (ms):")
            print(time_stats['median'].round(2))
            print("\nStandard deviation of solving times (ms):")
            print(time_stats['std'].round(2))

def run_comprehensive_analysis():
    """Run comprehensive analysis of SAT solvers"""
    analyzer = ComprehensiveSATAnalyzer()
    
    print("Starting Comprehensive SAT Solver Analysis")
    print("=" * 80)
    
    # Configuration
    small_n = [5, 10, 15]  # For all solvers
    large_n = [20, 25, 30]  # For DPLL only
    ratios = [3.0, 3.5, 3.8, 4.0, 4.1, 4.2, 4.26, 4.3, 4.4, 4.6, 4.8, 5.0]
    trials_per_config = 10
    
    results = {}
    
    # Run small instances with all solvers
    print("\nAnalyzing small instances...")
    for n in small_n:
        print(f"\nTesting n = {n}")
        for ratio in ratios:
            print(f"  m/n = {ratio:.2f}", end='\r')
            for _ in range(trials_per_config):
                analyzer.compare_solvers(n, ratio, results, 
                                      solvers_to_run=['dpll', 'random', 'exhaustive'])
    
    # Run larger instances with DPLL only
    print("\nAnalyzing larger instances...")
    for n in large_n:
        print(f"\nTesting n = {n}")
        for ratio in ratios:
            print(f"  m/n = {ratio:.2f}", end='\r')
            for _ in range(trials_per_config):
                analyzer.compare_solvers(n, ratio, results, 
                                      solvers_to_run=['dpll'])
    
    print("\nGenerating analysis and visualizations...")
    analyzer.plot_results(results)

if __name__ == "__main__":
    run_comprehensive_analysis()