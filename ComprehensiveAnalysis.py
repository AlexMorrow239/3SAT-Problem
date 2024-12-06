import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from SAT import DPLLSolver, ExhaustiveSATSolver, RandomSATSolver
from Utilities import RandomFormulaGenerator

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
        self._plot_heatmaps(df)
        self._print_statistical_analysis(df)
    
    def _plot_phase_transition(self, df: pd.DataFrame) -> None:
        """Plot clean and interpretable phase transition analysis"""
        plt.figure(figsize=(12, 8))
        
        # Select fewer, well-spaced n values
        n_values = sorted(df['n'].unique())
        selected_n = []
        
        if len(n_values) > 3:
            # Take only 3 values: smallest, middle, and largest
            selected_n = [
                n_values[0],          # smallest
                n_values[len(n_values)//2],  # middle
                n_values[-1]          # largest
            ]
        else:
            selected_n = n_values
        
        # Use distinct colors that are easy to distinguish
        colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(selected_n)]
        
        # Plot data with smoothing
        for n, color in zip(selected_n, colors):
            subset = df[df['n'] == n]
            # Calculate mean for each ratio with smoothing
            grouped = subset.groupby('ratio')
            sat_prob = grouped['dpll_result'].mean() * 100
            
            # Sort by ratio for proper line connection
            sat_prob = sat_prob.sort_index()
            
            # Create smooth line
            plt.plot(sat_prob.index, sat_prob.values,
                    'o-', label=f'n = {n}',
                    color=color,
                    linewidth=3,
                    markersize=8,
                    markeredgewidth=2,
                    markeredgecolor='white')
        
        # Add phase transition line with clear styling
        plt.axvline(x=4.26, color='#D64933', linestyle='--', 
                    linewidth=2.5, label='Phase Transition',
                    alpha=0.7)
        
        # Create distinct regions
        plt.axvspan(min(df['ratio']), 4.26, alpha=0.1, color='green', label='SAT Region')
        plt.axvspan(4.26, max(df['ratio']), alpha=0.1, color='red', label='UNSAT Region')
        
        # Improve grid and styling
        plt.grid(True, linestyle='--', alpha=0.2)
        plt.xlabel('Clause-to-Variable Ratio (α = m/n)', fontsize=12, labelpad=10)
        plt.ylabel('Probability of Satisfiability (%)', fontsize=12, labelpad=10)
        plt.title('Phase Transition in Random 3-SAT', fontsize=14, pad=20)
        
        # Customize legend
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                borderaxespad=0, frameon=True, fontsize=11)
        
        # Set axis limits
        plt.xlim(min(df['ratio']) * 0.98, max(df['ratio']) * 1.02)
        plt.ylim(-2, 102)
        
        # Set background color
        plt.gca().set_facecolor('white')
        
        # Add subtle tick lines
        plt.gca().tick_params(axis='both', which='major', labelsize=10)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
        
        plt.tight_layout()
        plt.show()

    def _plot_time_complexity(self, df: pd.DataFrame) -> None:
        """Plot time complexity analysis with seamless extrapolation for Exhaustive and Random solvers"""
        plt.figure(figsize=(12, 8))
        
        # Get unique n values and ratios
        all_n = sorted(df['n'].unique())
        small_n = [n for n in all_n if n <= 15]  # n values where all solvers were tested
        ratios = sorted(df['ratio'].unique())
        mid_ratio = ratios[len(ratios)//2]
        
        colors = {
            'dpll': '#2E86AB',
            'random': '#A23B72',
            'exhaustive': '#F18F01'
        }
        
        # Plot DPLL actual data
        dpll_data = df[df['dpll_time'].notna()]
        dpll_times = dpll_data.groupby('n')['dpll_time'].mean()
        plt.plot(dpll_times.index, dpll_times.values, 
                'o-', label='DPLL (measured)', 
                color=colors['dpll'], 
                linewidth=2, 
                markersize=8)

        # Calculate extrapolation for each solver
        for solver in ['exhaustive', 'random']:
            time_col = f'{solver}_time'
            solver_data = df[df[time_col].notna()]
            if not solver_data.empty:
                # Get actual data for small n
                actual_times = solver_data.groupby('n')[time_col].mean()
                
                # Plot actual data points
                plt.plot(actual_times.index, actual_times.values,
                        'o-', label=f'{solver.title()} (measured)',
                        color=colors[solver], linewidth=2, markersize=8)
                
                # Use last actual data point for scaling
                last_n = actual_times.index[-1]
                last_time = actual_times.values[-1]
                
                # Calculate scaling factor based on last actual point
                if solver == 'exhaustive':
                    scaling_factor = last_time / (2 ** last_n)
                    extrapolated_times = scaling_factor * (2 ** np.array(all_n))
                else:  # random
                    scaling_factor = last_time / (2 ** (last_n/2))
                    extrapolated_times = scaling_factor * (2 ** (np.array(all_n)/2))
                
                # Get the index where extrapolation should start
                start_idx = len(small_n) - 1
                
                # Plot extrapolated line starting from last actual point
                plt.plot(all_n[start_idx:], extrapolated_times[start_idx:],
                        '--', label=f'{solver.title()} (extrapolated)',
                        color=colors[solver], linewidth=2, alpha=0.7)
        
        # Add complexity reference lines
        n_ref = np.array(all_n)
        ref_point = dpll_times.values[-1] / (n_ref[-1] * np.log(n_ref[-1]))
        plt.plot(n_ref, ref_point * n_ref * np.log(n_ref),
                ':', color='gray', alpha=0.5, label='O(n log n) reference')
        
        # Customization
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Number of Variables (n)', fontsize=12)
        plt.ylabel('Average Solving Time (ms)', fontsize=12)
        plt.title(f'Solver Time Complexity Analysis\n(at α ≈ {mid_ratio:.2f})', 
                fontsize=14, pad=20)
        
        # Add shaded region between actual and extrapolated data
        plt.axvspan(small_n[-1], all_n[-1], color='gray', alpha=0.1, 
                    label='Extrapolation Region')
        
        # Customize legend
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                borderaxespad=0, frameon=True, fontsize=10)
        
        plt.tight_layout()
        plt.show()     
    
    def _plot_heatmaps(self, df: pd.DataFrame) -> None:
        """Plot heatmap analysis with wider depth heatmap and robust statistics"""
        metrics = {
            'dpll_result': 'SAT Probability (%)',
            'decision_depths_avg': 'Search Tree Depth'
        }
        
        # Create figure with extra space for annotations
        fig = plt.figure(figsize=(16, 8))
        
        # Create a gridspec with different column widths (40% for first, 60% for second)
        gs = plt.GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[5, 5], hspace=0.3, wspace=0.3)
        axes = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])]
        
        for idx, (metric, title) in enumerate(metrics.items()):
            if metric not in df.columns:
                continue
                
            # Create pivot table
            pivot = df.pivot_table(
                values=metric,
                index='n',
                columns='ratio',
                aggfunc='mean'
            )
            
            # Convert satisfiability to percentage
            if metric == 'dpll_result':
                pivot *= 100
                fmt = '.0f'
                cmap = 'RdYlBu_r'
                center = 50
                cbar_label = 'Percentage'
                annot_kws = {'size': 10}
            else:
                # Round depth values to 2 decimal places
                pivot = pivot.round(2)
                fmt = '.1f'
                cmap = 'viridis'
                center = None
                cbar_label = 'Average Depth'
                annot_kws = {'size': 8, "rotation": 45}
            
            # Create annotation array
            annot = pivot.copy()
            
            # Create heatmap
            im = sns.heatmap(
                pivot, 
                ax=axes[idx],
                cmap=cmap,
                annot=annot,
                fmt=fmt,
                cbar_kws={'label': cbar_label},
                center=center,
                annot_kws=annot_kws
            )
            
            # Customize axes
            axes[idx].set_title(title, pad=10)
            axes[idx].set_xlabel('Clause-to-Variable Ratio (α)')
            axes[idx].set_ylabel('Number of Variables (n)')
            
            # Add phase transition line and region labels for SAT probability
            if metric == 'dpll_result':
                transition_idx = np.abs(pivot.columns - 4.26).argmin()
                axes[idx].axvline(x=transition_idx, color='black', 
                                linestyle='--', alpha=0.5, linewidth=1)
                
                # Add region labels
                axes[idx].text(-0.2, pivot.shape[0]/2, 'SAT', 
                            rotation=90, verticalalignment='center')
                axes[idx].text(pivot.shape[1], pivot.shape[0]/2, 'UNSAT', 
                            rotation=90, verticalalignment='center')
            
            # Rotate x-axis labels
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45)
        
        # Add insights boxes below each heatmap
        insight_boxes = [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])]
        
        # Calculate key statistics more robustly
        sat_avg = df.groupby('ratio')['dpll_result'].mean() * 100
        
        # Find the ratio closest to 50% satisfiability
        hardest_ratio = sat_avg.index[np.abs(sat_avg - 50).argmin()]
        
        max_depth_ratio = df.groupby('ratio')['decision_depths_avg'].mean().idxmax()
        max_depth = df.groupby('ratio')['decision_depths_avg'].mean().max()
        
        # Create insight text
        sat_insights = (
            "Key Insights:\n"
            "• Clear phase transition at α ≈ 4.26\n"
            f"• Critical ratio observed at α ≈ {hardest_ratio:.2f}\n"
            "• Higher n values show sharper transition"
        )
        
        depth_insights = (
            "Key Insights:\n"
            "• Deeper searches needed near phase transition\n"
            f"• Peak computational effort at α ≈ {max_depth_ratio:.2f}\n"
            f"• Maximum average depth: {max_depth:.2f}"
        )
        
        # Add insight boxes with different styling
        for idx, (box, text) in enumerate(zip(insight_boxes, [sat_insights, depth_insights])):
            box.text(0.05, 0.5, text,
                    verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=1',
                            facecolor='white',
                            alpha=0.8,
                            edgecolor='gray'))
            box.axis('off')
        
        plt.suptitle('SAT Solver Performance Analysis', fontsize=14, y=0.95)
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
    large_n = [20, 25, 30, 40]  # For DPLL only
    ratios = [2.0, 3.0, 3.5, 4.26, 5.0, 7.0]
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