import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from SAT import RandomFormulaGenerator, ExhaustiveSATSolver, DPLLSolver
from typing import Dict, Tuple, Any

def compare_solvers(n: int, ratio: float, results: Dict[Tuple[int, float], Dict[str, Any]], run_exhaustive=True):
    """Compare DPLL and optionally Exhaustive search on the same formula."""
    generator = RandomFormulaGenerator()
    formula = generator.generate(n, int(n * ratio))
    
    # Test DPLL
    dpll_solver = DPLLSolver()
    dpll_start = time.time()
    dpll_result = dpll_solver.solve(formula)
    dpll_time = time.time() - dpll_start
    
    # Only run exhaustive search if flag is True
    if run_exhaustive:
        exhaustive_solver = ExhaustiveSATSolver()
        exhaustive_start = time.time()
        exhaustive_result = exhaustive_solver.solve(formula)
        exhaustive_time = time.time() - exhaustive_start
    else:
        exhaustive_time = None
    
    # Store results
    key = (n, ratio)
    if key not in results:
        results[key] = []
    
    results[key].append({
        'dpll_time': dpll_time * 1000,  # Convert to ms
        'exhaustive_time': exhaustive_time * 1000 if exhaustive_time is not None else None,
        'is_sat': dpll_result is not None,
        'backtracks': dpll_solver.stats['backtracks']
    })

def plot_results(results: dict):
    """Create visualizations focusing on satisfiability and time complexity analysis."""
    # Create DataFrame from results
    data_list = []
    for (n, ratio), trials in results.items():
        for trial in trials:
            data_list.append({
                'n': n,
                'ratio': ratio,
                **trial
            })
    df = pd.DataFrame(data_list)
    
    # 1. Detailed Satisfiability Analysis
    plt.figure(figsize=(12, 6))
    
    # Select 5 evenly spaced values of n
    all_n = sorted(df['n'].unique())
    
    # Main satisfiability plot
    for n in all_n:
        subset = df[df['n'] == n]
        sat_prob = subset.groupby('ratio')['is_sat'].mean() * 100  # Convert to percentage
        plt.plot(sat_prob.index, sat_prob.values, 'o-', label=f'n={n}', linewidth=2)
    
    plt.axvline(x=4.26, color='r', linestyle='--', label='Theoretical Phase Transition')
    plt.xlabel('Clause-to-Variable Ratio (m/n)')
    plt.ylabel('Satisfiability Rate (%)')
    plt.title('Phase Transition Analysis in Random 3-SAT\n(5 Selected Values of n)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 2. Time Complexity Analysis (DPLL Only)
    plt.figure(figsize=(12, 6))
    
    # Plot time complexity for all ratios
    for ratio in sorted(df['ratio'].unique()):
        times = []
        sizes = []
        for n in sorted(df['n'].unique()):
            subset = df[(df['n'] == n) & (np.abs(df['ratio'] - ratio) < 0.01)]
            if not subset.empty:
                mean_dpll_time = subset['dpll_time'].mean()
                times.append(mean_dpll_time)
                sizes.append(n)
        
        plt.plot(sizes, times, 'o-', label=f'ratio={ratio}', linewidth=2)
    
    plt.xlabel('Number of Variables (n)')
    plt.ylabel('Average DPLL Solving Time (ms)')
    plt.title('DPLL Time Complexity Analysis')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 3. Heatmap of satisfiability rates
    plt.figure(figsize=(12, 6))
    sat_rates = df.groupby(['n', 'ratio'])['is_sat'].mean() * 100
    sat_matrix = sat_rates.unstack()
    
    sns.heatmap(sat_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r',
                vmin=0, vmax=100, center=50)
    plt.title('Satisfiability Rate (%) Heatmap')
    plt.xlabel('Clause-to-Variable Ratio (m/n)')
    plt.ylabel('Number of Variables (n)')
    plt.tight_layout()
    plt.show()

    # Statistical Analysis
    print("\nDetailed Satisfiability Analysis:")
    print("================================")
    
    # Print sat rates
    sat_pivot = sat_rates.unstack()
    print("\nSatisfiability Rates (%):")
    print(sat_pivot.round(1))
    
    # Print median solving times
    time_pivot = pd.pivot_table(
        df,
        values='dpll_time',
        index='ratio',
        columns='n',
        aggfunc=['median', 'mean', 'std']
    )
    
    print("\nDPLL Solving Times (ms):")
    print("\nMedian times:")
    print(time_pivot['median'].round(2))
    print("\nMean times:")
    print(time_pivot['mean'].round(2))
    print("\nStandard deviation:")
    print(time_pivot['std'].round(2))

def main():
    print("3SAT Solver Analysis")
    print("=" * 50)

    # Define two sets of parameters: one for exhaustive search and one for DPLL only
    small_n = [5, 10, 15]  # For both DPLL and exhaustive
    large_n = [20, 25, 30, 35, 40]  # For DPLL only
    
    # More granular ratios, especially around phase transition
    ratios = [3.0, 3.5, 3.8, 4.0, 4.1, 4.2, 4.26, 4.3, 4.4, 4.6, 4.8, 5.0, 5.5, 7.0]
    trials = 10
    
    results = {}
    
    # Run both solvers for small n
    print("\nRunning both solvers for small n...")
    for n in small_n:
        print(f"\nTesting with n = {n}")
        for ratio in ratios:
            print(f"m/n = {ratio:.2f}", end='\r')
            key = (n, ratio)
            results[key] = []
            for _ in range(trials):
                compare_solvers(n, ratio, results, run_exhaustive=True)
    
    # Run only DPLL for large n
    print("\nRunning DPLL only for large n...")
    for n in large_n:
        print(f"\nTesting with n = {n}")
        for ratio in ratios:
            print(f"m/n = {ratio:.2f}", end='\r')
            key = (n, ratio)
            results[key] = []
            for _ in range(trials):
                compare_solvers(n, ratio, results, run_exhaustive=False)
    
    print("\nGenerating analysis and visualizations...")
    plot_results(results)

if __name__ == "__main__":
    main()
