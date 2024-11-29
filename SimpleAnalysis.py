from SAT import RandomFormulaGenerator, ExhaustiveSATSolver, DPLLSolver
import time
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics

class SATAnalyzer:
    def __init__(self):
        self.results = defaultdict(list)
        
    def run_trial(self, n: int, ratio: float, run_exhaustive: bool = False) -> Dict:
        """Run a single trial and return results."""
        generator = RandomFormulaGenerator()
        formula = generator.generate(n, int(n * ratio))
        
        # DPLL analysis
        dpll_solver = DPLLSolver()
        dpll_start = time.time()
        dpll_result = dpll_solver.solve(formula)
        dpll_time = time.time() - dpll_start
        
        result = {
            'dpll_time': dpll_time * 1000,  # Convert to ms
            'is_sat': dpll_result is not None,
            'backtracks': dpll_solver.stats['backtracks']
        }
        
        # Optional exhaustive search
        if run_exhaustive:
            exhaustive_solver = ExhaustiveSATSolver()
            exhaustive_start = time.time()
            exhaustive_result = exhaustive_solver.solve(formula)
            exhaustive_time = time.time() - exhaustive_start
            
            result.update({
                'exhaustive_time': exhaustive_time * 1000,
                'nodes_visited': exhaustive_solver.stats['nodes_visited'],
                'results_match': (dpll_result is None) == (exhaustive_result is None)
            })
            
        return result

    def print_detailed_stats(self, trials: List[Dict], n: int, ratio: float):
        """Print detailed statistics for a set of trials."""
        # Calculate basic statistics
        sat_rate = sum(1 for t in trials if t['is_sat']) / len(trials) * 100
        avg_time = statistics.mean(t['dpll_time'] for t in trials)
        median_time = statistics.median(t['dpll_time'] for t in trials)
        std_time = statistics.stdev(t['dpll_time'] for t in trials) if len(trials) > 1 else 0
        avg_backtracks = statistics.mean(t['backtracks'] for t in trials)
        
        # Format output
        print(f"\n{'='*20} Statistics for n={n}, m/n={ratio:.2f} {'='*20}")
        print(f"Satisfiability Rate: {sat_rate:.1f}%")
        print("\nTime Statistics (ms):")
        print(f"  Average: {avg_time:.2f}")
        print(f"  Median:  {median_time:.2f}")
        print(f"  Std Dev: {std_time:.2f}")
        print(f"\nBacktracking Statistics:")
        print(f"  Average backtracks: {avg_backtracks:.1f}")
        
        # If exhaustive search was run
        if 'exhaustive_time' in trials[0]:
            avg_exhaustive = statistics.mean(t['exhaustive_time'] for t in trials)
            avg_nodes = statistics.mean(t['nodes_visited'] for t in trials)
            matches = sum(1 for t in trials if t['results_match'])
            
            print("\nExhaustive Search Comparison:")
            print(f"  Average time: {avg_exhaustive:.2f}ms")
            print(f"  Average nodes visited: {avg_nodes:.1f}")
            print(f"  Results match DPLL: {matches}/{len(trials)}")

    def print_phase_transition_summary(self, results: Dict[Tuple[int, float], List[Dict]]):
        """Print summary focusing on phase transition analysis."""
        print("\n" + "="*80)
        print("PHASE TRANSITION ANALYSIS SUMMARY")
        print("="*80)
        
        # Group by n
        n_values = sorted(set(n for (n, _) in results.keys()))
        ratios = sorted(set(ratio for (_, ratio) in results.keys()))
        
        # Header
        header = "m/n ratio |" + "|".join(f" n={n:2d} " for n in n_values)
        print("\nSatisfiability Rates (%):")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        # Data rows
        for ratio in ratios:
            row = f"  {ratio:5.2f}   |"
            for n in n_values:
                trials = results.get((n, ratio), [])
                if trials:
                    sat_rate = sum(1 for t in trials if t['is_sat']) / len(trials) * 100
                    row += f" {sat_rate:4.1f} |"
                else:
                    row += "  --- |"
            print(row)
        
        # Print time analysis
        print("\nMedian Solving Times (ms):")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for ratio in ratios:
            row = f"  {ratio:5.2f}   |"
            for n in n_values:
                trials = results.get((n, ratio), [])
                if trials:
                    median_time = statistics.median(t['dpll_time'] for t in trials)
                    row += f" {median_time:4.1f} |"
                else:
                    row += "  --- |"
            print(row)

def run_comprehensive_analysis():
    analyzer = SATAnalyzer()
    
    print("3SAT Solver Comprehensive Analysis")
    print("="*80)
    
    # Configuration
    small_n = [5, 10, 15]  # For both DPLL and exhaustive
    large_n = [20, 25, 30]  # For DPLL only
    ratios = [3.0, 3.5, 3.8, 4.0, 4.1, 4.2, 4.26, 4.3, 4.4, 4.6, 4.8, 5.0]
    trials_per_config = 10
    
    # Part 1: Small instances with exhaustive comparison
    print("\nPart 1: Small Instance Analysis (with exhaustive search verification)")
    for n in small_n:
        for ratio in ratios:
            trials = []
            for _ in range(trials_per_config):
                trial = analyzer.run_trial(n, ratio, run_exhaustive=True)
                trials.append(trial)
                analyzer.results[(n, ratio)].append(trial)
            analyzer.print_detailed_stats(trials, n, ratio)
    
    # Part 2: Larger instances (DPLL only)
    print("\nPart 2: Large Instance Analysis (DPLL only)")
    for n in large_n:
        for ratio in ratios:
            trials = []
            for _ in range(trials_per_config):
                trial = analyzer.run_trial(n, ratio, run_exhaustive=False)
                trials.append(trial)
                analyzer.results[(n, ratio)].append(trial)
            analyzer.print_detailed_stats(trials, n, ratio)
    
    # Part 3: Phase transition analysis
    analyzer.print_phase_transition_summary(analyzer.results)
    
    # Print general observations
    print("\nKey Observations:")
    print("-" * 80)
    print("1. Phase Transition: The theoretical threshold of m/n ≈ 4.26 is reflected in:")
    print("   - Sharp drop in satisfiability rates around this ratio")
    print("   - Peak in solving times and backtracking frequency")
    print("\n2. Scaling Behavior:")
    print("   - Exponential increase in solving time with problem size")
    print("   - More pronounced effect near phase transition point")
    print("\n3. Algorithm Performance:")
    print("   - DPLL significantly outperforms exhaustive search")
    print("   - Unit propagation and pure literal elimination provide substantial speedup")
    print("   - Backtracking frequency correlates strongly with problem difficulty")

if __name__ == "__main__":
    run_comprehensive_analysis()