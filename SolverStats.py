from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum
import time
import statistics
from collections import defaultdict


class StatisticType(Enum):
    """Types of statistics that can be tracked"""
    COUNTER = "counter"
    TIMER = "timer"
    RATIO = "ratio"
    LIST = "list"


@dataclass
class StatisticValue:
    """Container for a statistic with its type and value"""
    type: StatisticType
    value: any
    description: str = ""
    
    def __post_init__(self):
        if self.type == StatisticType.LIST and not isinstance(self.value, list):
            self.value = []
        elif self.type == StatisticType.COUNTER and not isinstance(self.value, int):
            self.value = 0


@dataclass
class SolverStatistics:
    """Enhanced statistics tracking for SAT solvers"""
    # Common statistics for all solvers
    solving_time_ms: StatisticValue = field(default_factory=lambda: StatisticValue(
        StatisticType.TIMER, 0, "Total solving time in milliseconds"
    ))
    successful_solves: StatisticValue = field(default_factory=lambda: StatisticValue(
        StatisticType.COUNTER, 0, "Number of successful solutions found"
    ))
    failed_solves: StatisticValue = field(default_factory=lambda: StatisticValue(
        StatisticType.COUNTER, 0, "Number of unsatisfiable or timeout cases"
    ))
    variable_assignments: StatisticValue = field(default_factory=lambda: StatisticValue(
        StatisticType.LIST, [], "History of variable assignments made"
    ))
    
    # Solver-specific statistics
    stats: Dict[str, StatisticValue] = field(default_factory=dict)
    
    def __post_init__(self):
        self._start_time: Optional[float] = None
    
    def start_timer(self):
        """Start the solving timer"""
        self._start_time = time.perf_counter()
    
    def stop_timer(self):
        """Stop the solving timer and update total time"""
        if self._start_time is not None:
            self.solving_time_ms.value = (time.perf_counter() - self._start_time) * 1000
            self._start_time = None
    
    def increment(self, stat_name: str):
        """Increment a counter statistic"""
        if stat_name in self.stats and self.stats[stat_name].type == StatisticType.COUNTER:
            self.stats[stat_name].value += 1
    
    def append(self, stat_name: str, value: any):
        """Append a value to a list statistic"""
        if stat_name in self.stats and self.stats[stat_name].type == StatisticType.LIST:
            self.stats[stat_name].value.append(value)
    
    def set_value(self, stat_name: str, value: any):
        """Set the value of any statistic"""
        if stat_name in self.stats:
            self.stats[stat_name].value = value
    
    def reset(self):
        """Reset all statistics to their initial values"""
        self.solving_time_ms.value = 0
        self.successful_solves.value = 0
        self.failed_solves.value = 0
        self.variable_assignments.value = []
        for stat in self.stats.values():
            if stat.type == StatisticType.COUNTER:
                stat.value = 0
            elif stat.type == StatisticType.LIST:
                stat.value = []
            elif stat.type == StatisticType.TIMER:
                stat.value = 0


class DPLLStatistics(SolverStatistics):
    """Statistics specific to DPLL solver"""
    def __init__(self):
        super().__init__()
        self.stats.update({
            "backtracks": StatisticValue(
                StatisticType.COUNTER, 0, "Number of backtracking operations"
            ),
            "unit_propagations": StatisticValue(
                StatisticType.COUNTER, 0, "Number of unit propagations performed"
            ),
            "pure_literals": StatisticValue(
                StatisticType.COUNTER, 0, "Number of pure literals eliminated"
            ),
            "two_clause_rules": StatisticValue(
                StatisticType.COUNTER, 0, "Number of two-clause rules applied"
            ),
            "decision_depths": StatisticValue(
                StatisticType.LIST, [], "Depths of decision tree branches"
            ),
            "clause_sizes": StatisticValue(
                StatisticType.LIST, [], "Sizes of clauses after simplification"
            ),
            "variable_frequencies": StatisticValue(
                StatisticType.LIST, [], "Frequency of variable selections"
            )
        })


class RandomWalkStatistics(SolverStatistics):
    """Statistics specific to Random Walk solver"""
    def __init__(self):
        super().__init__()
        self.stats.update({
            "total_flips": StatisticValue(
                StatisticType.COUNTER, 0, "Total number of variable flips"
            ),
            "successful_flips": StatisticValue(
                StatisticType.COUNTER, 0, "Number of flips that improved solution"
            ),
            "restart_count": StatisticValue(
                StatisticType.COUNTER, 0, "Number of random restarts"
            ),
            "local_minima": StatisticValue(
                StatisticType.COUNTER, 0, "Number of local minima encountered"
            ),
            "unsatisfied_clauses": StatisticValue(
                StatisticType.LIST, [], "History of unsatisfied clause counts"
            ),
            "flip_improvements": StatisticValue(
                StatisticType.LIST, [], "Improvement in satisfied clauses per flip"
            )
        })


class ExhaustiveStatistics(SolverStatistics):
    """Statistics specific to Exhaustive solver"""
    def __init__(self):
        super().__init__()
        self.stats.update({
            "nodes_visited": StatisticValue(
                StatisticType.COUNTER, 0, "Total number of nodes visited"
            ),
            "assignments_tested": StatisticValue(
                StatisticType.COUNTER, 0, "Number of complete assignments tested"
            ),
            "partial_validations": StatisticValue(
                StatisticType.COUNTER, 0, "Number of partial assignment validations"
            ),
            "branch_depths": StatisticValue(
                StatisticType.LIST, [], "Depths of branches explored"
            ),
            "satisfying_depths": StatisticValue(
                StatisticType.LIST, [], "Depths where solutions were found"
            )
        })


class StatisticsAnalyzer:
    """Analyzer for solver statistics"""
    
    def __init__(self):
        self.results: Dict[str, List[SolverStatistics]] = defaultdict(list)
    
    def add_result(self, solver_name: str, stats: SolverStatistics):
        """Add a solver's statistics to the results"""
        self.results[solver_name].append(stats)
    
    def get_summary(self, solver_name: str) -> Dict[str, any]:
        """Generate summary statistics for a solver"""
        stats_list = self.results[solver_name]
        if not stats_list:
            return {}
            
        summary = {
            "total_runs": len(stats_list),
            "success_rate": self._calculate_success_rate(stats_list),
            "avg_solving_time": statistics.mean(s.solving_time_ms.value for s in stats_list),
            "median_solving_time": statistics.median(s.solving_time_ms.value for s in stats_list)
        }
        
        # Add solver-specific statistics
        first_stats = stats_list[0]
        for stat_name, stat_value in first_stats.stats.items():
            if stat_value.type == StatisticType.COUNTER:
                summary[f"avg_{stat_name}"] = statistics.mean(
                    s.stats[stat_name].value for s in stats_list
                )
            elif stat_value.type == StatisticType.LIST:
                # For list statistics, we'll calculate some basic statistics
                all_values = [v for s in stats_list for v in s.stats[stat_name].value]
                if all_values:
                    summary[f"{stat_name}_avg"] = statistics.mean(all_values)
                    summary[f"{stat_name}_max"] = max(all_values)
                    summary[f"{stat_name}_min"] = min(all_values)
        
        return summary
    
    def _calculate_success_rate(self, stats_list: List[SolverStatistics]) -> float:
        """Calculate the success rate from a list of statistics"""
        total_successes = sum(s.successful_solves.value for s in stats_list)
        total_attempts = sum(s.successful_solves.value + s.failed_solves.value for s in stats_list)
        return total_successes / total_attempts if total_attempts > 0 else 0


def create_solver_statistics(solver_type: str) -> SolverStatistics:
    """Factory function to create appropriate statistics object"""
    if solver_type.lower() == "dpll":
        return DPLLStatistics()
    elif solver_type.lower() == "random":
        return RandomWalkStatistics()
    elif solver_type.lower() == "exhaustive":
        return ExhaustiveStatistics()
    else:
        return SolverStatistics()