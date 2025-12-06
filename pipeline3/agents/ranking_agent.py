"""
Simplified Ranking Agent for Pipeline3 - Genetic Algorithm Implementation

This agent handles selection in the genetic algorithm by performing
Elo tournament ranking with pairwise comparisons.

Removed from Pipeline2:
- Scientific debates and argumentative reasoning
- Multi-round debate tournaments
- Complex debate strategies

Kept:
- Elo tournament system with pairwise comparisons
- PROMPT_PAIRWISE_COMPARISON for winner determination
- Population ranking and selection
- All Pipeline2 tournament infrastructure
"""

import logging
import random
import math
import re
import time
from typing import Dict, List, Any, Optional, Tuple, FrozenSet

from external_tools.gpt4o import gpt4o_generate
from utils.elo_scoring import update_elo_score_pairwise
from config import DEFAULT_ELO_SCORE
from prompts import PROMPT_PAIRWISE_COMPARISON


class RankingAgent:
    """
    Ranking Agent implements tournament approach to evaluate and rank
    scientific hypotheses using Elo-based pairwise comparisons.
    
    Simplified from Pipeline2 by removing scientific debates.
    """
    
    def __init__(self, config=None, mode="drug-repurposing"):
        """
        Initialize the Ranking Agent.

        Args:
            config: Optional configuration dictionary for customizing agent behavior
            mode: Pipeline mode ("drug-repurposing" or "general")
        """
        self.mode = mode
        # Set default configuration values
        self.config = {
            # Thresholds for determining top-ranked hypothesis status
            "top_elo_threshold": 1500,  # ELO score threshold for top-ranked hypotheses
            "close_match_threshold": 40,  # ELO difference threshold for close matches
            "min_matches_for_ranking": 5,  # Minimum matches needed for reliable ranking
            
            # Tournament parameters
            "max_recent_matches": 10,  # Number of recent matches to track to avoid rematches
            
            # Pair selection scoring weights
            "pair_score_recency_factor": 50,  # Weight for recency in pair selection
            "pair_score_rank_divisor": 100,  # Divisor for combined ELO scores in pair selection
        }
        
        # Update with any provided config values
        if config:
            self.config.update(config)
            
        logging.info("Ranking Agent initialized with Elo tournament system")
    
    def run_tournament(self, hypotheses: List[Dict[str, Any]], 
                      proximity_graph: Optional[Dict[str, Any]] = None,
                      research_goal: str = "",
                      preferences: str = "") -> Dict[str, Any]:
        """
        DEPRECATED: Tournament ranking no longer used in GA-aligned approach.
        Kept for reference only. Use fitness-based selection instead.
        
        Run a tournament to rank hypotheses using pairwise comparisons only.
        
        Args:
            hypotheses: List of hypothesis dictionaries to rank
            proximity_graph: Optional proximity graph (unused but kept for compatibility)
            research_goal: The scientific research goal
            preferences: Criteria for hypothesis evaluation
            
        Returns:
            Dictionary with rankings and tournament details
        """
        if not hypotheses:
            return {"rankings": []}
            
        n_hypotheses = len(hypotheses)
        logging.info(f"Running tournament with {n_hypotheses} hypotheses")
        
        # Create a copy of hypotheses to avoid modifying originals
        tournament_entries = []
        for hyp in hypotheses:
            # Create tournament entry with original hypothesis and tracking data
            entry = {
                "hypothesis": hyp,
                "id": hyp.get("id", "unknown"),
                "elo_score": hyp.get("elo_score", DEFAULT_ELO_SCORE),
                "wins": 0,
                "losses": 0,
                "matches": 0
            }
            tournament_entries.append(entry)
        
        # Run pairwise comparisons
        total_matches = min(10, n_hypotheses * (n_hypotheses - 1) // 2)  # Limit matches
        matches_run = 0
        
        for i in range(n_hypotheses):
            for j in range(i + 1, n_hypotheses):
                if matches_run >= total_matches:
                    break
                
                # Run pairwise comparison
                winner_idx = self._run_pairwise_comparison(
                    tournament_entries[i]["hypothesis"],
                    tournament_entries[j]["hypothesis"],
                    research_goal,
                    preferences
                )
                
                # Update ELO scores
                if winner_idx == 0:  # hypothesis i wins
                    tournament_entries[i]["wins"] += 1
                    tournament_entries[j]["losses"] += 1
                    tournament_entries[i]["elo_score"], tournament_entries[j]["elo_score"] = \
                        update_elo_score_pairwise(
                            tournament_entries[i]["elo_score"],
                            tournament_entries[j]["elo_score"],
                            1  # winner = 1
                        )
                else:  # hypothesis j wins
                    tournament_entries[j]["wins"] += 1
                    tournament_entries[i]["losses"] += 1
                    tournament_entries[j]["elo_score"], tournament_entries[i]["elo_score"] = \
                        update_elo_score_pairwise(
                            tournament_entries[j]["elo_score"],
                            tournament_entries[i]["elo_score"],
                            1  # winner = 1
                        )
                
                tournament_entries[i]["matches"] += 1
                tournament_entries[j]["matches"] += 1
                matches_run += 1
            
            if matches_run >= total_matches:
                break
        
        # Update original hypotheses with new ELO scores
        for i, entry in enumerate(tournament_entries):
            hypotheses[i]["elo_score"] = entry["elo_score"]
        
        # Sort by ELO score
        tournament_entries.sort(key=lambda x: x["elo_score"], reverse=True)
        
        # Create rankings
        rankings = []
        for rank, entry in enumerate(tournament_entries):
            rankings.append({
                "rank": rank + 1,
                "hypothesis": entry["hypothesis"],
                "elo_score": entry["elo_score"],
                "wins": entry["wins"],
                "losses": entry["losses"],
                "matches": entry["matches"]
            })
        
        logging.info(f"Tournament complete. Top ELO: {rankings[0]['elo_score']}")
        
        return {
            "rankings": rankings,
            "total_matches": matches_run,
            "tournament_stats": {
                "hypotheses_count": n_hypotheses,
                "matches_run": matches_run,
                "avg_elo": sum(r["elo_score"] for r in rankings) / len(rankings)
            }
        }
    
    def _run_pairwise_comparison(self, hyp1: Dict[str, Any], hyp2: Dict[str, Any], 
                               research_goal: str, preferences: str) -> int:
        """
        DEPRECATED: Pairwise comparisons no longer used in GA-aligned approach.
        Kept for reference only.
        
        Run a pairwise comparison between two hypotheses using Pipeline2's approach.
        
        Args:
            hyp1: First hypothesis
            hyp2: Second hypothesis  
            research_goal: The research goal
            preferences: Evaluation preferences
            
        Returns:
            0 if hyp1 wins, 1 if hyp2 wins
        """
        # Create comparison prompt
        prompt = PROMPT_PAIRWISE_COMPARISON.format(
            goal=research_goal,
            preferences=preferences,
            hypothesis_1_title=hyp1.get("title", "Untitled"),
            hypothesis_1_description=hyp1.get("description", "No description"),
            hypothesis_2_title=hyp2.get("title", "Untitled"),
            hypothesis_2_description=hyp2.get("description", "No description")
        )
        
        # Generate comparison result
        result = gpt4o_generate(prompt)
        
        # Parse result to determine winner
        if "WINNER: HYPOTHESIS 1" in result.upper():
            return 0
        elif "WINNER: HYPOTHESIS 2" in result.upper():
            return 1
        else:
            # Default to random if unclear
            logging.warning("Unclear winner from comparison, choosing randomly")
            return random.randint(0, 1)
    
    # Note: tournament_selection method removed - now using integrated tournament + fitness approach
    
    def _combine_population_with_scores(self, population: List[Dict[str, Any]], 
                                      fitness_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine population with their fitness scores.
        
        Args:
            population: List of hypotheses
            fitness_evaluations: List of fitness evaluations
            
        Returns:
            List of hypotheses with integrated fitness scores
        """
        # Create lookup for fitness scores
        fitness_lookup = {}
        for eval_result in fitness_evaluations:
            hyp_id = eval_result.get("hypothesis_id")
            if hyp_id:
                fitness_lookup[hyp_id] = eval_result
        
        # Combine with population
        scored_population = []
        for hypothesis in population:
            hyp_id = hypothesis.get("id")
            fitness_data = fitness_lookup.get(hyp_id, {})
            
            # Create combined entry
            combined = hypothesis.copy()
            combined.update({
                "fitness_score": fitness_data.get("fitness_score", 0),
                "correctness_score": fitness_data.get("correctness_score", 3),
                "novelty_score": fitness_data.get("novelty_score", 3),
                "quality_score": fitness_data.get("quality_score", 3),
                "review_summary": fitness_data.get("review_summary", "")
            })
            
            scored_population.append(combined)
        
        return scored_population
    
    def rank_population(self, population: List[Dict[str, Any]], 
                       fitness_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank population by fitness scores for final results.
        
        Args:
            population: List of hypotheses to rank
            fitness_evaluations: List of fitness evaluations (unused but kept for compatibility)
            
        Returns:
            List of hypotheses sorted by fitness score (descending)
        """
        if not population:
            return []
        
        # Sort by fitness score (highest first)
        ranked_population = sorted(population, 
                                 key=lambda x: x.get("fitness_score", 0), 
                                 reverse=True)
        
        logging.info(f"Ranked population of {len(ranked_population)} hypotheses")
        
        return ranked_population