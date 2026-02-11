"""
Tournament Agent for Pipeline3 - Swiss-style Cross-Pair Tournament.

This agent implements a Swiss-style tournament system for ranking hypotheses
across multiple gene pairs based on scientific rigor and reasoning strength.
"""

import logging
import random
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from utils.elo_scoring import (
    DEFAULT_ELO_SCORE,
    update_elo_score_pairwise,
    calculate_k_factor
)
from external_tools.llm_client import llm_generate


class TournamentAgent:
    """
    Swiss-style tournament agent for cross-pair hypothesis ranking.

    The Swiss system pairs hypotheses with similar ELO scores each round,
    which efficiently identifies the strongest hypotheses without requiring
    a full round-robin tournament.
    """

    def __init__(self, judging_criteria: str = None):
        """
        Initialize the tournament agent.

        Args:
            judging_criteria: Custom judging criteria prompt. If None, uses default.
        """
        self.judging_criteria = judging_criteria or self._get_default_judging_criteria()
        self.config = {
            "max_rounds": 10,
            "elo_pairing_range": 50,      # Match within �50 ELO
            "stability_check_k": 20,       # Top-K to check for stability
            "stability_rounds": 2,         # Unchanged for N rounds = stable
        }
        self.match_history = []

    def _get_default_judging_criteria(self) -> str:
        """Return the default judging criteria prompt."""
        return """
You are judging two synthetic lethality hypotheses from a CRISPR screen.
Compare them based on scientific rigor and reasoning strength.

HYPOTHESIS A (Gene Pair: {gene_pair_a}):
{hypothesis_a}

HYPOTHESIS B (Gene Pair: {gene_pair_b}):
{hypothesis_b}

Judge which hypothesis is STRONGER using these criteria:

1. **Biological Relevance**: Is the hypothesis biologically grounded?

2. **Novelty**: Is the mechanistic interpretation novel, not already well-studied?
   (The pair may be previously reported but mechanism never discussed)

3. **Mechanistic Clarity**: Clear explanation of known vs gaps?
   Intermediate components mapped out? Clear pathway visualization?

4. **Follow-up Tractability**: Can the mechanism be tested with simple,
   effective experiments?

5. **Rival Quality**: Are alternative explanations mutually exclusive
   with trackable predictions?

6. **Clinical Relevance** (secondary): Cancer mutation frequency?
   Druggability of gene partners?

IMPORTANT: Judge based on reasoning quality, NOT biological similarity.
Different gene pairs can be fairly compared on scientific rigor.

Provide brief analysis for each criterion, then conclude:
WINNER: [A or B]
REASONING: [One sentence explaining why]
"""

    def run_swiss_tournament(self, hypotheses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a Swiss-style tournament to rank hypotheses.

        Args:
            hypotheses: List of hypothesis dictionaries with at minimum:
                - id: Unique identifier
                - description: Full hypothesis text
                - gene_a, gene_b: Gene pair for this hypothesis

        Returns:
            Dictionary containing:
                - rankings: List of hypotheses sorted by final ELO
                - match_history: All match results
                - rounds_completed: Number of rounds run
                - stability_achieved: Whether rankings stabilized
        """
        if len(hypotheses) < 2:
            logging.warning("Not enough hypotheses for tournament")
            return {
                "rankings": hypotheses,
                "match_history": [],
                "rounds_completed": 0,
                "stability_achieved": True
            }

        logging.info(f"Starting Swiss tournament with {len(hypotheses)} hypotheses")

        # Initialize tournament entries with ELO scores
        entries = self._initialize_entries(hypotheses)

        # Track rankings for stability check
        previous_rankings = []
        stable_rounds = 0
        rounds_completed = 0

        for round_num in range(1, self.config["max_rounds"] + 1):
            logging.info(f"=== Tournament Round {round_num} ===")

            # Create pairings for this round
            if round_num == 1:
                pairings = self._create_random_pairings(entries)
            else:
                pairings = self._create_elo_pairings(entries)

            if not pairings:
                logging.info("No valid pairings available, ending tournament")
                break

            # Run all matches in this round
            round_results = self._run_round(pairings, entries)
            self.match_history.extend(round_results)

            rounds_completed = round_num

            # Get current rankings
            current_rankings = self._get_top_k_ids(entries, self.config["stability_check_k"])

            # Check stability
            if current_rankings == previous_rankings:
                stable_rounds += 1
                logging.info(f"Rankings stable for {stable_rounds} rounds")

                if stable_rounds >= self.config["stability_rounds"]:
                    logging.info("Rankings stabilized, ending tournament early")
                    break
            else:
                stable_rounds = 0

            previous_rankings = current_rankings

        # Create final rankings
        final_rankings = self._create_final_rankings(entries)

        return {
            "rankings": final_rankings,
            "match_history": self.match_history,
            "rounds_completed": rounds_completed,
            "stability_achieved": stable_rounds >= self.config["stability_rounds"]
        }

    def _initialize_entries(self, hypotheses: List[Dict]) -> List[Dict]:
        """Initialize tournament entries with ELO scores and tracking data."""
        entries = []
        for hyp in hypotheses:
            entry = {
                "hypothesis": hyp,
                "id": hyp.get("id", f"hyp_{len(entries)}"),
                "elo_score": hyp.get("elo_score", DEFAULT_ELO_SCORE),
                "wins": 0,
                "losses": 0,
                "matches": 0,
                "opponents": set()  # Track who they've faced
            }
            entries.append(entry)
        return entries

    def _create_random_pairings(self, entries: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Create random pairings for round 1."""
        shuffled = entries.copy()
        random.shuffle(shuffled)

        pairings = []
        for i in range(0, len(shuffled) - 1, 2):
            pairings.append((shuffled[i], shuffled[i + 1]))

        return pairings

    def _create_elo_pairings(self, entries: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """
        Create pairings based on ELO scores (Swiss system).

        Pair hypotheses with similar ELO scores that haven't faced each other.
        """
        # Sort by ELO descending
        sorted_entries = sorted(entries, key=lambda e: e["elo_score"], reverse=True)

        paired = set()
        pairings = []

        for i, entry1 in enumerate(sorted_entries):
            if entry1["id"] in paired:
                continue

            # Find best available opponent
            best_opponent = None
            best_score = float('inf')

            for j, entry2 in enumerate(sorted_entries):
                if i == j or entry2["id"] in paired:
                    continue

                # Skip if they've already faced each other
                if entry2["id"] in entry1["opponents"]:
                    continue

                # Calculate pairing score (prefer similar ELO)
                elo_diff = abs(entry1["elo_score"] - entry2["elo_score"])

                if elo_diff < best_score:
                    best_score = elo_diff
                    best_opponent = entry2

            if best_opponent:
                pairings.append((entry1, best_opponent))
                paired.add(entry1["id"])
                paired.add(best_opponent["id"])

        logging.info(f"Created {len(pairings)} pairings for this round")
        return pairings

    def _run_round(self, pairings: List[Tuple[Dict, Dict]],
                   entries: List[Dict]) -> List[Dict]:
        """Run all matches in a round and update ELO scores."""
        round_results = []

        for entry1, entry2 in pairings:
            hyp1 = entry1["hypothesis"]
            hyp2 = entry2["hypothesis"]

            logging.info(f"Match: {entry1['id']} vs {entry2['id']}")

            # Run the comparison (now returns 3 values including validity flag)
            winner_id, transcript, is_valid = self.compare_hypotheses(hyp1, hyp2)

            # Track opponents regardless of outcome
            entry1["opponents"].add(entry2["id"])
            entry2["opponents"].add(entry1["id"])
            entry1["matches"] += 1
            entry2["matches"] += 1

            if not is_valid or winner_id is None:
                # Draw - no ELO changes, record as draw
                logging.warning(f"Match {entry1['id']} vs {entry2['id']} resulted in DRAW (comparison failed)")

                result = {
                    "hyp1_id": entry1["id"],
                    "hyp2_id": entry2["id"],
                    "winner_id": None,  # Draw
                    "is_draw": True,
                    "hyp1_elo_before": entry1["elo_score"],
                    "hyp2_elo_before": entry2["elo_score"],
                    "hyp1_elo_after": entry1["elo_score"],  # No change
                    "hyp2_elo_after": entry2["elo_score"],  # No change
                    "transcript": transcript,
                    "timestamp": datetime.now().isoformat()
                }
                round_results.append(result)
                continue

            # Valid match with winner - update ELO
            if winner_id == entry1["id"]:
                winner_entry, loser_entry = entry1, entry2
            else:
                winner_entry, loser_entry = entry2, entry1

            # Calculate K-factors
            k_winner = calculate_k_factor(winner_entry["elo_score"], winner_entry["matches"])
            k_loser = calculate_k_factor(loser_entry["elo_score"], loser_entry["matches"])
            k_factor = (k_winner + k_loser) // 2  # Use average

            # Update ELO scores
            old_winner_elo = winner_entry["elo_score"]
            old_loser_elo = loser_entry["elo_score"]

            new_winner_elo, new_loser_elo = update_elo_score_pairwise(
                old_winner_elo, old_loser_elo, k_factor
            )

            winner_entry["elo_score"] = new_winner_elo
            loser_entry["elo_score"] = new_loser_elo

            # Update win/loss stats
            winner_entry["wins"] += 1
            loser_entry["losses"] += 1

            # Record match result
            result = {
                "hyp1_id": entry1["id"],
                "hyp2_id": entry2["id"],
                "winner_id": winner_id,
                "is_draw": False,
                "hyp1_elo_before": old_winner_elo if winner_id == entry1["id"] else old_loser_elo,
                "hyp2_elo_before": old_loser_elo if winner_id == entry1["id"] else old_winner_elo,
                "hyp1_elo_after": new_winner_elo if winner_id == entry1["id"] else new_loser_elo,
                "hyp2_elo_after": new_loser_elo if winner_id == entry1["id"] else new_winner_elo,
                "transcript": transcript,
                "timestamp": datetime.now().isoformat()
            }
            round_results.append(result)

            logging.info(f"Winner: {winner_id} | ELO: {old_winner_elo} -> {new_winner_elo}")

        return round_results

    def compare_hypotheses(self, hyp1: Dict, hyp2: Dict, max_retries: int = 3) -> Tuple[str, str, bool]:
        """
        Compare two hypotheses using the judging criteria.

        Args:
            hyp1: First hypothesis dictionary
            hyp2: Second hypothesis dictionary
            max_retries: Maximum retry attempts for LLM calls

        Returns:
            Tuple of (winner_id, comparison_transcript, is_valid)
            is_valid is False if comparison failed and should be treated as draw
        """
        # Format the comparison prompt
        gene_pair_a = f"{hyp1.get('gene_a', 'Unknown')} - {hyp1.get('gene_b', 'Unknown')}"
        gene_pair_b = f"{hyp2.get('gene_a', 'Unknown')} - {hyp2.get('gene_b', 'Unknown')}"

        prompt = self.judging_criteria.format(
            gene_pair_a=gene_pair_a,
            hypothesis_a=hyp1.get("description", "No description available"),
            gene_pair_b=gene_pair_b,
            hypothesis_b=hyp2.get("description", "No description available")
        )

        # Get LLM comparison with retry logic
        response = None
        last_error = None

        for attempt in range(max_retries):
            try:
                logging.info(f"===TOURNAMENT_COMPARISON_START=== (attempt {attempt + 1}/{max_retries})")
                response = llm_generate(prompt)
                logging.info("===TOURNAMENT_COMPARISON_END===")
                break
            except Exception as e:
                last_error = e
                logging.warning(f"Tournament comparison attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff

        if response is None:
            logging.error(f"All {max_retries} tournament comparison attempts failed: {last_error}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            # Return as draw - no winner determined, match is invalid
            return None, f"Error after {max_retries} attempts: {last_error}", False

        # Extract winner from response, passing full context for potential clarification
        winner_id, parse_success = self._extract_winner_with_retry(
            response, hyp1, hyp2, gene_pair_a, gene_pair_b
        )

        if not parse_success:
            # Could not determine winner even after retry - treat as draw
            return None, response, False

        return winner_id, response, True

    def _extract_winner_with_retry(self, response: str, hyp1: Dict, hyp2: Dict,
                                    gene_pair_a: str, gene_pair_b: str) -> Tuple[Optional[str], bool]:
        """
        Extract winner from response, with retry if parsing fails.

        Args:
            response: Original LLM response
            hyp1, hyp2: Hypothesis dictionaries
            gene_pair_a, gene_pair_b: Gene pair names for context

        Returns:
            Tuple of (winner_id or None, success_bool)
        """
        # First attempt: try to parse the response
        winner_id = self._extract_winner(response, hyp1, hyp2)
        if winner_id is not None:
            return winner_id, True

        # Parsing failed - retry with explicit instruction AND full hypothesis context
        logging.warning("Could not parse winner from response, retrying with explicit format request")

        # Include full context so LLM can make informed decision
        clarification_prompt = f"""Your previous comparison did not clearly indicate a winner.
Please re-evaluate and respond with ONLY "WINNER: A" or "WINNER: B".

HYPOTHESIS A (Gene Pair: {gene_pair_a}):
{hyp1.get("description", "No description")[:1500]}

HYPOTHESIS B (Gene Pair: {gene_pair_b}):
{hyp2.get("description", "No description")[:1500]}

Your previous analysis concluded:
{response[:1000]}

Based on the hypotheses above and your analysis, which is stronger?
Respond with exactly: WINNER: A or WINNER: B
"""

        try:
            clarification_response = llm_generate(clarification_prompt)
            winner_id = self._extract_winner(clarification_response, hyp1, hyp2)
            if winner_id is not None:
                logging.info(f"Successfully extracted winner after clarification: {winner_id}")
                return winner_id, True
        except Exception as e:
            import traceback
            logging.error(f"Clarification request failed: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")

        # Still could not determine winner
        logging.error("Could not determine winner even after clarification - treating as draw")
        return None, False

    def _extract_winner(self, response: str, hyp1: Dict, hyp2: Dict) -> Optional[str]:
        """
        Extract the winner ID from the LLM response.

        Returns:
            Winner hypothesis ID, or None if winner could not be determined
        """
        response_lower = response.lower()

        # Try multiple patterns
        patterns = [
            r"winner:\s*\[?([ab])\]?",
            r"winner\s*(?:is)?:\s*(?:hypothesis\s*)?([ab])",
            r"(?:hypothesis\s*)?([ab])\s*(?:is\s*)?(?:the\s*)?winner",
            r"([ab])\s*is\s*(?:the\s*)?(?:stronger|better|superior)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                winner_letter = match.group(1).upper()
                if winner_letter == "A":
                    return hyp1.get("id", "hyp1")
                else:
                    return hyp2.get("id", "hyp2")

        # Fallback: look for "A" or "B" near "winner" or "better"
        if "winner" in response_lower or "better" in response_lower or "stronger" in response_lower:
            # Count mentions after the keyword
            a_mentions = response_lower.count("hypothesis a") + response_lower.count("winner: a")
            b_mentions = response_lower.count("hypothesis b") + response_lower.count("winner: b")

            if a_mentions > b_mentions:
                return hyp1.get("id", "hyp1")
            elif b_mentions > a_mentions:
                return hyp2.get("id", "hyp2")

        # Could not determine winner - return None (will be handled by caller)
        return None

    def _get_top_k_ids(self, entries: List[Dict], k: int) -> List[str]:
        """Get the IDs of the top K entries by ELO."""
        sorted_entries = sorted(entries, key=lambda e: e["elo_score"], reverse=True)
        return [e["id"] for e in sorted_entries[:k]]

    def _create_final_rankings(self, entries: List[Dict]) -> List[Dict]:
        """Create the final rankings with hypothesis data."""
        # Sort by ELO descending
        sorted_entries = sorted(entries, key=lambda e: e["elo_score"], reverse=True)

        rankings = []
        for rank, entry in enumerate(sorted_entries, 1):
            hyp = entry["hypothesis"].copy()
            hyp["tournament_rank"] = rank
            hyp["final_elo"] = entry["elo_score"]
            hyp["tournament_wins"] = entry["wins"]
            hyp["tournament_losses"] = entry["losses"]
            hyp["tournament_matches"] = entry["matches"]
            rankings.append(hyp)

        return rankings
