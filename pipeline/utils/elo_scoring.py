import logging
import math
from typing import Dict, List, Tuple

# Default starting ELO score for all hypotheses
DEFAULT_ELO_SCORE = 1200


def update_elo_score(current_elo: int, outcome: str, k_factor: int = 20, opponent_elo: int = None) -> int:
    """Update the ELO score based on the outcome of a hypothesis comparison.

    Args:
        current_elo (int): The current ELO score.
        outcome (str): 'win', 'loss', or 'draw' indicating the result of a tournament.
        k_factor (int, optional): The K-factor determines how much the ELO score changes. Defaults to 20.
        opponent_elo (int, optional): The opponent's ELO score for more accurate calculations.

    Returns:
        int: The updated ELO score.
    """
    # Simple mode - fixed ELO adjustments
    if opponent_elo is None:
        if outcome == 'win':
            return current_elo + k_factor
        elif outcome == 'loss':
            return current_elo - k_factor
        elif outcome == 'draw':
            return current_elo
        else:
            logging.warning("Unknown outcome provided: %s. ELO score remains unchanged.", outcome)
            return current_elo
    
    # Advanced mode - calculate ELO based on opponent's score
    expected_score = calculate_expected_score(current_elo, opponent_elo)
    actual_score = 1.0 if outcome == 'win' else 0.0 if outcome == 'loss' else 0.5
    
    # Calculate new ELO
    new_elo = current_elo + k_factor * (actual_score - expected_score)
    return round(new_elo)


def calculate_expected_score(player_elo: int, opponent_elo: int) -> float:
    """
    Calculate the expected score for a player in a match.
    
    Args:
        player_elo (int): The player's current ELO rating
        opponent_elo (int): The opponent's current ELO rating
        
    Returns:
        float: The expected score (between 0 and 1)
    """
    return 1.0 / (1.0 + math.pow(10, (opponent_elo - player_elo) / 400.0))


def calculate_k_factor(current_elo: int, num_matches: int = 0) -> int:
    """
    Calculate an appropriate K-factor based on the player's rating and experience.
    
    K-factor determines how much a rating changes after each match.
    - Higher K: Ratings change more quickly (good for new players)
    - Lower K: Ratings change more slowly (good for established players)
    
    Args:
        current_elo (int): The player's current ELO rating
        num_matches (int): Number of matches played
        
    Returns:
        int: The K-factor to use
    """
    # New players have higher K-factor
    if num_matches < 10:
        return 40
    
    # Very strong hypotheses have lower K-factor
    if current_elo > 1500:
        return 16
    
    # Default K-factor
    return 24


def update_elo_score_pairwise(winner_elo: int, loser_elo: int, k_factor: int = 20) -> Tuple[int, int]:
    """Update ELO scores for a winner and loser in a pairwise match.
    
    This function matches the signature expected by the ranking agent.
    
    Args:
        winner_elo (int): Current ELO score of the winner
        loser_elo (int): Current ELO score of the loser
        k_factor (int): K-factor for ELO calculation
        
    Returns:
        Tuple[int, int]: (new_winner_elo, new_loser_elo)
    """
    # Calculate expected scores
    winner_expected = calculate_expected_score(winner_elo, loser_elo)
    loser_expected = calculate_expected_score(loser_elo, winner_elo)
    
    # Winner gets score of 1, loser gets score of 0
    winner_actual = 1.0
    loser_actual = 0.0
    
    # Calculate new ELO scores
    new_winner_elo = winner_elo + k_factor * (winner_actual - winner_expected)
    new_loser_elo = loser_elo + k_factor * (loser_actual - loser_expected)
    
    return round(new_winner_elo), round(new_loser_elo)


def run_elo_tournament(hypotheses: List[Dict], 
                      matchups: List[Tuple[int, int]], 
                      results: List[str]) -> List[Dict]:
    """
    Run a tournament and update ELO scores for all participants.
    
    Args:
        hypotheses (List[Dict]): List of hypotheses with 'id' and 'elo_score' keys
        matchups (List[Tuple[int, int]]): List of (hyp1_idx, hyp2_idx) matchups
        results (List[str]): List of 'win1', 'win2', or 'draw' results
        
    Returns:
        List[Dict]: Updated list of hypotheses with new ELO scores
    """
    if len(matchups) != len(results):
        logging.error("Mismatch between matchups and results counts")
        return hypotheses
    
    # Create a copy to avoid modifying the original
    updated_hypotheses = [h.copy() for h in hypotheses]
    
    # Track match count for each hypothesis
    match_counts = {h.get('id', f'hyp_{i}'): 0 for i, h in enumerate(updated_hypotheses)}
    
    # Process each match and update scores
    for (idx1, idx2), result in zip(matchups, results):
        if idx1 >= len(updated_hypotheses) or idx2 >= len(updated_hypotheses):
            logging.warning(f"Invalid matchup indices: {idx1}, {idx2}")
            continue
            
        hyp1 = updated_hypotheses[idx1]
        hyp2 = updated_hypotheses[idx2]
        
        # Update match counts
        hyp1_id = hyp1.get('id', f'hyp_{idx1}')
        hyp2_id = hyp2.get('id', f'hyp_{idx2}')
        match_counts[hyp1_id] = match_counts.get(hyp1_id, 0) + 1
        match_counts[hyp2_id] = match_counts.get(hyp2_id, 0) + 1
        
        # Calculate appropriate K-factors
        k1 = calculate_k_factor(hyp1.get('elo_score', 1000), match_counts[hyp1_id])
        k2 = calculate_k_factor(hyp2.get('elo_score', 1000), match_counts[hyp2_id])
        
        # Determine outcomes for each hypothesis
        if result == 'win1':
            hyp1_outcome = 'win'
            hyp2_outcome = 'loss'
        elif result == 'win2':
            hyp1_outcome = 'loss'
            hyp2_outcome = 'win'
        else:  # Draw
            hyp1_outcome = 'draw'
            hyp2_outcome = 'draw'
            
        # Update ELO scores
        hyp1['elo_score'] = update_elo_score(
            hyp1.get('elo_score', 1000), 
            hyp1_outcome, 
            k1, 
            hyp2.get('elo_score', 1000)
        )
        
        hyp2['elo_score'] = update_elo_score(
            hyp2.get('elo_score', 1000), 
            hyp2_outcome, 
            k2, 
            hyp1.get('elo_score', 1000)
        )
    
    return updated_hypotheses