import logging

def compute_analytics(hypotheses: list, rankings: list) -> dict:
    """Compute analytics on agent performance and hypothesis metrics.

    Returns a dictionary with metrics such as number of hypotheses, average ELO score, and clustering statistics.
    """
    num_hypotheses = len(hypotheses)
    num_rankings = len(rankings)
    total_elo = sum(hyp.get("elo_score", 1000) for hyp in hypotheses)
    avg_elo = total_elo / num_hypotheses if num_hypotheses > 0 else 0
    cluster_ids = [hyp.get("cluster_id") for hyp in hypotheses if hyp.get("cluster_id") is not None]
    distinct_clusters = len(set(cluster_ids)) if cluster_ids else 0
    analytics = {
        "num_hypotheses": num_hypotheses,
        "num_rankings": num_rankings,
        "average_elo": avg_elo,
        "distinct_clusters": distinct_clusters
    }
    logging.info("Analytics computed: %s", analytics)
    return analytics

class AnalyticsUtility:
    def __init__(self):
        pass

    def compute_scores(self, ranked_hypotheses: list) -> dict:
        """Compute analytics on the ranked hypotheses.

        This method computes metrics such as number of hypotheses, average ELO score,
        and distinct clusters directly from the list of ranked hypotheses.
        """
        num_hypotheses = len(ranked_hypotheses)
        total_elo = sum(hyp.get("elo_score", 1000) for hyp in ranked_hypotheses)
        avg_elo = total_elo / num_hypotheses if num_hypotheses > 0 else 0
        cluster_ids = [hyp.get("cluster_id") for hyp in ranked_hypotheses if hyp.get("cluster_id") is not None]
        distinct_clusters = len(set(cluster_ids)) if cluster_ids else 0
        analytics = {
            "num_hypotheses": num_hypotheses,
            "average_elo": avg_elo,
            "distinct_clusters": distinct_clusters
        }
        logging.info("Analytics computed from ranked hypotheses: %s", analytics)
        return analytics 