import logging
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.cluster import AgglomerativeClustering


def cluster_hypotheses(hypotheses: list) -> list:
    """Assigns cluster IDs to hypotheses in a simple, deterministic way.

    This is a dummy clustering that assigns a cluster id based on the hypothesis index modulus 2.
    """
    logging.info("Clustering %d hypotheses", len(hypotheses))
    for i, hyp in enumerate(hypotheses):
        hyp['cluster_id'] = i % 2  # grouping into 2 clusters: even and odd
    return hypotheses


class ClusteringUtility:
    """
    ClusteringUtility provides methods for grouping hypotheses based on similarity,
    as described in the AI Co-scientist paper.
    """
    
    def __init__(self):
        pass

    def group(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Groups hypotheses by invoking the cluster_hypotheses function.
        
        This is a simplified version that uses the basic clustering function.
        For more sophisticated clustering, use cluster_with_proximity.
        """
        logging.info("Clustering %d hypotheses using simple method", len(hypotheses))
        return cluster_hypotheses(hypotheses)
    
    def cluster_with_proximity(self, hypotheses: List[Dict[str, Any]], 
                              proximity_graph: Dict[str, Any],
                              n_clusters: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Cluster hypotheses using the proximity graph information.
        
        Args:
            hypotheses: List of hypothesis dictionaries
            proximity_graph: Proximity graph as returned by ProximityAgent
            n_clusters: Number of clusters to form (default: sqrt(n))
            
        Returns:
            List of hypotheses with cluster_id field added/updated
        """
        if not hypotheses:
            return []
            
        try:
            logging.info("Clustering %d hypotheses using proximity graph", len(hypotheses))
            
            # Extract proximity matrix
            proximity_matrix = np.array(proximity_graph.get("proximity_matrix", []))
            
            if len(proximity_matrix) == 0:
                logging.warning("Empty proximity matrix, falling back to simple clustering")
                return self.group(hypotheses)
                
            # Convert similarity to distance
            distance_matrix = 1.0 - proximity_matrix
            
            # Determine number of clusters if not specified
            if n_clusters is None:
                n_clusters = max(2, min(int(np.sqrt(len(hypotheses))), 10))
                
            logging.info(f"Using {n_clusters} clusters for {len(hypotheses)} hypotheses")
            
            # Perform hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                linkage='average'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Assign cluster IDs to hypotheses
            for i, hypothesis in enumerate(hypotheses):
                if i < len(cluster_labels):
                    hypothesis['cluster_id'] = int(cluster_labels[i])
                else:
                    hypothesis['cluster_id'] = 0
            
            # Count hypotheses per cluster
            cluster_counts = {}
            for label in cluster_labels:
                cluster_counts[int(label)] = cluster_counts.get(int(label), 0) + 1
                
            logging.info(f"Cluster distribution: {cluster_counts}")
            
            return hypotheses
            
        except Exception as e:
            logging.error(f"Error in proximity-based clustering: {e}")
            logging.warning("Falling back to simple clustering")
            return self.group(hypotheses)
    
    def get_cluster_representatives(self, hypotheses: List[Dict[str, Any]],
                                   n_representatives: int = 1) -> List[Dict[str, Any]]:
        """
        Get representative hypotheses from each cluster.
        
        Args:
            hypotheses: List of hypothesis dictionaries with cluster_id field
            n_representatives: Number of representatives to select per cluster
            
        Returns:
            List of representative hypotheses
        """
        # Group hypotheses by cluster
        clusters = {}
        for hyp in hypotheses:
            cluster_id = hyp.get('cluster_id', 0)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(hyp)
        
        # Select representatives based on ELO score or other metrics
        representatives = []
        for cluster_id, cluster_hyps in clusters.items():
            # Sort by ELO score (descending)
            sorted_hyps = sorted(cluster_hyps, 
                               key=lambda h: h.get('elo_score', 0), 
                               reverse=True)
            
            # Take top n_representatives
            representatives.extend(sorted_hyps[:n_representatives])
        
        return representatives 