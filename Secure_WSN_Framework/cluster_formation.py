# cluster_formation.py
"""
Cluster Formation Module for SECURE_WSN_FRAMEWORK.
Responsible for assigning normal sensor nodes to their nearest Cluster Head (CH),
generating statistics, CSV output, and a beautiful cluster visualization.
"""

import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config

class ClusterFormation:
    """
    Manages the cluster formation process where normal sensor nodes are grouped 
    under the nearest Cluster Head selected in Module 2.
    """
    def __init__(self, nodes, base_station):
        """
        Initializes the ClusterFormation manager.
        
        Args:
            nodes (list): List of Node objects.
            base_station (BaseStation): The Base Station object.
        """
        self.nodes = nodes
        self.base_station = base_station
        self.cluster_heads = []
        self.cluster_dict = {}
        self.stats = []

    def identify_cluster_heads(self):
        """
        Identifies and returns all active Cluster Heads in the network.
        
        Returns:
            list: Active Cluster Head Node objects.
        """
        self.cluster_heads = [node for node in self.nodes if node.is_alive() and getattr(node, 'is_ch', False)]
        return self.cluster_heads

    def assign_nodes_to_clusters(self):
        """
        Assigns each alive sensor node to its nearest Cluster Head.
        Saves coordinates and link distances. Sets own ID for Cluster Heads.
        """
        chs = self.identify_cluster_heads()
        if not chs:
            print("[CLUSTERING ERROR] No active Cluster Heads identified. Assigning all to fallback.")
            return

        for node in self.nodes:
            if not node.is_alive():
                # Reset clustering attributes for dead nodes
                node.cluster_id = None
                node.assigned_ch = None
                node.distance_to_ch = float('inf')
                continue

            if getattr(node, 'is_ch', False):
                # Cluster Head assigns to itself
                node.cluster_id = node.node_id
                node.assigned_ch = node.node_id
                node.distance_to_ch = 0.0
            else:
                # Find nearest CH for normal sensor nodes
                min_dist = float('inf')
                closest_ch = None
                for ch in chs:
                    dist = node.distance_to(ch)
                    if dist < min_dist:
                        min_dist = dist
                        closest_ch = ch
                
                if closest_ch:
                    node.cluster_id = closest_ch.node_id
                    node.assigned_ch = closest_ch.node_id
                    node.distance_to_ch = min_dist
                else:
                    node.cluster_id = None
                    node.assigned_ch = None
                    node.distance_to_ch = float('inf')

    def create_cluster_dictionary(self):
        """
        Creates a mapping of Cluster Head ID to a list of its member Node objects.
        """
        chs = self.identify_cluster_heads()
        self.cluster_dict = {ch.node_id: [] for ch in chs}
        
        for node in self.nodes:
            if node.is_alive() and not getattr(node, 'is_ch', False):
                ch_id = getattr(node, 'cluster_id', None)
                if ch_id in self.cluster_dict:
                    self.cluster_dict[ch_id].append(node)
                    
        return self.cluster_dict

    def calculate_cluster_statistics(self):
        """
        Computes performance and topological statistics for each cluster.
        """
        self.stats = []
        chs = self.identify_cluster_heads()
        cluster_map = self.create_cluster_dictionary()
        
        for ch in chs:
            members = cluster_map.get(ch.node_id, [])
            num_members = len(members)
            
            if num_members > 0:
                avg_distance = np.mean([node.distance_to_ch for node in members])
            else:
                avg_distance = 0.0
                
            self.stats.append({
                "Cluster_ID": ch.node_id,
                "Cluster_Head_ID": ch.node_id,
                "Number_of_Members": num_members,
                "Average_Distance": round(avg_distance, 4),
                "CH_Energy": round(ch.energy, 4),
                "CH_Trust": round(ch.trust_value, 4)
            })
            
        return self.stats

    def generate_csv_output(self, node_filepath="results/cluster_formation.csv", stats_filepath="results/cluster_statistics.csv"):
        """
        Saves cluster assignment and cluster statistics details into CSV files.
        """
        os.makedirs(os.path.dirname(node_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(stats_filepath), exist_ok=True)
        
        # 1. Generate cluster_formation.csv
        formation_data = []
        for node in self.nodes:
            node_type = "CH" if getattr(node, 'is_ch', False) else "Normal"
            assigned_ch = getattr(node, 'assigned_ch', None)
            distance_to_ch = getattr(node, 'distance_to_ch', None)
            
            # Formatting values for CSV output
            if assigned_ch is None:
                assigned_ch_str = "Dead" if not node.is_alive() else "None"
            else:
                assigned_ch_str = str(assigned_ch)
                
            if distance_to_ch is None or distance_to_ch == float('inf'):
                dist_str = "-inf" if not node.is_alive() else "N/A"
            else:
                dist_str = round(distance_to_ch, 4)

            formation_data.append({
                "Node_ID": node.node_id,
                "Node_Type": node_type,
                "Assigned_CH": assigned_ch_str,
                "Distance_to_CH": dist_str,
                "Energy": round(node.energy, 4),
                "Trust_Value": round(node.trust_value, 4)
            })
            
        df_formation = pd.DataFrame(formation_data)
        df_formation.to_csv(node_filepath, index=False)
        print(f"[CLUSTERING] Node cluster assignments saved to {node_filepath}")
        
        # 2. Generate cluster_statistics.csv
        df_stats = pd.DataFrame(self.stats)
        df_stats.to_csv(stats_filepath, index=False)
        print(f"[CLUSTERING] Cluster statistics saved to {stats_filepath}")

    def generate_visualization(self, filepath="graphs/cluster_formation.png"):
        """
        Generates a 2D network plot of WSN showing cluster associations.
        Links normal nodes to their corresponding Cluster Head with colored lines.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.figure(figsize=(10, 8))
        
        # Plot sensed boundary
        plt.plot(
            [0, config.NETWORK_WIDTH, config.NETWORK_WIDTH, 0, 0], 
            [0, 0, config.NETWORK_HEIGHT, config.NETWORK_HEIGHT, 0], 
            color='grey', 
            linestyle='--', 
            linewidth=1.5, 
            label='Sensing Boundary'
        )
        
        # Plot Base Station (green triangle)
        plt.scatter(
            self.base_station.x, 
            self.base_station.y, 
            c='green', 
            marker='^', 
            s=180, 
            label='Base Station (BS)', 
            edgecolors='black',
            zorder=6
        )
        
        # Label Base Station
        plt.text(
            self.base_station.x + 2, 
            self.base_station.y - 2, 
            f"BS ({self.base_station.x}, {self.base_station.y})", 
            fontsize=10, 
            color='green', 
            fontweight='bold'
        )

        chs = self.identify_cluster_heads()
        
        # Generate custom distinct colors for each cluster
        cmap = plt.colormaps.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(chs))]
        
        # Create map for fast access to CH colors
        ch_color_map = {chs[i].node_id: colors[i] for i in range(len(chs))}
        
        # Plot dead nodes
        dead_nodes = [n for n in self.nodes if not n.is_alive()]
        if dead_nodes:
            plt.scatter(
                [n.x for n in dead_nodes],
                [n.y for n in dead_nodes],
                c='red',
                marker='x',
                s=40,
                label='Dead Nodes',
                alpha=0.5
            )

        # Plot normal member nodes and link lines
        for idx, ch in enumerate(chs):
            color = ch_color_map[ch.node_id]
            members = self.cluster_dict.get(ch.node_id, [])
            
            # Plot members of this specific cluster
            if members:
                plt.scatter(
                    [n.x for n in members],
                    [n.y for n in members],
                    color=color,
                    marker='o',
                    s=45,
                    edgecolors='black',
                    linewidths=0.5,
                    alpha=0.8
                )
                
                # Draw line from each normal node to its CH
                for member in members:
                    plt.plot(
                        [member.x, ch.x],
                        [member.y, ch.y],
                        color=color,
                        linestyle='-',
                        linewidth=1.0,
                        alpha=0.5
                    )
            
            # Plot the Cluster Head itself
            plt.scatter(
                ch.x,
                ch.y,
                color=color,
                marker='D',  # Diamond marker for CH
                s=120,
                edgecolors='black',
                linewidths=1.5,
                zorder=5
            )
            
            # Label Cluster Head
            plt.text(
                ch.x + 1.8, 
                ch.y + 1.8, 
                f"CH{idx + 1}", 
                fontsize=9, 
                fontweight='bold', 
                color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                zorder=10
            )

        plt.xlim(-5, config.NETWORK_WIDTH + 5)
        plt.ylim(-5, max(config.NETWORK_HEIGHT, self.base_station.y) + 15)
        plt.xlabel('X Coordinate (meters)', fontsize=12)
        plt.ylabel('Y Coordinate (meters)', fontsize=12)
        plt.title('WSN Cluster Formation Topology Map', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right', fontsize=10)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[CLUSTERING] Cluster formation visual map saved to {filepath}")

    def print_statistics(self):
        """
        Prints detailed cluster formation and size metrics on the console.
        """
        print("\n" + "="*50)
        print("          WSN CLUSTER FORMATION STATISTICS          ")
        print("="*50)
        total_members = 0
        for cluster in self.stats:
            print(f"Cluster Head {cluster['Cluster_Head_ID']}:")
            print(f"  - Members            : {cluster['Number_of_Members']}")
            print(f"  - Avg Member Distance: {cluster['Average_Distance']:.2f} meters")
            print(f"  - CH Energy          : {cluster['CH_Energy']:.4f} J")
            print(f"  - CH Trust Value     : {cluster['CH_Trust']:.2f}")
            print("-" * 50)
            total_members += cluster['Number_of_Members']
            
        avg_size = total_members / len(self.stats) if self.stats else 0
        print(f"Total Clusters Created  : {len(self.stats)}")
        print(f"Average Cluster Size    : {avg_size:.2f} nodes")
        print("="*50 + "\n")

    def form_clusters(self):
        """
        Orchestration method to execute the complete cluster formation process.
        """
        self.assign_nodes_to_clusters()
        self.create_cluster_dictionary()
        self.calculate_cluster_statistics()
        self.generate_csv_output()
        self.generate_visualization()
        self.print_statistics()
        return self.cluster_dict
