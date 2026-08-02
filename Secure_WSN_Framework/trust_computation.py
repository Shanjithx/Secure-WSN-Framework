# trust_computation.py
"""
Trust Computation Module for SECURE_WSN_FRAMEWORK.
Responsible for computing direct, energy, communication, behavioral, and overall trust 
for all nodes in the network using configuration settings, identifying suspicious nodes,
generating trust value CSV reports using pandas, and creating a spatial trust distribution visualization.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import config

class TrustComputation:
    """
    Manages trust calculation, verification, and evaluation for wireless sensor nodes.
    Computes multiple components of trust (Direct, Energy, Communication, Behavioral) 
    and integrates them into a weighted overall trust score using settings from config.py.
    """
    def __init__(self, nodes, base_station):
        """
        Initializes the TrustComputation manager.
        
        Args:
            nodes (list): List of Node objects.
            base_station (BaseStation): The Base Station object.
        """
        self.nodes = nodes
        self.base_station = base_station

    def initialize_trust(self):
        """
        Initializes trust attributes dynamically on each node.
        Alive nodes start with initial values, while dead nodes are set to 0.
        """
        try:
            for node in self.nodes:
                if node.is_alive():
                    node.direct_trust = 1.0
                    node.energy_trust = 1.0
                    node.communication_trust = 1.0
                    node.behavioral_trust = 1.0
                    node.trust_value = 1.0
                    node.is_suspicious = False
                else:
                    node.direct_trust = 0.0
                    node.energy_trust = 0.0
                    node.communication_trust = 0.0
                    node.behavioral_trust = 0.0
                    node.trust_value = 0.0
                    node.is_suspicious = True
        except Exception as e:
            raise RuntimeError(f"Error in initialize_trust: {str(e)}")

    def calculate_direct_trust(self, node=None):
        """
        Calculates the Direct Trust (DT) for nodes, representing direct communication reliability.
        Initially, assume all legitimate alive nodes forward packets correctly (DT = 1.0).
        
        Args:
            node (Node, optional): Single node to evaluate. Defaults to None.
            
        Returns:
            float/None: Direct trust value if a single node is evaluated, else None.
        """
        try:
            if node is not None:
                if not node.is_alive():
                    node.direct_trust = 0.0
                else:
                    # Initially assume 1.0; can be updated dynamically in future routing simulations.
                    node.direct_trust = 1.0
                return node.direct_trust

            for n in self.nodes:
                self.calculate_direct_trust(n)
        except Exception as e:
            raise RuntimeError(f"Error in calculate_direct_trust: {str(e)}")

    def calculate_energy_trust(self, node=None):
        """
        Calculates the Energy Trust (ET) for nodes.
        ET = Current Energy / Initial Energy, normalized and clamped strictly between 0 and 1.
        
        Args:
            node (Node, optional): Single node to evaluate. Defaults to None.
            
        Returns:
            float/None: Energy trust value if a single node is evaluated, else None.
        """
        try:
            if node is not None:
                if not node.is_alive() or getattr(node, 'initial_energy', 0) <= 0:
                    node.energy_trust = 0.0
                else:
                    et = node.energy / node.initial_energy
                    node.energy_trust = min(max(et, 0.0), 1.0)
                return node.energy_trust

            for n in self.nodes:
                self.calculate_energy_trust(n)
        except Exception as e:
            raise RuntimeError(f"Error in calculate_energy_trust: {str(e)}")

    def calculate_communication_trust(self, node=None):
        """
        Calculates the Communication Trust (CT) for nodes.
        Initially assign CT = 1.0 for alive nodes, preparing the framework for packet delivery ratios.
        
        Args:
            node (Node, optional): Single node to evaluate. Defaults to None.
            
        Returns:
            float/None: Communication trust value if a single node is evaluated, else None.
        """
        try:
            if node is not None:
                if not node.is_alive():
                    node.communication_trust = 0.0
                else:
                    # Initially 1.0; ready for future packet success rate logs.
                    node.communication_trust = 1.0
                return node.communication_trust

            for n in self.nodes:
                self.calculate_communication_trust(n)
        except Exception as e:
            raise RuntimeError(f"Error in calculate_communication_trust: {str(e)}")

    def calculate_behavioral_trust(self, node=None):
        """
        Calculates the Behavioral Trust (BT) for nodes.
        Initially assign BT = 1.0 for alive nodes, representing non-malicious routing behavior.
        
        Args:
            node (Node, optional): Single node to evaluate. Defaults to None.
            
        Returns:
            float/None: Behavioral trust value if a single node is evaluated, else None.
        """
        try:
            if node is not None:
                if not node.is_alive():
                    node.behavioral_trust = 0.0
                else:
                    # Initially 1.0; decreases if IDS/anomaly detection reports malicious activity.
                    node.behavioral_trust = 1.0
                return node.behavioral_trust

            for n in self.nodes:
                self.calculate_behavioral_trust(n)
        except Exception as e:
            raise RuntimeError(f"Error in calculate_behavioral_trust: {str(e)}")

    def calculate_overall_trust(self, node=None):
        """
        Calculates the overall trust value using a weighted combination of trust parameters:
        Trust = W_DIRECT_TRUST * DT + W_ENERGY_TRUST * ET + W_COMMUNICATION_TRUST * CT + W_BEHAVIORAL_TRUST * BT
        
        Args:
            node (Node, optional): Single node to evaluate. Defaults to None.
            
        Returns:
            float/None: Overall trust value if a single node is evaluated, else None.
        """
        try:
            if node is not None:
                if not node.is_alive():
                    node.trust_value = 0.0
                else:
                    dt = getattr(node, 'direct_trust', 1.0)
                    et = getattr(node, 'energy_trust', 1.0)
                    ct = getattr(node, 'communication_trust', 1.0)
                    bt = getattr(node, 'behavioral_trust', 1.0)
                    
                    # Compute weighted sum from config parameters
                    overall = (config.W_DIRECT_TRUST * dt) + \
                              (config.W_ENERGY_TRUST * et) + \
                              (config.W_COMMUNICATION_TRUST * ct) + \
                              (config.W_BEHAVIORAL_TRUST * bt)
                    node.trust_value = min(max(overall, 0.0), 1.0)
                return node.trust_value

            for n in self.nodes:
                self.calculate_overall_trust(n)
        except Exception as e:
            raise RuntimeError(f"Error in calculate_overall_trust: {str(e)}")

    def identify_low_trust_nodes(self):
        """
        Flags nodes as suspicious (node.is_suspicious = True) if their overall trust 
        falls below config.TRUST_THRESHOLD. Does not isolate them from the network.
        """
        try:
            for node in self.nodes:
                # Compare overall trust value to configured threshold
                if getattr(node, 'trust_value', 1.0) < config.TRUST_THRESHOLD:
                    node.is_suspicious = True
                else:
                    node.is_suspicious = False
        except Exception as e:
            raise RuntimeError(f"Error in identify_low_trust_nodes: {str(e)}")

    def generate_csv_output(self, filepath="results/trust_values.csv"):
        """
        Saves computed trust values, sub-metrics, status, and classification to a CSV file.
        Uses a pandas DataFrame as requested.
        
        Args:
            filepath (str): Target filepath for CSV file.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data = []
            
            for node in self.nodes:
                cluster_id = getattr(node, 'cluster_id', None)
                cluster_id_str = str(cluster_id) if cluster_id is not None else "None"
                
                data.append({
                    "Node_ID": node.node_id,
                    "Cluster_ID": cluster_id_str,
                    "Direct_Trust": round(getattr(node, 'direct_trust', 0.0), 4),
                    "Energy_Trust": round(getattr(node, 'energy_trust', 0.0), 4),
                    "Communication_Trust": round(getattr(node, 'communication_trust', 0.0), 4),
                    "Behavioral_Trust": round(getattr(node, 'behavioral_trust', 0.0), 4),
                    "Overall_Trust": round(node.trust_value, 4),
                    "Suspicious": getattr(node, 'is_suspicious', False),
                    "Energy": round(node.energy, 4),
                    "Status": node.status
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            print(f"[TRUST] Trust metrics successfully saved to {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error in generate_csv_output: {str(e)}")

    def generate_visualization(self, filepath="graphs/trust_distribution.png"):
        """
        Generates a 2D spatial scatter plot of the sensor network where node colors 
        represent their overall trust score using a colorbar scale (RdYlGn).
        Normal nodes are dots, Cluster Heads are diamond shapes, and the BS is a green triangle.
        
        Args:
            filepath (str): Target path to save the generated graph.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            plt.figure(figsize=(10, 8))
            
            # Plot physical sensing boundary
            plt.plot(
                [0, config.NETWORK_WIDTH, config.NETWORK_WIDTH, 0, 0], 
                [0, 0, config.NETWORK_HEIGHT, config.NETWORK_HEIGHT, 0], 
                color='grey', 
                linestyle='--', 
                linewidth=1.5, 
                label='Sensing Boundary'
            )
            
            # Plot Base Station as a distinct green triangle
            plt.scatter(
                self.base_station.x, 
                self.base_station.y, 
                c='green', 
                marker='^', 
                s=200, 
                edgecolors='black',
                zorder=10
            )
            
            # Label Base Station
            plt.text(
                self.base_station.x + 2, 
                self.base_station.y - 2, 
                f"BS ({self.base_station.x}, {self.base_station.y})", 
                fontsize=10, 
                color='green', 
                fontweight='bold',
                zorder=10
            )
            
            # Split nodes into normal and CH nodes for plotting different markers
            normal_nodes = [n for n in self.nodes if not getattr(n, 'is_ch', False)]
            ch_nodes = [n for n in self.nodes if getattr(n, 'is_ch', False)]
            
            # Draw light connection lines between nodes and their assigned CHs to show network structure
            for node in normal_nodes:
                if node.is_alive() and getattr(node, 'assigned_ch', None) is not None:
                    # Find matching CH
                    ch_obj = next((ch for ch in ch_nodes if ch.node_id == node.assigned_ch), None)
                    if ch_obj:
                        plt.plot(
                            [node.x, ch_obj.x],
                            [node.y, ch_obj.y],
                            color='lightgray',
                            linestyle=':',
                            linewidth=0.8,
                            zorder=1
                        )
            
            # Use RdYlGn colormap: Red (Low Trust), Yellow (Medium Trust), Green (High Trust)
            sc = None
            
            # Plot normal nodes (circles)
            if normal_nodes:
                normal_x = [n.x for n in normal_nodes]
                normal_y = [n.y for n in normal_nodes]
                normal_trust = [n.trust_value for n in normal_nodes]
                
                sc = plt.scatter(
                    normal_x,
                    normal_y,
                    c=normal_trust,
                    cmap='RdYlGn',
                    vmin=0.0,
                    vmax=1.0,
                    marker='o',
                    s=60,
                    edgecolors='black',
                    linewidths=0.5,
                    zorder=3
                )
                
            # Plot Cluster Heads (diamonds, larger, thick borders)
            if ch_nodes:
                ch_x = [ch.x for ch in ch_nodes]
                ch_y = [ch.y for ch in ch_nodes]
                ch_trust = [ch.trust_value for ch in ch_nodes]
                
                sc_ch = plt.scatter(
                    ch_x,
                    ch_y,
                    c=ch_trust,
                    cmap='RdYlGn',
                    vmin=0.0,
                    vmax=1.0,
                    marker='D',
                    s=145,
                    edgecolors='black',
                    linewidths=2.0,
                    zorder=5
                )
                # If no normal nodes are present, use CH scatter object for colorbar
                if sc is None:
                    sc = sc_ch
                    
                # Label Cluster Heads
                for idx, ch in enumerate(ch_nodes):
                    plt.text(
                        ch.x + 2.0, 
                        ch.y + 2.0, 
                        f"CH{ch.node_id}", 
                        fontsize=8, 
                        fontweight='bold', 
                        color='black',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=1),
                        zorder=6
                    )
                    
            # Add colorbar to represent trust values
            if sc is not None:
                cbar = plt.colorbar(sc, label='Trust Score (0.0: Low/Red, 0.5: Med/Yellow, 1.0: High/Green)')
                cbar.set_label('Node Trust Score Scale', fontsize=11, fontweight='bold')
                
            # Create professional legend showing both node roles and trust categories
            legend_handles = [
                mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=10, markeredgecolor='black', label='Base Station (BS)'),
                mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=8, markeredgecolor='black', label='Normal Node'),
                mlines.Line2D([], [], color='gray', marker='D', linestyle='None', markersize=10, markeredgecolor='black', label='Cluster Head (CH)'),
                mlines.Line2D([], [], color='green', marker='s', linestyle='None', markersize=8, label='High Trust (>= 0.80)'),
                mlines.Line2D([], [], color='yellow', marker='s', linestyle='None', markersize=8, markeredgecolor='black', label='Medium Trust (0.50 - 0.79)'),
                mlines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=8, label='Low Trust (< 0.50)')
            ]
            plt.legend(handles=legend_handles, loc='upper right', fontsize=9)
            
            plt.xlim(-5, config.NETWORK_WIDTH + 5)
            plt.ylim(-5, max(config.NETWORK_HEIGHT, self.base_station.y) + 15)
            plt.xlabel('X Coordinate (meters)', fontsize=12)
            plt.ylabel('Y Coordinate (meters)', fontsize=12)
            plt.title('WSN Node Trust Spatial Distribution Map', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.5)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[TRUST] Trust visualization saved to {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error in generate_visualization: {str(e)}")

    def print_statistics(self):
        """
        Prints average, maximum, minimum trust scores, count of suspicious nodes, 
        and average energy to the console.
        """
        try:
            if not self.nodes:
                print("[TRUST ERROR] No nodes in the network to evaluate statistics.")
                return

            trust_values = [n.trust_value for n in self.nodes]
            avg_trust = np.mean(trust_values)
            max_trust = max(trust_values)
            min_trust = min(trust_values)
            
            suspicious_count = sum(1 for n in self.nodes if getattr(n, 'is_suspicious', False))
            
            energies = [n.energy for n in self.nodes]
            avg_energy = np.mean(energies)
            
            dt_values = [getattr(n, 'direct_trust', 0.0) for n in self.nodes]
            avg_dt = np.mean(dt_values)
            
            et_values = [getattr(n, 'energy_trust', 0.0) for n in self.nodes]
            avg_et = np.mean(et_values)
            
            ct_values = [getattr(n, 'communication_trust', 0.0) for n in self.nodes]
            avg_ct = np.mean(ct_values)
            
            bt_values = [getattr(n, 'behavioral_trust', 0.0) for n in self.nodes]
            avg_bt = np.mean(bt_values)

            print("\n" + "="*50)
            print("          WSN TRUST COMPUTATION STATISTICS          ")
            print("="*50)
            print(f"Average Trust Score             : {avg_trust:.4f}")
            print(f"Maximum Trust Score             : {max_trust:.4f}")
            print(f"Minimum Trust Score             : {min_trust:.4f}")
            print(f"Number of Suspicious Nodes      : {suspicious_count}")
            print(f"Average Node Energy             : {avg_energy:.4f} J")
            print(f"Average Direct Trust            : {avg_dt:.4f}")
            print(f"Average Energy Trust            : {avg_et:.4f}")
            print(f"Average Communication Trust     : {avg_ct:.4f}")
            print(f"Average Behavioral Trust        : {avg_bt:.4f}")
            print("="*50 + "\n")
        except Exception as e:
            raise RuntimeError(f"Error in print_statistics: {str(e)}")

    def compute_trust(self):
        """
        Orchestration method to run the entire trust computation workflow.
        """
        self.initialize_trust()
        self.calculate_direct_trust()
        self.calculate_energy_trust()
        self.calculate_communication_trust()
        self.calculate_behavioral_trust()
        self.calculate_overall_trust()
        self.identify_low_trust_nodes()
        self.generate_csv_output()
        self.generate_visualization()
        self.print_statistics()
