# deployment.py
"""
Node Deployment Module for SECURE_WSN_FRAMEWORK.
Responsible for representing sensor nodes and the base station, generating nodes 
in a random spatial distribution, plotting their positions, and saving deployment records.
"""

import os
import csv
import numpy as np
import matplotlib
# Use a non-interactive backend for Matplotlib to avoid display/GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config

class Node:
    """
    Class to represent a sensor node in the WSN.
    """
    def __init__(self, node_id, x, y, initial_energy):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.energy = initial_energy
        self.initial_energy = initial_energy
        self.status = "Alive"   # Status: 'Alive' or 'Dead'
        self.is_ch = False      # Status check if node is selected as Cluster Head
        self.cluster_id = None  # ID of the cluster head associated with this node
        self.trust_value = 1.0  # Trust value initialized to 1.0 (range [0, 1])

    def is_alive(self):
        """
        Returns True if the node is alive (status is 'Alive'), False otherwise.
        """
        return self.status == "Alive"

    def distance_to(self, other_node):
        """
        Calculates and returns the Euclidean distance between this node and another 
        node instance, the base station, or a coordinate tuple (x, y).
        """
        if hasattr(other_node, 'x') and hasattr(other_node, 'y'):
            ox, oy = other_node.x, other_node.y
        elif isinstance(other_node, (list, tuple)) and len(other_node) == 2:
            ox, oy = other_node[0], other_node[1]
        else:
            raise TypeError("other_node must be a Node, BaseStation, or a coordinate tuple (x, y)")
        
        return ((self.x - ox) ** 2 + (self.y - oy) ** 2) ** 0.5

    def update_energy(self, energy_spent):
        """
        Deducts energy and updates node status if it runs out of energy.
        """
        if not self.is_alive():
            return
        
        self.energy = max(0.0, self.energy - energy_spent)
        if self.energy <= 0:
            self.status = "Dead"
            self.is_ch = False
            self.cluster_id = None

    def __repr__(self):
        return (f"Node(ID: {self.node_id}, X: {self.x:.2f}, Y: {self.y:.2f}, "
                f"Energy: {self.energy:.2f}J, Status: {self.status}, Trust: {self.trust_value:.2f})")


class BaseStation:
    """
    Class to represent the Base Station (Sink Node) in the WSN.
    The base station typically has unlimited power and serves as the destination for routing.
    """
    def __init__(self, x, y):
        self.node_id = "BS"
        self.x = x
        self.y = y
        self.energy = float('inf')  # Infinite energy supply
        self.status = "Alive"

    def distance_to(self, other_node):
        """
        Calculates and returns the Euclidean distance between the base station and 
        another node instance or coordinate tuple (x, y).
        """
        if hasattr(other_node, 'x') and hasattr(other_node, 'y'):
            ox, oy = other_node.x, other_node.y
        elif isinstance(other_node, (list, tuple)) and len(other_node) == 2:
            ox, oy = other_node[0], other_node[1]
        else:
            raise TypeError("other_node must be a Node or a coordinate tuple (x, y)")
        
        return ((self.x - ox) ** 2 + (self.y - oy) ** 2) ** 0.5

    def __repr__(self):
        return f"BaseStation(ID: {self.node_id}, X: {self.x:.2f}, Y: {self.y:.2f})"


def generate_nodes():
    """
    Generates NUM_NODES sensor nodes with random spatial coordinates (X, Y) 
    within the NETWORK_WIDTH by NETWORK_HEIGHT sensing field, initializing each 
    with INITIAL_ENERGY. Node IDs start from 1 up to NUM_NODES.
    """
    np.random.seed(config.RANDOM_SEED)
    nodes = []
    for i in range(1, config.NUM_NODES + 1):
        x = np.random.uniform(0, config.NETWORK_WIDTH)
        y = np.random.uniform(0, config.NETWORK_HEIGHT)
        node = Node(node_id=i, x=x, y=y, initial_energy=config.INITIAL_ENERGY)
        nodes.append(node)
    return nodes


def save_node_info(nodes):
    """
    Saves the coordinates, energy status, and other initial attributes 
    of all generated sensor nodes to a CSV file.
    """
    # Ensure results directory exists
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    
    with open(config.NODE_INFO_CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Node_ID", "X_Coordinate", "Y_Coordinate", "Initial_Energy", "Current_Energy", "Status", "Trust_Value"])
        # Write node details
        for node in nodes:
            writer.writerow([
                node.node_id, 
                round(node.x, 2), 
                round(node.y, 2), 
                node.initial_energy, 
                round(node.energy, 2), 
                node.status, 
                node.trust_value
            ])


def plot_network_deployment(nodes, base_station):
    """
    Visualizes the physical deployment of sensor nodes and the base station in a 2D plane.
    Saves the visual plot as a PNG image in the graphs directory.
    """
    # Ensure graphs directory exists
    os.makedirs(config.GRAPH_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Extract node positions
    node_xs = [node.x for node in nodes]
    node_ys = [node.y for node in nodes]
    
    # Plot normal sensor nodes (blue dots)
    plt.scatter(node_xs, node_ys, c='blue', marker='o', s=40, label='Sensor Nodes (Alive)', alpha=0.7, edgecolors='darkblue')
    
    # Plot Base Station (BS) in green
    plt.scatter(base_station.x, base_station.y, c='green', marker='^', s=150, label='Base Station (BS)', edgecolors='black')
    
    # Draw boundary of sensing field (0,0) to (100,100)
    plt.plot([0, config.NETWORK_WIDTH, config.NETWORK_WIDTH, 0, 0], 
             [0, 0, config.NETWORK_HEIGHT, config.NETWORK_HEIGHT, 0], 
             color='grey', linestyle='--', linewidth=1.5, label='Sensing Boundary')
    
    # Setup grid and axes limits
    plt.xlim(-5, config.NETWORK_WIDTH + 5)
    plt.ylim(-5, max(config.NETWORK_HEIGHT, base_station.y) + 15)
    plt.xlabel('X Coordinate (meters)', fontsize=12)
    plt.ylabel('Y Coordinate (meters)', fontsize=12)
    plt.title('Wireless Sensor Network Node Deployment (Refactored)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    
    # Label Base Station coordinates in green
    plt.text(base_station.x + 2, base_station.y - 2, f"BS ({base_station.x}, {base_station.y})", 
             fontsize=10, color='green', fontweight='bold')
    
    # Save image
    plt.savefig(config.DEPLOYMENT_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()


def deploy_network():
    """
    Main function to execute the deployment process.
    """
    print("[DEPLOYMENT] Initializing Wireless Sensor Network deployment...")
    
    # Generate sensor nodes
    nodes = generate_nodes()
    print(f"[DEPLOYMENT] Successfully generated {len(nodes)} sensor nodes (IDs 1 to {len(nodes)}).")
    
    # Create Base Station using tuple unpacking
    base_station = BaseStation(*config.BASE_STATION)
    print(f"[DEPLOYMENT] Positioned Base Station at {config.BASE_STATION}.")
    
    # Save node data to CSV file
    save_node_info(nodes)
    print(f"[DEPLOYMENT] Saved node metadata to {config.NODE_INFO_CSV_PATH}.")
    
    # Plot and save WSN network topology
    plot_network_deployment(nodes, base_station)
    print(f"[DEPLOYMENT] Plot saved as network topology image to {config.DEPLOYMENT_PLOT_PATH}.")
    
    return nodes, base_station


if __name__ == "__main__":
    # Test script locally
    nodes, bs = deploy_network()
