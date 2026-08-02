# soa_routing.py
"""
Secure Routing Module using Seagull Optimization Algorithm (SOA) for SECURE_WSN_FRAMEWORK.
Determines optimal multi-hop paths from all active Cluster Heads to the Base Station.
Avoids suspicious/low-trust nodes using a weighted cost function, updates energy levels 
based on the Heinzelman first-order radio energy model, estimates delay, and outputs CSV files and plots.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import config

class SOARouting:
    """
    Manages the Seagull Optimization Algorithm (SOA) for secure, trust-based,
    and energy-efficient multi-hop routing in wireless sensor networks.
    """
    def __init__(self, nodes, base_station):
        """
        Initializes the SOARouting engine.
        
        Args:
            nodes (list): List of Node objects.
            base_station (BaseStation): The Base Station object.
        """
        self.nodes = nodes
        self.base_station = base_station
        self.cluster_heads = []
        self.selected_routes = {}  # Maps CH node_id to list of Node objects representing the path
        
        # History lists for collective network convergence tracking
        self.history_best = np.zeros(config.SOA_ITERATIONS)
        self.history_avg = np.zeros(config.SOA_ITERATIONS)
        self.history_worst = np.zeros(config.SOA_ITERATIONS)

    def identify_cluster_heads(self):
        """
        Identifies and returns all active Cluster Heads in the network.
        
        Returns:
            list: Active Cluster Head Node objects.
        """
        try:
            self.cluster_heads = [node for node in self.nodes if node.is_alive() and getattr(node, 'is_ch', False)]
            return self.cluster_heads
        except Exception as e:
            raise RuntimeError(f"Error in identify_cluster_heads: {str(e)}")

    def generate_candidate_paths(self, ch):
        """
        Generates loop-free candidate paths from a Cluster Head to the Base Station
        where each link distance is within config.MAX_HOP_DISTANCE.
        
        Args:
            ch (Node): Source Cluster Head node.
            
        Returns:
            list of list of Node: Candidate paths starting with ch and ending with base_station.
        """
        try:
            candidate_paths = []
            max_hops = 6  # Restrict path depth to prevent path explosion in large networks
            max_paths_limit = 500  # Enforce a safety cap on candidate paths to avoid memory/CPU explosion
            
            # DFS helper function to find paths
            def dfs(current_node, current_path):
                if len(candidate_paths) >= max_paths_limit:
                    return
                if len(current_path) > max_hops:
                    return
                
                # If Base Station is within max transmission range, we can terminate path
                dist_to_bs = current_node.distance_to(self.base_station)
                if dist_to_bs <= config.MAX_HOP_DISTANCE:
                    candidate_paths.append(current_path + [self.base_station])
                    return  # BASE CASE reached: direct path to BS, no need to search further
                
                # Traverse neighbors
                for neighbor in self.nodes:
                    if len(candidate_paths) >= max_paths_limit:
                        return
                    # Neighbor must be alive, not already visited, and not another Cluster Head (preferred routing)
                    if neighbor.is_alive() and neighbor not in current_path and not getattr(neighbor, 'is_ch', False):
                        dist = current_node.distance_to(neighbor)
                        if dist <= config.MAX_HOP_DISTANCE:
                            dfs(neighbor, current_path + [neighbor])

            # Try to search using normal nodes as relays first
            dfs(ch, [ch])
            
            # If no paths were found, allow traversing other Cluster Heads as relays
            if not candidate_paths:
                def dfs_with_chs(current_node, current_path):
                    if len(candidate_paths) >= max_paths_limit:
                        return
                    if len(current_path) > max_hops:
                        return
                    
                    dist_to_bs = current_node.distance_to(self.base_station)
                    if dist_to_bs <= config.MAX_HOP_DISTANCE:
                        candidate_paths.append(current_path + [self.base_station])
                        return
                    
                    for neighbor in self.nodes:
                        if len(candidate_paths) >= max_paths_limit:
                            return
                        if neighbor.is_alive() and neighbor not in current_path:
                            dist = current_node.distance_to(neighbor)
                            if dist <= config.MAX_HOP_DISTANCE:
                                dfs_with_chs(neighbor, current_path + [neighbor])
                dfs_with_chs(ch, [ch])

            # Fallback: if still no paths exist, connect directly to the Base Station (single-hop)
            if not candidate_paths:
                candidate_paths.append([ch, self.base_station])
                
            return candidate_paths
        except Exception as e:
            raise RuntimeError(f"Error in generate_candidate_paths: {str(e)}")

    def calculate_routing_cost(self, path):
        """
        Computes the routing cost score for a path.
        Routing Cost = (TRUST_WEIGHT * Trust Score) + (ENERGY_WEIGHT * Energy Score) 
                       + (DISTANCE_WEIGHT * Inverse Distance) + (HOP_WEIGHT * Hop Score)
        
        Higher score is better. Node score is calculated for each sender in the path.
        
        Args:
            path (list of Node): Sequence of hops.
            
        Returns:
            float: Mean routing cost of the path.
        """
        try:
            if len(path) <= 1:
                return -float('inf')
            
            total_score = 0.0
            # Evaluate all nodes that transmit packets along this path (excluding BS)
            for node in path[:-1]:
                # 1. Trust Score
                # If node is suspicious, apply a heavy routing penalty
                if getattr(node, 'is_suspicious', False):
                    trust_score = 0.0
                    penalty = -50.0
                else:
                    trust_score = getattr(node, 'trust_value', 1.0)
                    penalty = 0.0
                    
                # 2. Energy Score
                energy_score = node.energy / config.INITIAL_ENERGY if config.INITIAL_ENERGY > 0 else 0.0
                
                # 3. Inverse Distance (normalized distance progress towards BS)
                dist_to_bs = node.distance_to(self.base_station)
                inv_dist = 1.0 / (dist_to_bs + 1.0)
                
                # 4. Hop Score (estimated remaining hops)
                est_hops = math.ceil(dist_to_bs / config.MAX_HOP_DISTANCE) if config.MAX_HOP_DISTANCE > 0 else 1
                hop_score = 1.0 / (est_hops + 1.0)
                
                # Weighted Linear Cost Function
                node_cost = (config.TRUST_WEIGHT * trust_score) + \
                            (config.ENERGY_WEIGHT * energy_score) + \
                            (config.DISTANCE_WEIGHT * inv_dist) + \
                            (config.HOP_WEIGHT * hop_score) + \
                            penalty
                
                total_score += node_cost
                
            # Path routing cost is the average cost of the hops
            return total_score / (len(path) - 1)
        except Exception as e:
            raise RuntimeError(f"Error in calculate_routing_cost: {str(e)}")

    def initialize_population(self, pop_size, K):
        """
        Initializes the continuous position vector of seagulls.
        Each position maps to a path index in range [0, K-1].
        
        Args:
            pop_size (int): Seagull population size.
            K (int): Number of candidate paths.
            
        Returns:
            np.ndarray: Initial positions.
        """
        try:
            return np.random.uniform(0.0, max(0.0, K - 1.0), pop_size)
        except Exception as e:
            raise RuntimeError(f"Error in initialize_population: {str(e)}")

    def boundary_check(self, positions, K):
        """
        Clamps seagull positions strictly within search space boundaries [0, K-1].
        
        Args:
            positions (np.ndarray): Seagull positions.
            K (int): Number of candidate paths.
            
        Returns:
            np.ndarray: Boundary-checked positions.
        """
        try:
            return np.clip(positions, 0.0, max(0.0, K - 1.0))
        except Exception as e:
            raise RuntimeError(f"Error in boundary_check: {str(e)}")

    def evaluate_population(self, positions, candidate_paths):
        """
        Evaluates the routing fitness (cost score) for each seagull in the population.
        
        Args:
            positions (np.ndarray): Seagull positions.
            candidate_paths (list): Candidate paths list.
            
        Returns:
            list: Fitness values.
        """
        try:
            fitness = []
            K = len(candidate_paths)
            for pos in positions:
                idx = int(round(pos)) % K
                path = candidate_paths[idx]
                fitness.append(self.calculate_routing_cost(path))
            return fitness
        except Exception as e:
            raise RuntimeError(f"Error in evaluate_population: {str(e)}")

    def exploration_phase(self, pos, best_pos, A):
        """
        Migration step of the Seagull Optimization Algorithm.
        Avoids collisions and moves towards the best seagull.
        
        Args:
            pos (float): Position of the current seagull.
            best_pos (float): Position of the best seagull.
            A (float): Collision avoidance coefficient.
            
        Returns:
            float: Distance vector Ds.
        """
        # Cs represents collision avoidance: Cs = A * Xp(t)
        Cs = A * pos
        
        # Ms represents movement towards the best seagull: Ms = B * (Xbest - Xp(t))
        r = np.random.rand()
        B = 2 * (A ** 2) * r
        Ms = B * (best_pos - pos)
        
        # Distance Ds = |Cs + Ms|
        Ds = abs(Cs + Ms)
        return Ds

    def exploitation_phase(self, Ds):
        """
        Attacking behavior of the Seagull Optimization Algorithm.
        Uses spiral coordinate transformation in space.
        
        Args:
            Ds (float): Distance vector.
            
        Returns:
            float: Spiral coordinate translation along the search dimension.
        """
        theta = np.random.uniform(0, 2 * np.pi)
        u = 1.0
        v = 0.1  # Spiral shaping coefficient
        r_spiral = u * math.exp(v * theta)  # r = u * e^(v * theta)
        
        # Spiral attack projection: x = r * cos(theta)
        x_spiral = r_spiral * math.cos(theta)
        
        # Displacement = Ds * x'
        displacement = Ds * x_spiral
        return displacement

    def update_positions(self, pos, best_pos, Ds, displacement):
        """
        Combines exploration and exploitation displacements to compute new position.
        Position = displacement + Xbest
        """
        return displacement + best_pos

    def find_best_route(self, ch, candidate_paths):
        """
        Optimizes candidate routing paths using the Seagull Optimization Algorithm (SOA).
        
        Args:
            ch (Node): The Cluster Head.
            candidate_paths (list): List of candidate paths.
            
        Returns:
            list of Node: The optimal route found by SOA.
        """
        try:
            K = len(candidate_paths)
            if K == 0:
                return [ch, self.base_station]
            if K == 1:
                return candidate_paths[0]
                
            pop_size = config.SOA_POPULATION
            max_iter = config.SOA_ITERATIONS
            
            # Initialize positions and tracking variables
            positions = self.initialize_population(pop_size, K)
            best_pos = positions[0]
            best_fitness = -float('inf')
            
            ch_best_history = []
            ch_avg_history = []
            ch_worst_history = []
            
            for t in range(max_iter):
                # Evaluate fitness
                fitness = self.evaluate_population(positions, candidate_paths)
                
                # Check for global best seagull
                for i in range(pop_size):
                    if fitness[i] > best_fitness:
                        best_fitness = fitness[i]
                        best_pos = positions[i]
                        
                # Log iteration data
                ch_best_history.append(best_fitness)
                ch_avg_history.append(np.mean(fitness))
                ch_worst_history.append(min(fitness))
                
                # Frequency control parameter decreases linearly from 2.0 to 0.0
                fc = 2.0
                A = fc - (t * (fc / max_iter))
                
                # Update seagull positions
                for i in range(pop_size):
                    # Exploration: Avoid collision and compute distance to best
                    Ds = self.exploration_phase(positions[i], best_pos, A)
                    
                    # Exploitation: Spiral attack displacement
                    displacement = self.exploitation_phase(Ds)
                    
                    # Compute new position
                    positions[i] = self.update_positions(positions[i], best_pos, Ds, displacement)
                    
                # Apply boundary clamp
                positions = self.boundary_check(positions, K)
                
            # Accumulate history curves for global performance aggregation
            self.history_best += np.array(ch_best_history)
            self.history_avg += np.array(ch_avg_history)
            self.history_worst += np.array(ch_worst_history)
            
            # Map best continuous position to candidate path index
            best_idx = int(round(best_pos)) % K
            return candidate_paths[best_idx]
        except Exception as e:
            raise RuntimeError(f"Error in find_best_route for CH {ch.node_id}: {str(e)}")

    def calculate_route_energy(self, path):
        """
        Computes the total transmission and receiving energy spent along a path.
        Based on the Heinzelman first-order radio energy model.
        
        Args:
            path (list of Node): Node sequence.
            
        Returns:
            float: Energy consumed in Joules.
        """
        try:
            if len(path) <= 1:
                return 0.0
            
            total_energy = 0.0
            k = config.PACKET_SIZE
            E_elec = config.E_ELEC
            E_fs = config.E_FS
            E_mp = config.E_MP
            d0 = math.sqrt(E_fs / E_mp)
            
            for i in range(len(path) - 1):
                sender = path[i]
                receiver = path[i+1]
                d = sender.distance_to(receiver)
                
                # Heinzelman Tx model:
                # Etx = k * Eelec + k * Efs * d^2 (if d < d0)
                # Etx = k * Eelec + k * Emp * d^4 (if d >= d0)
                if d < d0:
                    e_tx = k * E_elec + k * E_fs * (d ** 2)
                else:
                    e_tx = k * E_elec + k * E_mp * (d ** 4)
                
                total_energy += e_tx
                
                # Rx model: Erx = k * Eelec (Base station has unlimited power, ignore Erx)
                if receiver.node_id != "BS":
                    e_rx = k * E_elec
                    total_energy += e_rx
                    
            return total_energy
        except Exception as e:
            raise RuntimeError(f"Error in calculate_route_energy: {str(e)}")

    def calculate_route_delay(self, path):
        """
        Estimates the end-to-end routing delay along the path.
        Delay = Hops * (Transmission Delay + Processing Delay) + Propagation Delay
        
        Args:
            path (list of Node): Node sequence.
            
        Returns:
            float: Delay in seconds.
        """
        try:
            if len(path) <= 1:
                return 0.0
            
            # WSN Communication parameters
            BANDWIDTH = 250000.0  # 250 kbps (standard IEEE 802.15.4)
            SPEED_OF_LIGHT = 3e8  # 3 * 10^8 m/s
            PROCESSING_DELAY = 0.001  # 1 ms processing delay per hop
            
            # Tx delay = packet size / bandwidth
            transmission_delay = config.PACKET_SIZE / BANDWIDTH
            
            total_delay = 0.0
            for i in range(len(path) - 1):
                sender = path[i]
                receiver = path[i+1]
                dist = sender.distance_to(receiver)
                
                prop_delay = dist / SPEED_OF_LIGHT
                hop_delay = transmission_delay + PROCESSING_DELAY + prop_delay
                total_delay += hop_delay
                
            return total_delay
        except Exception as e:
            raise RuntimeError(f"Error in calculate_route_delay: {str(e)}")

    def calculate_route_hops(self, path):
        """
        Returns the number of hops (edges) in the path.
        """
        return max(0, len(path) - 1)

    def update_node_energy(self, path):
        """
        Deducts energy from transmitting and receiving nodes along the path
        after a successful transmission. Prevents negative energy.
        
        Args:
            path (list of Node): Chosen path.
        """
        try:
            if len(path) <= 1:
                return
            
            k = config.PACKET_SIZE
            E_elec = config.E_ELEC
            E_fs = config.E_FS
            E_mp = config.E_MP
            d0 = math.sqrt(E_fs / E_mp)
            
            for i in range(len(path) - 1):
                sender = path[i]
                receiver = path[i+1]
                d = sender.distance_to(receiver)
                
                # Compute transmission energy to subtract from sender
                if d < d0:
                    e_tx = k * E_elec + k * E_fs * (d ** 2)
                else:
                    e_tx = k * E_elec + k * E_mp * (d ** 4)
                
                if sender.node_id != "BS":
                    # Update sender energy and clamp
                    sender.energy = max(0.0, sender.energy - e_tx)
                    if sender.energy <= 0.0:
                        sender.status = "Dead"
                
                # Compute receiving energy to subtract from relay (if not BS)
                if receiver.node_id != "BS":
                    e_rx = k * E_elec
                    receiver.energy = max(0.0, receiver.energy - e_rx)
                    if receiver.energy <= 0.0:
                        receiver.status = "Dead"
        except Exception as e:
            raise RuntimeError(f"Error in update_node_energy: {str(e)}")

    def generate_csv_output(self):
        """
        Writes routing path evaluations and SOA history convergence curves into CSV files.
        Uses a pandas DataFrame as requested.
        """
        try:
            # 1. Generate routing_paths.csv
            routing_data = []
            for ch in self.cluster_heads:
                path = self.selected_routes.get(ch.node_id, [])
                path_str = " -> ".join(str(n.node_id) for n in path)
                hops = self.calculate_route_hops(path)
                energy = self.calculate_route_energy(path)
                delay = self.calculate_route_delay(path)
                
                # Trust score is the mean trust of nodes in path (excluding BS)
                trust_scores = [n.trust_value for n in path[:-1]]
                avg_path_trust = np.mean(trust_scores) if trust_scores else 0.0
                
                cost = self.calculate_routing_cost(path)
                
                routing_data.append({
                    "Cluster_Head": ch.node_id,
                    "Destination": "BS",
                    "Path": path_str,
                    "Hop_Count": hops,
                    "Energy_Consumed": round(energy, 6),
                    "Delay": round(delay, 6),
                    "Trust": round(avg_path_trust, 4),
                    "Routing_Cost": round(cost, 4)
                })
                
            df_routes = pd.DataFrame(routing_data)
            routes_path = "results/routing_paths.csv"
            os.makedirs(os.path.dirname(routes_path), exist_ok=True)
            df_routes.to_csv(routes_path, index=False)
            print(f"[ROUTING] Secure paths saved to {routes_path}")
            
            # 2. Generate soa_history.csv
            # Combine aggregated curves (averaged over active Cluster Heads)
            num_chs = len(self.cluster_heads) if self.cluster_heads else 1
            history_data = []
            for t in range(config.SOA_ITERATIONS):
                history_data.append({
                    "Iteration": t + 1,
                    "Best_Cost": round(self.history_best[t] / num_chs, 4),
                    "Average_Cost": round(self.history_avg[t] / num_chs, 4),
                    "Worst_Cost": round(self.history_worst[t] / num_chs, 4)
                })
                
            df_history = pd.DataFrame(history_data)
            history_path = "results/soa_history.csv"
            df_history.to_csv(history_path, index=False)
            print(f"[ROUTING] SOA history saved to {history_path}")
        except Exception as e:
            raise RuntimeError(f"Error in generate_csv_output: {str(e)}")

    def generate_visualization(self):
        """
        Generates plots for network routing paths (graphs/secure_routing.png)
        and SOA convergence curves (graphs/soa_convergence.png).
        """
        try:
            # 1. Routing Paths plot
            path_routing = "graphs/secure_routing.png"
            os.makedirs(os.path.dirname(path_routing), exist_ok=True)
            plt.figure(figsize=(10, 8))
            
            # Plot physical boundary
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
                s=200, 
                edgecolors='black',
                zorder=10
            )
            plt.text(
                self.base_station.x + 2, 
                self.base_station.y - 2, 
                f"BS ({self.base_station.x}, {self.base_station.y})", 
                fontsize=10, 
                color='green', 
                fontweight='bold',
                zorder=10
            )
            
            # Plot normal nodes (gray circles)
            normal_nodes = [n for n in self.nodes if not getattr(n, 'is_ch', False)]
            if normal_nodes:
                plt.scatter(
                    [n.x for n in normal_nodes],
                    [n.y for n in normal_nodes],
                    c='lightgray',
                    marker='o',
                    s=45,
                    edgecolors='gray',
                    linewidths=0.5,
                    alpha=0.6
                )
                
            # Plot Cluster Heads (diamonds, larger)
            if self.cluster_heads:
                plt.scatter(
                    [ch.x for ch in self.cluster_heads],
                    [ch.y for ch in self.cluster_heads],
                    c='orange',
                    marker='D',
                    s=120,
                    edgecolors='black',
                    linewidths=1.0,
                    zorder=5
                )
                
            # Plot paths using a unique color for each route
            cmap = plt.colormaps.get_cmap('tab20')
            for idx, ch in enumerate(self.cluster_heads):
                path = self.selected_routes.get(ch.node_id, [])
                if len(path) > 1:
                    color = cmap(idx % 20)
                    xs = [n.x for n in path]
                    ys = [n.y for n in path]
                    
                    # Plot path lines
                    plt.plot(xs, ys, color=color, linestyle='-', linewidth=2.0, alpha=0.8)
                    
                    # Highlight intermediate relay nodes
                    relays = path[1:-1]
                    if relays:
                        plt.scatter(
                            [r.x for r in relays],
                            [r.y for r in relays],
                            color=color,
                            marker='o',
                            s=80,
                            edgecolors='black',
                            linewidths=1.0,
                            zorder=6
                        )
                        
            plt.xlim(-5, config.NETWORK_WIDTH + 5)
            plt.ylim(-5, max(config.NETWORK_HEIGHT, self.base_station.y) + 15)
            plt.xlabel('X Coordinate (meters)', fontsize=12)
            plt.ylabel('Y Coordinate (meters)', fontsize=12)
            plt.title('WSN Secure Multi-hop Routing Paths (SOA)', fontsize=14, fontweight='bold')
            
            # Custom legend
            legend_handles = [
                mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=10, markeredgecolor='black', label='Base Station (BS)'),
                mlines.Line2D([], [], color='orange', marker='D', linestyle='None', markersize=8, markeredgecolor='black', label='Cluster Head (CH)'),
                mlines.Line2D([], [], color='lightgray', marker='o', linestyle='None', markersize=8, markeredgecolor='gray', label='Sensor Node'),
                mlines.Line2D([], [], color='blue', linestyle='-', linewidth=2, label='Routing Path')
            ]
            plt.legend(handles=legend_handles, loc='upper right', fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.savefig(path_routing, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[ROUTING] Routing paths map saved to {path_routing}")
            
            # 2. SOA Convergence Plot
            path_conv = "graphs/soa_convergence.png"
            plt.figure(figsize=(8, 6))
            num_chs = len(self.cluster_heads) if self.cluster_heads else 1
            iterations = np.arange(1, config.SOA_ITERATIONS + 1)
            
            plt.plot(iterations, self.history_best / num_chs, color='green', linestyle='-', linewidth=2.0, label='Best Routing Cost')
            plt.plot(iterations, self.history_avg / num_chs, color='blue', linestyle='--', linewidth=1.5, label='Average Routing Cost')
            
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Routing Cost Score', fontsize=12)
            plt.title('Seagull Optimization Algorithm (SOA) Convergence Trace', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(loc='lower right', fontsize=10)
            plt.savefig(path_conv, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[ROUTING] SOA convergence curve saved to {path_conv}")
        except Exception as e:
            raise RuntimeError(f"Error in generate_visualization: {str(e)}")

    def print_statistics(self):
        """
        Prints detailed routing statistics to the console.
        """
        try:
            if not self.cluster_heads:
                print("[ROUTING ERROR] No Cluster Heads identified. Statistics not available.")
                return

            hops = [self.calculate_route_hops(p) for p in self.selected_routes.values()]
            delays = [self.calculate_route_delay(p) for p in self.selected_routes.values()]
            costs = [self.calculate_routing_cost(p) for p in self.selected_routes.values()]
            energies = [self.calculate_route_energy(p) for p in self.selected_routes.values()]
            
            path_trusts = []
            for path in self.selected_routes.values():
                t_scores = [n.trust_value for n in path[:-1]]
                path_trusts.append(np.mean(t_scores) if t_scores else 0.0)
                
            # Suspicious nodes avoided calculation
            total_suspicious = sum(1 for n in self.nodes if getattr(n, 'is_suspicious', False))
            suspicious_used = set()
            for path in self.selected_routes.values():
                for node in path[:-1]:
                    if getattr(node, 'is_suspicious', False):
                        suspicious_used.add(node.node_id)
            suspicious_avoided = total_suspicious - len(suspicious_used)

            print("\n" + "="*50)
            print("          WSN SECURE SOA ROUTING STATISTICS         ")
            print("="*50)
            print(f"Average Hop Count           : {np.mean(hops):.2f} hops")
            print(f"Average Routing Delay       : {np.mean(delays):.6f} seconds")
            print(f"Average Routing Cost        : {np.mean(costs):.4f}")
            print(f"Average Energy Consumption  : {np.mean(energies):.6f} J")
            print(f"Average Route Trust Score   : {np.mean(path_trusts):.4f}")
            print(f"Suspicious Nodes Avoided    : {suspicious_avoided} / {total_suspicious}")
            print("="*50 + "\n")
        except Exception as e:
            raise RuntimeError(f"Error in print_statistics: {str(e)}")

    def run_soa(self):
        """
        Orchestration method to execute the complete secure multi-hop routing process.
        """
        # Reset tracking history
        self.history_best = np.zeros(config.SOA_ITERATIONS)
        self.history_avg = np.zeros(config.SOA_ITERATIONS)
        self.history_worst = np.zeros(config.SOA_ITERATIONS)
        
        self.identify_cluster_heads()
        
        for ch in self.cluster_heads:
            # 1. Generate paths
            candidate_paths = self.generate_candidate_paths(ch)
            
            # 2. Run SOA to find the best route
            best_route = self.find_best_route(ch, candidate_paths)
            self.selected_routes[ch.node_id] = best_route
            
            # 3. Simulate transmission by deducting energy from the nodes in the path
            self.update_node_energy(best_route)
            
        # 4. Generate outputs
        self.generate_csv_output()
        self.generate_visualization()
        self.print_statistics()
