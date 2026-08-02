# cluster_head_selection.py
"""
Cluster Head Selection Module for SECURE_WSN_FRAMEWORK.

RESEARCH DESIGN DECISIONS & RATIONALE:
1. Why Normalization is Used:
   WSN node attributes possess highly diverse physical dimensions and scale ranges 
   (e.g., residual energy is between 0.0 and 2.0 J, trust is between 0.0 and 1.0, 
   node degree can be 0 to 30+, and distance to the base station can be 20 to 150+ meters). 
   Summing raw values directly would cause the largest scale attributes (such as distance 
   or node degree) to dominate the fitness function, rendering other critical attributes 
   (like trust or energy) ineffective. Min-max normalization scales all metrics into 
   the [0, 1] range, ensuring equal-footing comparison. Clamping prevents small 
   floating-point errors from exceeding these boundaries.

2. Why Weighted Fitness is Preferred:
   In WSN deployment, different optimization goals have different priorities depending 
   on the application context. By using a weighted fitness model (W_ENERGY, W_DISTANCE, 
   W_DEGREE, W_TRUST), we can tune the importance of energy conservation, security, 
   and network coverage without altering the underlying code. The weights sum to 1.0, 
   making the fitness value a convex combination of normalized metrics.

3. Why Communication Circles are Shown:
   Visualizing the communication range (config.TRANSMISSION_RANGE) around each selected 
   Cluster Head (CH) allows researchers to visually evaluate network coverage, overlap 
   redundancy, and confirm whether the selected CH configuration leaves any normal 
   sensor nodes isolated from direct or multi-hop routing paths.

4. How this Module Prepares for Ip-GJOA:
   The Improved Golden Jackal Optimization Algorithm (Ip-GJOA) is a population-based 
   metaheuristic that will optimize the selection of CHs. Rather than optimizing a single 
   node, GJOA optimizes the combination (the set) of CHs. This module prepares for GJOA by:
   - Defining the candidate search space (initialize_candidate_nodes).
   - Binding normalized, clamped metrics (normalized_energy, rank, etc.) onto each node, 
     acting as pre-computed state variables to speed up GJOA's iterative objective calculations.
   - Defining the interface structure (normalize_metrics, calculate_fitness, rank_nodes) 
     so that the greedy solver can be swapped out for GJOA seamlessly in Module 2B.
"""

import os
import math
import matplotlib
# Use a non-interactive backend for Matplotlib to avoid display/GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config

class ClusterHeadSelector:
    """
    Class to manage the Cluster Head (CH) selection framework.
    Uses normalization, weighted fitness functions, and modular components
    to facilitate initial selection and future metaheuristic optimizations.
    """
    def __init__(self, nodes, base_station):
        """
        Initializes the ClusterHeadSelector with a list of nodes and the base station.
        
        Args:
            nodes (list): List of Node objects from the deployment module.
            base_station (BaseStation): The Base Station object.
        """
        self.nodes = nodes
        self.base_station = base_station

    def initialize_candidate_nodes(self):
        """
        Returns all alive nodes that are candidates to become Cluster Heads.
        """
        return [node for node in self.nodes if node.is_alive()]

    def calculate_node_degree(self, node):
        """
        Counts the number of neighboring nodes inside the transmission range.
        
        Args:
            node (Node): The node for which the degree is calculated.
            
        Returns:
            int: The number of neighboring alive nodes.
        """
        if not node.is_alive():
            return 0
        
        # Neighbors are other alive nodes within the transmission range
        degree = sum(
            1 for other in self.nodes
            if other != node and other.is_alive() and node.distance_to(other) <= config.TRANSMISSION_RANGE
        )
        return degree

    def calculate_distance_to_bs(self, node):
        """
        Calculates and returns the Euclidean distance to the base station.
        Uses the distance_to() function from Module 1.
        
        Args:
            node (Node): The node to measure distance from.
            
        Returns:
            float: Distance to the base station.
        """
        return node.distance_to(self.base_station)

    def normalize_metrics(self):
        """
        Normalizes node attributes (Residual Energy, Distance to BS, Node Degree, Trust Value)
        across all alive candidate nodes using Min-Max Normalization to lie between 0 and 1.
        Clamps values to exactly [0, 1] to prevent floating point inaccuracies.
        Stores the normalized values on each node.
        """
        candidates = self.initialize_candidate_nodes()
        if not candidates:
            return

        # Precompute raw degree and distance for all candidates
        for node in candidates:
            node.node_degree = self.calculate_node_degree(node)
            node.distance_to_bs = self.calculate_distance_to_bs(node)

        # Extract attribute lists for min/max boundary calculations
        energies = [n.energy for n in candidates]
        distances = [n.distance_to_bs for n in candidates]
        degrees = [n.node_degree for n in candidates]
        trusts = [n.trust_value for n in candidates]

        # Helper function for Min-Max Normalization
        def min_max_norm(val, min_val, max_val, invert=False):
            if max_val == min_val:
                return 1.0
            norm = (val - min_val) / (max_val - min_val)
            return 1.0 - norm if invert else norm

        # Compute min/max values
        min_e, max_e = min(energies), max(energies)
        min_d, max_d = min(distances), max(distances)
        min_deg, max_deg = min(degrees), max(degrees)
        min_t, max_t = min(trusts), max(trusts)

        # Store normalized and clamped attributes in each candidate node
        for node in candidates:
            raw_norm_e = min_max_norm(node.energy, min_e, max_e)
            raw_norm_d = min_max_norm(node.distance_to_bs, min_d, max_d, invert=True)
            raw_norm_deg = min_max_norm(node.node_degree, min_deg, max_deg)
            raw_norm_t = min_max_norm(node.trust_value, min_t, max_t)

            # Clamp normalized values strictly between 0.0 and 1.0 to handle floating point anomalies
            node.normalized_energy = min(max(raw_norm_e, 0.0), 1.0)
            node.normalized_distance = min(max(raw_norm_d, 0.0), 1.0)
            node.normalized_degree = min(max(raw_norm_deg, 0.0), 1.0)
            node.normalized_trust = min(max(raw_norm_t, 0.0), 1.0)

    def calculate_fitness(self, node=None):
        """
        Calculates the weighted fitness score of candidate nodes.
        If node is specified, computes and returns the fitness for that node.
        Otherwise, computes and stores fitness for all alive nodes.
        
        Fitness = w_energy * Norm_Energy + w_dist * Norm_Dist + w_degree * Norm_Degree + w_trust * Norm_Trust
        
        Args:
            node (Node, optional): Single node to evaluate. Defaults to None.
            
        Returns:
            float/None: Fitness score if a single node is evaluated, else None.
        """
        w_energy = getattr(config, 'W_ENERGY', 0.35)
        w_dist = getattr(config, 'W_DISTANCE', 0.25)
        w_deg = getattr(config, 'W_DEGREE', 0.20)
        w_trust = getattr(config, 'W_TRUST', 0.20)

        # Single node calculation case for backward compatibility
        if node is not None:
            if not node.is_alive():
                node.fitness = -float('inf')
                return node.fitness
            
            # Ensure degree and distance attributes exist
            if not hasattr(node, 'node_degree'):
                node.node_degree = self.calculate_node_degree(node)
            if not hasattr(node, 'distance_to_bs'):
                node.distance_to_bs = self.calculate_distance_to_bs(node)
                
            # If normalized values don't exist, use fallbacks
            norm_e = getattr(node, 'normalized_energy', 1.0)
            norm_d = getattr(node, 'normalized_distance', 1.0)
            norm_deg = getattr(node, 'normalized_degree', 1.0)
            norm_t = getattr(node, 'normalized_trust', 1.0)
            
            node.fitness = (w_energy * norm_e) + (w_dist * norm_d) + (w_deg * norm_deg) + (w_trust * norm_t)
            return node.fitness

        # Calculate for all candidates
        candidates = self.initialize_candidate_nodes()
        for n in candidates:
            norm_e = getattr(n, 'normalized_energy', 1.0)
            norm_d = getattr(n, 'normalized_distance', 1.0)
            norm_deg = getattr(n, 'normalized_degree', 1.0)
            norm_t = getattr(n, 'normalized_trust', 1.0)
            
            n.fitness = (w_energy * norm_e) + (w_dist * norm_d) + (w_deg * norm_deg) + (w_trust * norm_t)

        # Set dead nodes attributes to default values
        for n in self.nodes:
            if not n.is_alive():
                n.normalized_energy = 0.0
                n.normalized_distance = 0.0
                n.normalized_degree = 0.0
                n.normalized_trust = 0.0
                n.fitness = -float('inf')

    def rank_nodes(self):
        """
        Sorts the candidate nodes based on fitness and assigns ranks (1 = highest fitness).
        Stores 'rank' inside each node.
        """
        candidates = self.initialize_candidate_nodes()
        # Sort candidates descending by fitness
        candidates.sort(key=lambda n: getattr(n, 'fitness', -float('inf')), reverse=True)
        
        for idx, node in enumerate(candidates):
            node.rank = idx + 1
            
        # Assign fallback rank to dead nodes
        for node in self.nodes:
            if not node.is_alive():
                node.rank = len(self.nodes) + 1

    def select_cluster_heads(self):
        """
        Selects the top percentage of candidate nodes as Cluster Heads based on rank.
        Uses config.CH_PERCENTAGE and math.ceil for ceiling logic.
        Marks node.is_ch = True for selected nodes.
        
        Returns:
            list: Selected Cluster Head nodes.
        """
        # Reset CH flag for all nodes
        for node in self.nodes:
            node.is_ch = False
            
        candidates = self.initialize_candidate_nodes()
        if not candidates:
            return []
            
        ch_percentage = getattr(config, 'CH_PERCENTAGE', 0.10)
        num_ch = max(1, math.ceil(len(candidates) * ch_percentage))
        
        # Sort candidate list by rank
        sorted_candidates = sorted(candidates, key=lambda n: getattr(n, 'rank', len(self.nodes)))
        selected_chs = sorted_candidates[:num_ch]
        
        for node in selected_chs:
            node.is_ch = True
            
        return selected_chs

    def select_initial_cluster_heads(self):
        """
        Backward-compatible wrapper method to execute the complete modular pipeline.
        Runs normalization, fitness calculation, ranking, and CH selection.
        
        Returns:
            list: Selected Cluster Head nodes.
        """
        self.normalize_metrics()
        self.calculate_fitness()
        self.rank_nodes()
        return self.select_cluster_heads()

    def initialize_population(self, pop_size=30):
        """
        Creates the initial jackal population.
        Each jackal is represented by a dictionary containing:
        - 'position': numpy array of continuous values corresponding to candidate indices
        - 'solution': list of unique alive candidate node IDs
        - 'fitness': float representing the fitness of this solution set
        """
        candidates = self.initialize_candidate_nodes()
        ch_percentage = getattr(config, 'CH_PERCENTAGE', 0.10)
        num_ch = max(1, math.ceil(len(candidates) * ch_percentage))
        
        population = []
        for _ in range(pop_size):
            # Generate random candidate indices without replacement
            indices = np.random.choice(len(candidates), size=num_ch, replace=False)
            position = indices.astype(float)
            
            # Use boundary_check to format and get the solution
            position, solution = self.boundary_check(position, candidates)
            fitness = self.evaluate_solution(solution)
            
            population.append({
                'position': position,
                'solution': solution,
                'fitness': fitness
            })
            
        return population

    def evaluate_solution(self, solution):
        """
        Calculates the quality of a jackal solution (list of node IDs).
        Uses the existing precomputed normalized and weighted fitness of each node.
        
        Fitness of the solution = Average of individual node fitnesses.
        Do NOT create a new fitness equation. Reuse the Module 2A fitness function.
        """
        if not solution:
            return -float('inf')
        
        node_dict = {node.node_id: node for node in self.nodes}
        total_fitness = 0.0
        for node_id in solution:
            node = node_dict.get(node_id)
            if node and node.is_alive():
                # Ensure node fitness is calculated
                fit = getattr(node, 'fitness', None)
                if fit is None:
                    fit = self.calculate_fitness(node)
                total_fitness += fit
            else:
                total_fitness += -100.0  # Big penalty for dead or invalid nodes
                
        return total_fitness / len(solution)

    def initialize_golden_jackals(self, population):
        """
        Finds the Male (best) and Female (second best) Golden Jackals from the population.
        """
        # Sort population descending by fitness
        sorted_pop = sorted(population, key=lambda j: j['fitness'], reverse=True)
        male_jackal = sorted_pop[0].copy()
        female_jackal = sorted_pop[1].copy() if len(sorted_pop) > 1 else sorted_pop[0].copy()
        
        return male_jackal, female_jackal

    def levy_flight(self, d, beta=1.5):
        """
        Generates steps of Levy flight for a dimension d using Mantegna's algorithm.
        
        Equations:
        LF(y) = 0.01 * (u * sigma) / (|v|^(1/beta))
        sigma = [Gamma(1+beta) * sin(pi*beta/2) / (Gamma((1+beta)/2) * beta * 2^((beta-1)/2))]^(1/beta)
        """
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                 (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        
        u = np.random.normal(0, sigma, d)
        v = np.random.normal(0, 1, d)
        
        # Avoid division by zero
        v_abs = np.abs(v)
        v_abs[v_abs == 0] = 1e-9
        
        step = u / (v_abs**(1 / beta))
        return step

    def exploration_phase(self, position, male_pos, female_pos, E):
        """
        Implements Golden Jackal exploration behavior.
        
        Equations:
        Y_1(t) = Y_M(t) - E * |Y_M(t) - r_l * Y(t)|
        Y_2(t) = Y_FM(t) - E * |Y_FM(t) - r_l * Y(t)|
        
        Returns the average of Y_1 and Y_2.
        """
        d = len(position)
        rl = 0.05 * self.levy_flight(d)
        Y_1 = male_pos - E * np.abs(male_pos - rl * position)
        Y_2 = female_pos - E * np.abs(female_pos - rl * position)
        return (Y_1 + Y_2) / 2.0

    def exploitation_phase(self, position, male_pos, female_pos, E):
        """
        Implements Golden Jackal exploitation (hunting) behavior.
        
        Equations:
        Y_1(t) = Y_M(t) - E * |Y_M(t) - Y(t)|
        Y_2(t) = Y_FM(t) - E * |Y_FM(t) - Y(t)|
        
        Returns the average of Y_1 and Y_2.
        """
        Y_1 = male_pos - E * np.abs(male_pos - position)
        Y_2 = female_pos - E * np.abs(female_pos - position)
        return (Y_1 + Y_2) / 2.0

    def update_position(self, population, male_jackal, female_jackal, t, max_iter, candidates):
        """
        Updates every candidate solution during each iteration.
        
        Energy Equation:
        E = E1 * E0
        E1 = 1.5 * (1 - t/T) (Prey energy decay)
        E0 = 2 * rand() - 1 (Escape energy)
        
        If |E| > 1: Exploration
        If |E| <= 1: Exploitation
        """
        male_pos = male_jackal['position']
        female_pos = female_jackal['position']
        
        # Decline linearly from 1.5 to 0
        E1 = 1.5 * (1.0 - (t / max_iter))
        
        updated_population = []
        for jackal in population:
            # Generate random escape energy vector of length num_ch
            num_ch = len(jackal['position'])
            E0 = 2.0 * np.random.rand(num_ch) - 1.0
            E = E1 * E0
            
            # Check phase based on mean magnitude of E
            mean_abs_E = np.mean(np.abs(E))
            
            if mean_abs_E > 1.0:
                new_pos = self.exploration_phase(jackal['position'], male_pos, female_pos, E)
            else:
                new_pos = self.exploitation_phase(jackal['position'], male_pos, female_pos, E)
                
            # Boundary check to clean position and retrieve unique candidate node IDs
            new_pos, new_sol = self.boundary_check(new_pos, candidates)
            new_fit = self.evaluate_solution(new_sol)
            
            # Greedy replacement: only accept if fitness is improved
            if new_fit > jackal['fitness']:
                updated_population.append({
                    'position': new_pos,
                    'solution': new_sol,
                    'fitness': new_fit
                })
            else:
                updated_population.append(jackal.copy())
                
        return updated_population

    def boundary_check(self, position, candidates):
        """
        Ensures that:
        - Cluster Heads are valid sensor nodes.
        - Dead nodes cannot become CH.
        - Duplicate CHs are removed.
        - CH count remains correct.
        """
        n_cand = len(candidates)
        if n_cand == 0:
            return np.array([]), []
            
        num_ch = len(position)
        # Clamp positions to the valid index range [0, n_cand - 1]
        clamped_pos = np.clip(position, 0.0, float(n_cand - 1))
        
        # Round to nearest integer candidate indices
        indices = np.round(clamped_pos).astype(int)
        
        # Filter duplicates and valid indices
        unique_indices = []
        seen = set()
        for idx in indices:
            if idx not in seen and 0 <= idx < n_cand:
                seen.add(idx)
                unique_indices.append(idx)
                
        # If duplicates were removed, fill the empty spots with other unique candidate indices
        if len(unique_indices) < num_ch:
            remaining = [i for i in range(n_cand) if i not in seen]
            if remaining:
                # Add unique candidate indices to fill the gap
                fill_size = min(len(remaining), num_ch - len(unique_indices))
                fill_indices = np.random.choice(remaining, size=fill_size, replace=False)
                for f_idx in fill_indices:
                    unique_indices.append(f_idx)
                    seen.add(f_idx)
                    
        # Ensure we have exactly num_ch elements
        unique_indices = unique_indices[:num_ch]
        
        new_position = np.array(unique_indices, dtype=float)
        solution = [candidates[idx].node_id for idx in unique_indices]
        
        return new_position, solution

    def run_ip_gjoa(self, pop_size=30, max_iter=100):
        """
        Runs the main optimization loop of the Improved Golden Jackal Optimization Algorithm.
        
        Flow:
        1. Initialize population
        2. Evaluate initial fitness (done during initialization)
        3. Identify Male and Female jackals
        4. Loop: update positions, boundary check, select new guides
        5. Track fitness metrics and log to CSV
        """
        # Initialize population
        population = self.initialize_population(pop_size=pop_size)
        
        # Select guides
        male_jackal, female_jackal = self.initialize_golden_jackals(population)
        
        history = []
        candidates = self.initialize_candidate_nodes()
        
        for t in range(1, max_iter + 1):
            # Update population positions
            population = self.update_position(population, male_jackal, female_jackal, t, max_iter, candidates)
            
            # Update guides
            male_jackal, female_jackal = self.initialize_golden_jackals(population)
            
            # Calculate stats for the iteration
            fitnesses = [j['fitness'] for j in population]
            best_fit = male_jackal['fitness']
            avg_fit = np.mean(fitnesses)
            worst_fit = np.min(fitnesses)
            
            history.append({
                "Iteration": t,
                "Best_Fitness": best_fit,
                "Average_Fitness": avg_fit,
                "Worst_Fitness": worst_fit
            })
            
        # Convert history to DataFrame and save to CSV
        history_df = pd.DataFrame(history)
        history_csv_path = "results/ip_gjoa_history.csv"
        os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)
        history_df.to_csv(history_csv_path, index=False)
        print(f"[OPTIMIZATION] Ip-GJOA history saved to {history_csv_path}")
        
        # Generate the convergence graph
        self.generate_gjoa_convergence_graph(history_df)
        
        return male_jackal['solution'], male_jackal['fitness']

    def generate_gjoa_convergence_graph(self, history_df, filepath="graphs/ip_gjoa_convergence.png"):
        """
        Generates a visually stunning convergence graph of the GJOA optimization process.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # Plot best fitness over iterations with a vibrant line
        plt.plot(
            history_df["Iteration"], 
            history_df["Best_Fitness"], 
            color='#E67E22',  # Premium Golden/Orange color
            linewidth=2.5, 
            label='Best Fitness (Male Jackal)'
        )
        
        # Plot average fitness to show population convergence
        plt.plot(
            history_df["Iteration"], 
            history_df["Average_Fitness"], 
            color='#3498DB',  # Sleek Blue color
            linewidth=1.5, 
            linestyle='--', 
            alpha=0.8,
            label='Average Population Fitness'
        )
        
        # Fill area under the best fitness curve with a soft gradient
        plt.fill_between(
            history_df["Iteration"], 
            history_df["Best_Fitness"], 
            history_df["Average_Fitness"],
            color='#E67E22', 
            alpha=0.1,
            label='Fitness Spread'
        )
        
        plt.title('Ip-GJOA Convergence Curve\n(Cluster Head Selection Optimization)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness Value', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6, color='#BDC3C7')
        plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#BDC3C7', fontsize=10)
        
        # Annotate final best fitness
        final_best = history_df["Best_Fitness"].iloc[-1]
        final_iter = history_df["Iteration"].iloc[-1]
        plt.annotate(
            f'Optimal Fitness: {final_best:.4f}',
            xy=(final_iter, final_best),
            xytext=(final_iter - 25, final_best - 0.05 if final_best > 0.1 else final_best + 0.05),
            arrowprops=dict(facecolor='#2C3E50', arrowstyle='->', lw=1.5),
            fontsize=10,
            fontweight='bold',
            color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9F9', edgecolor='#BDC3C7', alpha=0.9)
        )
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"[OPTIMIZATION] Convergence graph successfully saved to {filepath}")

    def select_cluster_heads_ip_gjoa(self, pop_size=30, max_iter=100):
        """
        Executes the Improved Golden Jackal Optimization Algorithm (Ip-GJOA)
        to select the optimal set of Cluster Heads.
        Sets node.is_ch = True for the selected nodes and returns them.
        """
        # Run GJOA
        best_sol, best_fit = self.run_ip_gjoa(pop_size=pop_size, max_iter=max_iter)
        
        # Reset CH flag for all nodes
        for node in self.nodes:
            node.is_ch = False
            
        # Map best solution node IDs back to Node objects
        selected_chs = []
        node_dict = {node.node_id: node for node in self.nodes}
        for node_id in best_sol:
            node = node_dict.get(node_id)
            if node:
                node.is_ch = True
                selected_chs.append(node)
                
        return selected_chs

    def generate_csv_output(self, filepath="results/cluster_heads.csv"):
        """
        Generates a CSV file containing cluster head selection parameters and fitness.
        Replaces csv.writer with pandas DataFrame.
        
        Args:
            filepath (str): Path to save the CSV file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = []
        for node in self.nodes:
            # Ensure attributes exist or compute fallbacks
            node_degree = getattr(node, 'node_degree', self.calculate_node_degree(node))
            distance_to_bs = getattr(node, 'distance_to_bs', self.calculate_distance_to_bs(node))
            
            data.append({
                "Node_ID": node.node_id,
                "Energy": round(node.energy, 4),
                "Normalized_Energy": round(getattr(node, 'normalized_energy', 0.0), 4),
                "Trust": round(node.trust_value, 4),
                "Normalized_Trust": round(getattr(node, 'normalized_trust', 0.0), 4),
                "Distance_to_BS": round(distance_to_bs, 4),
                "Normalized_Distance": round(getattr(node, 'normalized_distance', 0.0), 4),
                "Node_Degree": node_degree,
                "Normalized_Degree": round(getattr(node, 'normalized_degree', 0.0), 4),
                "Fitness": round(node.fitness, 4) if getattr(node, 'fitness', -float('inf')) != -float('inf') else "-inf",
                "Rank": getattr(node, 'rank', len(self.nodes) + 1),
                "Is_Cluster_Head": node.is_ch
            })
            
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"[CLUSTERING] CSV output successfully saved to {filepath} using Pandas.")

    def generate_visualization(self, filepath="graphs/cluster_heads.png"):
        """
        Generates a visual graph of WSN node deployment showing Cluster Heads.
        Adds larger CH markers, sequential labels (CH1, CH2...), and communication circles.
        
        Args:
            filepath (str): Path to save the visualization image.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.figure(figsize=(10, 8))
        
        # Filter nodes into normal and cluster heads (only considering alive nodes)
        normal_nodes = [n for n in self.nodes if n.is_alive() and not n.is_ch]
        cluster_heads = [n for n in self.nodes if n.is_alive() and n.is_ch]
        
        # Plot normal nodes (blue circle)
        if normal_nodes:
            plt.scatter(
                [n.x for n in normal_nodes], 
                [n.y for n in normal_nodes], 
                c='blue', 
                marker='o', 
                s=45, 
                label='Normal Sensor Nodes', 
                alpha=0.7, 
                edgecolors='darkblue'
            )
            
        # Plot Cluster Heads (orange circle) with increased size (s=150)
        if cluster_heads:
            plt.scatter(
                [n.x for n in cluster_heads], 
                [n.y for n in cluster_heads], 
                c='orange', 
                marker='o', 
                s=150, 
                label='Cluster Heads (CH)', 
                edgecolors='darkorange', 
                linewidths=2.5,
                zorder=5
            )
            
        # Plot Base Station (green triangle) with increased size (s=180)
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
        
        # Label every Cluster Head as CH1, CH2, CH3... and draw communication boundary
        for idx, ch in enumerate(cluster_heads):
            # Text label
            plt.text(
                ch.x + 1.8, 
                ch.y + 1.8, 
                f"CH{idx + 1}", 
                fontsize=9, 
                fontweight='bold', 
                color='darkorange',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                zorder=10
            )
            # Communication boundary circle (radius = TRANSMISSION_RANGE)
            circle = plt.Circle(
                (ch.x, ch.y), 
                config.TRANSMISSION_RANGE, 
                color='orange', 
                fill=False, 
                linestyle='--', 
                alpha=0.2,
                linewidth=1.2
            )
            plt.gca().add_patch(circle)

        # Draw boundary of sensing field (0,0) to (100,100)
        plt.plot(
            [0, config.NETWORK_WIDTH, config.NETWORK_WIDTH, 0, 0], 
            [0, 0, config.NETWORK_HEIGHT, config.NETWORK_HEIGHT, 0], 
            color='grey', 
            linestyle='--', 
            linewidth=1.5, 
            label='Sensing Boundary'
        )
        
        # Limits and labels
        plt.xlim(-5, config.NETWORK_WIDTH + 5)
        plt.ylim(-5, max(config.NETWORK_HEIGHT, self.base_station.y) + 15)
        plt.xlabel('X Coordinate (meters)', fontsize=12)
        plt.ylabel('Y Coordinate (meters)', fontsize=12)
        plt.title('WSN Cluster Head Selection Framework (Normalized & Weighted)', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right', fontsize=10)
        
        # Label Base Station
        plt.text(
            self.base_station.x + 2, 
            self.base_station.y - 2, 
            f"BS ({self.base_station.x}, {self.base_station.y})", 
            fontsize=10, 
            color='green', 
            fontweight='bold'
        )
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[CLUSTERING] Improved visualization graph saved to {filepath}.")

    def print_statistics(self):
        """
        Calculates and prints statistics for the selected Cluster Heads and WSN network.
        """
        candidates = self.initialize_candidate_nodes()
        selected_chs = [n for n in candidates if n.is_ch]
        
        if not candidates:
            print("[CLUSTERING] No alive candidate nodes to display statistics.")
            return

        total_nodes = len(self.nodes)
        chs_selected = len(selected_chs)
        avg_fitness = np.mean([n.fitness for n in candidates])
        max_fitness = max([n.fitness for n in candidates])
        min_fitness = min([n.fitness for n in candidates])
        avg_degree = np.mean([n.node_degree for n in candidates])
        avg_dist = np.mean([n.distance_to_bs for n in candidates])
        avg_energy = np.mean([n.energy for n in candidates])

        print("\n" + "="*50)
        print("          WSN CLUSTER HEAD SELECTION STATISTICS          ")
        print("="*50)
        print(f"Total Nodes                    : {total_nodes}")
        print(f"Cluster Heads Selected        : {chs_selected} ({chs_selected / total_nodes * 100:.1f}%)")
        print(f"Average Fitness                : {avg_fitness:.4f}")
        print(f"Maximum Fitness                : {max_fitness:.4f}")
        print(f"Minimum Fitness                : {min_fitness:.4f}")
        print(f"Average Node Degree            : {avg_degree:.2f}")
        print(f"Average Distance to BS (meters): {avg_dist:.2f}")
        print(f"Average Residual Energy (J)    : {avg_energy:.4f}")
        print("="*50 + "\n")
