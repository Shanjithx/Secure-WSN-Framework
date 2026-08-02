# main.py
"""
Main entry point for the SECURE_WSN_FRAMEWORK simulation.
Coordinates deployment, clustering, routing, security, and analysis modules.
"""

from deployment import deploy_network
from cluster_head_selection import ClusterHeadSelector
from cluster_formation import ClusterFormation
from trust_computation import TrustComputation
from soa_routing import SOARouting
from prcnn_ids import DatasetPreprocessor, PRCNN_IDS
from performance_metrics import PerformanceEvaluator






def main():
    import time
    pipeline_start_time = time.time()
    print("==========================================================")
    print("      SECURE WIRELESS SENSOR NETWORK (WSN) FRAMEWORK      ")
    print("==========================================================")
    
    # Module 1: Node Deployment
    print("\n--- Starting Module 1: Node Deployment ---")
    nodes, base_station = deploy_network()
    
    print("\n==========================================================")
    print("Module 1: Node Deployment Completed Successfully!")
    print(f"Total deployed nodes: {len(nodes)}")
    print(f"Base Station location: ({base_station.x}, {base_station.y})")
    print("==========================================================")
    
    # Module 2: Cluster Head Selection Framework
    print("\n--- Starting Module 2: Cluster Head Selection ---")
    selector = ClusterHeadSelector(nodes, base_station)
    
    # Pre-optimization fitness calculation and normalization
    selector.normalize_metrics()
    selector.calculate_fitness()
    selector.rank_nodes()
    
    # Ip-GJOA Optimization Selection
    import time
    start_time = time.time()
    print("Ip-GJOA Optimization Started")
    
    selected_chs = selector.select_cluster_heads_ip_gjoa(pop_size=30, max_iter=100)
    exec_time = time.time() - start_time
    
    best_fitness = selector.evaluate_solution([node.node_id for node in selected_chs])
    
    print(f"Best Fitness: {best_fitness:.4f}")
    print(f"Number of Iterations: 100")
    print(f"Selected Cluster Heads: {[node.node_id for node in selected_chs]}")
    print(f"Execution Time: {exec_time:.4f} seconds")
    print("Ip-GJOA Completed Successfully")
    
    # Generate CSV and Plot outputs
    selector.generate_csv_output()
    selector.generate_visualization()
    
    # Print statistics
    selector.print_statistics()
    
    print("\n==========================================================")
    print("Module 2: Cluster Head Selection Completed Successfully!")
    print(f"Number of selected Cluster Heads: {len(selected_chs)}")
    print("==========================================================")
    
    # Module 3: Cluster Formation
    print("\n--- Starting Module 3: Cluster Formation ---")
    print("Cluster Formation Started")
    
    formation = ClusterFormation(nodes, base_station)
    cluster_dict = formation.form_clusters()
    
    # Calculate statistics for main prints
    total_clusters = len(cluster_dict)
    total_members = sum(len(members) for members in cluster_dict.values())
    avg_size = total_members / total_clusters if total_clusters > 0 else 0
    
    print(f"Total Clusters Created: {total_clusters}")
    print(f"Average Cluster Size: {avg_size:.2f}")
    print("Cluster Formation Completed Successfully")

    # Module 4: Trust Computation
    print("\n--- Starting Module 4: Trust Computation ---")
    try:
        print("Trust Computation Started")
        trust_computer = TrustComputation(nodes, base_station)
        
        # Execute step-by-step computation to output status milestones
        trust_computer.initialize_trust()
        trust_computer.calculate_direct_trust()
        trust_computer.calculate_energy_trust()
        trust_computer.calculate_communication_trust()
        trust_computer.calculate_behavioral_trust()
        trust_computer.calculate_overall_trust()
        print("Trust Values Computed")
        
        trust_computer.identify_low_trust_nodes()
        print("Suspicious Nodes Detected")
        
        # Generate CSV reports, visualizations, and summary stats
        trust_computer.generate_csv_output()
        trust_computer.generate_visualization()
        trust_computer.print_statistics()
        
        print("Trust Computation Completed Successfully")
        
    except Exception as e:
        print("[ERROR] Module Name: Module 4 - Trust Computation")
        print(f"[ERROR] Reason: {str(e)}")
        print("[ERROR] Exiting gracefully.")

    # Module 5: Secure Routing using SOA
    print("\n--- Starting Module 5: Secure Routing (SOA) ---")
    try:
        print("SOA Routing Started")
        
        # Instantiate SOARouting
        soa_routing = SOARouting(nodes, base_station)
        
        # Execute routing orchestration
        soa_routing.identify_cluster_heads()
        for ch in soa_routing.cluster_heads:
            candidate_paths = soa_routing.generate_candidate_paths(ch)
            best_route = soa_routing.find_best_route(ch, candidate_paths)
            soa_routing.selected_routes[ch.node_id] = best_route
            soa_routing.update_node_energy(best_route)
            
        print("Routes Generated")
        print("SOA Optimization Completed")
        
        # Generate files & plots
        soa_routing.generate_csv_output()
        soa_routing.generate_visualization()
        
        print("Routing Statistics Generated")
        # Print stats
        soa_routing.print_statistics()
        
        print("SOA Routing Completed Successfully")
        
    except Exception as e:
        print("[ERROR] Module Name: Module 5 - Secure Routing using SOA")
        print(f"[ERROR] Reason: {str(e)}")
        print("[ERROR] Exiting gracefully.")

    # Module 6: CICIDS2017 Dataset Preprocessing
    print("\n--- Starting Module 6: CICIDS2017 Preprocessing ---")
    try:
        # Instantiate preprocessor
        preprocessor = DatasetPreprocessor()
        
        # Execute preprocessing orchestration with status milestones
        print("Dataset Loading Started")
        preprocessor.load_dataset()
        
        preprocessor.clean_dataset()
        print("Dataset Cleaned")
        
        preprocessor.encode_labels()
        
        preprocessor.scale_features()
        print("Features Scaled")
        
        preprocessor.split_dataset()
        print("Dataset Split Completed")
        
        preprocessor.save_processed_data()
        print("Preprocessed Dataset Saved")
        
        # Diagnostics & plotting
        preprocessor.generate_visualizations()
        preprocessor.print_statistics()
        
        print("Module 6 Completed Successfully")
        
    except Exception as e:
        print("[ERROR] Module Name: Module 6 - CICIDS2017 Dataset Preprocessing")
        print(f"[ERROR] Reason: {str(e)}")
        print("[ERROR] Exiting gracefully.")

    # Module 7: Parallel Recurrent Convolutional Neural Network (PRCNN) IDS
    print("\n--- Starting Module 7: PRCNN Intrusion Detection System ---")
    try:
        print("PRCNN IDS Started")
        
        # Instantiate model manager
        ids_model = PRCNN_IDS()
        
        # Execute training & evaluation flow with milestones
        ids_model.load_dataset()
        print("Dataset Loaded")
        
        ids_model.build_model()
        print("Model Created")
        
        ids_model.compile_model()
        
        print("Training Started")
        ids_model.train_model()
        print("Training Completed")
        
        ids_model.evaluate_model()
        print("Evaluation Completed")
        
        ids_model.save_model()
        print("Model Saved")
        
        ids_model.generate_graphs()
        print("Graphs Generated")
        
        ids_model.generate_reports()
        ids_model.print_statistics()
        
        print("Module 7 Completed Successfully")
        
    except Exception as e:
        print("[ERROR] Module Name: Module 7 - PRCNN Intrusion Detection System")
        print(f"[ERROR] Reason: {str(e)}")
        print("[ERROR] Exiting gracefully.")

    # Module 8: Performance Evaluation
    print("\n--- Starting Module 8: Performance Evaluation ---")
    try:
        print("Performance Evaluation Started")
        
        # Instantiate PerformanceEvaluator
        evaluator = PerformanceEvaluator(
            nodes=nodes,
            base_station=base_station,
            selected_routes=soa_routing.selected_routes,
            ids_model=ids_model
        )
        
        # Execute evaluation steps and print milestones
        evaluator.calculate_energy_metrics()
        evaluator.calculate_alive_dead_nodes()
        evaluator.calculate_network_lifetime()
        print("Network Metrics Calculated")
        
        evaluator.calculate_packet_delivery_ratio()
        evaluator.calculate_average_delay()
        evaluator.calculate_throughput()
        evaluator.calculate_routing_overhead()
        print("Routing Metrics Calculated")
        
        evaluator.calculate_trust_metrics()
        print("Trust Metrics Calculated")
        
        evaluator.calculate_ids_metrics()
        print("IDS Metrics Calculated")
        
        evaluator.generate_csv_reports()
        print("CSV Reports Generated")
        
        evaluator.generate_visualizations()
        print("Graphs Generated")
        
        evaluator.generate_final_summary()
        print("Final Summary Generated")
        
        # Print summary stats
        evaluator.print_statistics()
        
        # Set execution time and finalize
        evaluator.evaluate(pipeline_start_time)
        
        print("Performance Evaluation Completed Successfully")
        
    except Exception as e:
        print("[ERROR] Module Name: Module 8 - Performance Evaluation")
        print(f"[ERROR] Reason: {str(e)}")
        print("[ERROR] Exiting gracefully.")

if __name__ == "__main__":
    main()






