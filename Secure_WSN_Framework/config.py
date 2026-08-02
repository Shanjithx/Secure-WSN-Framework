# config.py
"""
Configuration module for the SECURE_WSN_FRAMEWORK.
Contains all global configuration parameters, constants, and network simulation settings.
"""

# Random Seed for reproducibility
RANDOM_SEED = 42

# Network Dimensions
NETWORK_WIDTH = 100
NETWORK_HEIGHT = 100

# Keep old naming for compatibility with other files
X_MAX = NETWORK_WIDTH
Y_MAX = NETWORK_HEIGHT

# Sensor Node Configuration
NUM_NODES = 100
INITIAL_ENERGY = 2.0  # Initial energy of each node in Joules

# Base Station Configuration
BASE_STATION = (50, 120)

# Keep old naming for compatibility
BS_X = BASE_STATION[0]
BS_Y = BASE_STATION[1]
BS_COORDINATES = BASE_STATION

# Wireless Communication
TRANSMISSION_RANGE = 25

# Radio Energy Model
PACKET_SIZE = 4000
E_ELEC = 50e-9
E_FS = 10e-12
E_MP = 0.0013e-12

# File Paths for saving plots and results
GRAPH_DIR = "graphs"
RESULT_DIR = "results"
DEPLOYMENT_PLOT_PATH = f"{GRAPH_DIR}/deployment.png"
NODE_INFO_CSV_PATH = f"{RESULT_DIR}/node_information.csv"

# Cluster Head Selection
CH_PERCENTAGE = 0.10
# Fitness Weights
W_ENERGY = 0.35
W_DISTANCE = 0.25
W_DEGREE = 0.20
W_TRUST = 0.20

# Trust Computation Configuration
TRUST_THRESHOLD = 0.50
W_DIRECT_TRUST = 0.30
W_ENERGY_TRUST = 0.30
W_COMMUNICATION_TRUST = 0.20
W_BEHAVIORAL_TRUST = 0.20

# Seagull Optimization Algorithm (SOA) Routing
SOA_POPULATION = 30
SOA_ITERATIONS = 100
MAX_HOP_DISTANCE = 30
TRUST_WEIGHT = 0.40
ENERGY_WEIGHT = 0.30
DISTANCE_WEIGHT = 0.20
HOP_WEIGHT = 0.10

# Dataset Preprocessing Configuration
DATASET_PATH = "datasets/cicids2017.csv"
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_STATE = 42
SCALER = "StandardScaler"
LABEL_COLUMN = "Label"

# PRCNN Model Configuration
PRCNN_EPOCHS = 25
PRCNN_BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.30
CNN_FILTERS = 64
CNN_KERNEL_SIZE = 3
POOL_SIZE = 2
LSTM_UNITS = 128
DENSE_UNITS = 64
EARLY_STOPPING_PATIENCE = 5
MODEL_SAVE_PATH = "models/prcnn_model.keras"

# Performance Evaluation Configuration
SIMULATION_TIME = 100
PACKETS_SENT = 5000
PACKETS_RECEIVED = 4850
ROUTING_PACKETS = 150
DATA_PACKETS = 4850





