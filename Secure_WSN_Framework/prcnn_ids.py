# prcnn_ids.py
"""
Intrusion Detection Module for SECURE_WSN_FRAMEWORK.
Includes DatasetPreprocessor (Module 6) and PRCNN_IDS (Module 7).
PRCNN_IDS trains a Parallel Recurrent Convolutional Neural Network on 
preprocessed CICIDS2017 flow data to classify network traffic.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config

class DatasetPreprocessor:
    """
    Handles preprocessing pipeline for CICIDS2017 network intrusion dataset.
    Performs data cleaning, label encoding, scaling, and stratified splitting.
    """
    def __init__(self):
        self.df = None
        self.df_clean = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.le = None
        self.scaler = None
        
        # Diagnostics
        self.original_rows = 0
        self.original_cols = 0
        self.final_rows = 0
        self.final_cols = 0
        self.missing_removed = 0
        self.duplicates_removed = 0

    def _generate_mock_dataset(self):
        """
        Generates a mock CICIDS2017 CSV file with realistic columns and classes
        to ensure the preprocessing pipeline runs successfully if the real dataset is absent.
        """
        os.makedirs(os.path.dirname(config.DATASET_PATH), exist_ok=True)
        np.random.seed(config.RANDOM_STATE)
        
        num_samples = 1000
        # Common CICIDS2017 columns
        columns = [
            " Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets",
            "Total Length of Fwd Packets", " Total Length of Bwd Packets", " Fwd Packet Length Max",
            " Fwd Packet Length Min", " Fwd Packet Length Mean", " Fwd Packet Length Std",
            "Bwd Packet Length Max", " Bwd Packet Length Min", " Bwd Packet Length Mean",
            " Bwd Packet Length Std", "Flow Bytes/s", " Flow Packets/s", " Flow IAT Mean",
            " Flow IAT Std", " Flow IAT Max", " Flow IAT Min", "Fwd IAT Total",
            " Fwd IAT Mean", " Fwd IAT Std", " Fwd IAT Max", " Fwd IAT Min",
            "Bwd IAT Total", " Bwd IAT Mean", " Bwd IAT Std", " Bwd IAT Max",
            " Bwd IAT Min", "Fwd PSH Flags", " Bwd PSH Flags", " Fwd URG Flags",
            " Bwd URG Flags", " Fwd Header Length", " Bwd Header Length", "Fwd Packets/s",
            " Bwd Packets/s", " Min Packet Length", " Max Packet Length", " Packet Length Mean",
            " Packet Length Std", " Packet Length Variance", "FIN Flag Count", " SYN Flag Count",
            " RST Flag Count", " PSH Flag Count", " ACK Flag Count", " URG Flag Count",
            " CWE Flag Count", " ECE Flag Count", " Down/Up Ratio", " Average Packet Size",
            " Avg Fwd Segment Size", " Avg Bwd Segment Size", " Fwd Header Length.1",
            "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", " Fwd Avg Write Status/Bulk",
            " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", "Bwd Avg Write Status/Bulk",
            "Subflow Fwd Packets", " Subflow Fwd Bytes", " Subflow Bwd Packets", " Subflow Bwd Bytes",
            "Init_Win_bytes_forward", " Init_Win_bytes_backward", " act_data_pkt_fwd",
            " min_seg_size_forward", "Active Mean", " Active Std", " Active Max", " Active Min",
            "Idle Mean", " Idle Std", " Idle Max", " Idle Min", "Label"
        ]
        
        data = {}
        for col in columns[:-1]:
            # Generate mock values
            if col in [" Destination Port", "FIN Flag Count", " SYN Flag Count", " RST Flag Count", 
                       " PSH Flag Count", " ACK Flag Count", " URG Flag Count"]:
                data[col] = np.random.randint(0, 100, num_samples)
            elif col in ["Fwd PSH Flags", " Bwd PSH Flags", " Fwd URG Flags", " Bwd URG Flags"]:
                data[col] = np.random.randint(0, 2, num_samples)
            else:
                data[col] = np.random.uniform(0.0, 1000.0, num_samples)
                
        # Insert duplicate constant columns to test constant feature removal
        data[" Constant_Feature_1"] = np.ones(num_samples)
        data[" Constant_Feature_2"] = np.zeros(num_samples)
        
        # Insert NaNs and Infs to test data cleaning
        data["Flow Bytes/s"][10] = np.nan
        data["Flow Bytes/s"][20] = np.inf
        data[" Flow Packets/s"][30] = -np.inf
        
        # Categorical labels typical in CICIDS2017
        labels = ["BENIGN"] * 600 + ["DoS"] * 150 + ["DDoS"] * 120 + ["PortScan"] * 80 + ["Bot"] * 50
        data["Label"] = labels
        
        df = pd.DataFrame(data)
        # Append explicit duplicates to test deduplication
        df = pd.concat([df, df.iloc[[5, 15, 25]]], ignore_index=True)
        
        df.to_csv(config.DATASET_PATH, index=False)
        print(f"[PREPROCESSING] Created mock dataset at {config.DATASET_PATH}.")

    def load_dataset(self):
        """
        Loads the raw dataset from config.DATASET_PATH.
        If file does not exist, generates a mock dataset to allow execution.
        """
        try:
            if not os.path.exists(config.DATASET_PATH):
                print(f"[PREPROCESSING WARNING] Dataset file not found at {config.DATASET_PATH}.")
                self._generate_mock_dataset()
                
            self.df = pd.read_csv(config.DATASET_PATH)
            self.original_rows, self.original_cols = self.df.shape
            
            # Print basic info
            print(f"[PREPROCESSING] Dataset Loaded: {self.original_rows} rows, {self.original_cols} columns.")
            print("[PREPROCESSING] Columns list: ", list(self.df.columns[:5]), "... [truncated]")
            print("[PREPROCESSING] Class Distribution in raw data:")
            print(self.df[config.LABEL_COLUMN].value_counts())
            return self.df
        except Exception as e:
            raise RuntimeError(f"Error in load_dataset: {str(e)}")

    def clean_dataset(self):
        """
        Cleans the loaded dataset by removing duplicates, handling missing/infinite values,
        converting non-numeric values, and removing constant features.
        """
        try:
            df_cleaned = self.df.copy()
            
            # 1. Remove duplicate rows
            df_cleaned = df_cleaned.drop_duplicates()
            self.duplicates_removed = self.original_rows - df_cleaned.shape[0]
            
            # 2. Convert non-numeric values (excluding Label) to numeric where required
            for col in df_cleaned.columns:
                if col != config.LABEL_COLUMN and df_cleaned[col].dtype == 'object':
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    
            # 3. Handle infinite values (convert to NaN)
            df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)
            
            # 4. Handle missing values (drop rows containing any NaNs)
            missing_count = df_cleaned.isna().sum().sum()
            df_cleaned = df_cleaned.dropna()
            self.missing_removed = missing_count
            
            # 5. Remove empty columns (all NaN)
            df_cleaned = df_cleaned.dropna(how='all', axis=1)
            
            # 6. Remove constant features (std == 0)
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            constant_cols = [col for col in numeric_cols if df_cleaned[col].std() == 0]
            if constant_cols:
                df_cleaned = df_cleaned.drop(columns=constant_cols)
                print(f"[PREPROCESSING] Removed constant columns: {constant_cols}")
                
            self.df_clean = df_cleaned
            self.final_rows, self.final_cols = self.df_clean.shape
            
            return self.df_clean
        except Exception as e:
            raise RuntimeError(f"Error in clean_dataset: {str(e)}")

    def encode_labels(self):
        """
        Encodes attack labels into numerical classes and saves the encoder.
        """
        try:
            from sklearn.preprocessing import LabelEncoder
            
            os.makedirs("models", exist_ok=True)
            self.le = LabelEncoder()
            
            # Strip whitespace from labels to avoid duplicates like 'DoS ' and 'DoS'
            labels_stripped = self.df_clean[config.LABEL_COLUMN].astype(str).str.strip()
            self.df_clean[config.LABEL_COLUMN] = self.le.fit_transform(labels_stripped)
            
            # Save fitted label encoder
            encoder_path = "models/label_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.le, f)
                
            print(f"[PREPROCESSING] Encoded classes: {list(self.le.classes_)}")
            return self.df_clean
        except Exception as e:
            raise RuntimeError(f"Error in encode_labels: {str(e)}")

    def scale_features(self):
        """
        Separates features (X) and labels (y), drops unnecessary columns, and scales numeric features.
        """
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Separate features and labels
            y = self.df_clean[config.LABEL_COLUMN]
            X = self.df_clean.drop(columns=[config.LABEL_COLUMN])
            
            # Identify columns to drop (metadata like Flow ID, timestamps, IP addresses)
            cols_to_drop = []
            for col in X.columns:
                col_lower = col.lower().strip()
                if col_lower in ["flow id", "source ip", "source port", "destination ip", "timestamp", "date"]:
                    cols_to_drop.append(col)
                    
            if cols_to_drop:
                X = X.drop(columns=cols_to_drop)
                print(f"[PREPROCESSING] Dropped metadata columns: {cols_to_drop}")
                
            # Fit and save StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Recreate DataFrame
            self.X = pd.DataFrame(X_scaled, columns=X.columns)
            self.y = y.reset_index(drop=True)
            
            scaler_path = "models/scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            return self.X, self.y
        except Exception as e:
            raise RuntimeError(f"Error in scale_features: {str(e)}")

    def split_dataset(self):
        """
        Splits dataset into stratified training, validation, and test subsets based on configuration weights.
        """
        try:
            from sklearn.model_selection import train_test_split
            
            # First split: Train vs Temp (Val + Test)
            # Temp size is VALIDATION + TEST = 0.30
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.X, self.y,
                test_size=(config.VALIDATION_SPLIT + config.TEST_SPLIT),
                random_state=config.RANDOM_STATE,
                stratify=self.y
            )
            
            # Second split: Split Temp equally into Val and Test (each receives 15% of original dataset)
            val_proportion = config.VALIDATION_SPLIT / (config.VALIDATION_SPLIT + config.TEST_SPLIT)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=(1.0 - val_proportion),
                random_state=config.RANDOM_STATE,
                stratify=y_temp
            )
            
            self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
            self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
            
            return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        except Exception as e:
            raise RuntimeError(f"Error in split_dataset: {str(e)}")

    def save_processed_data(self):
        """
        Generates final preprocessed CSV files for training and preprocessing_summary.csv.
        """
        try:
            os.makedirs("datasets", exist_ok=True)
            os.makedirs("results", exist_ok=True)
            
            # Save feature and label splits
            self.X_train.to_csv("datasets/X_train.csv", index=False)
            self.X_val.to_csv("datasets/X_validation.csv", index=False)
            self.X_test.to_csv("datasets/X_test.csv", index=False)
            self.y_train.to_csv("datasets/y_train.csv", index=False)
            self.y_val.to_csv("datasets/y_validation.csv", index=False)
            self.y_test.to_csv("datasets/y_test.csv", index=False)
            
            # Create preprocessing_summary.csv
            summary_data = {
                "Original Samples": [self.original_rows],
                "Final Samples": [self.final_rows],
                "Original Features": [self.original_cols],
                "Final Features": [self.final_cols],
                "Missing Values Removed": [self.missing_removed],
                "Duplicate Rows Removed": [self.duplicates_removed],
                "Scaling Method": [config.SCALER]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv("results/preprocessing_summary.csv", index=False)
            print("[PREPROCESSING] Preprocessed data saved successfully.")
        except Exception as e:
            raise RuntimeError(f"Error in save_processed_data: {str(e)}")

    def generate_visualizations(self):
        """
        Generates class_distribution.png, feature_correlation.png, and feature_distribution.png.
        """
        try:
            os.makedirs("graphs", exist_ok=True)
            
            # 1. Class Distribution
            plt.figure(figsize=(8, 6))
            classes = self.le.classes_
            class_counts = pd.Series(self.y_train).value_counts().sort_index()
            class_names = [classes[i] for i in class_counts.index]
            
            plt.bar(class_names, class_counts.values, color='skyblue', edgecolor='black')
            plt.xlabel('Attack Classes', fontsize=12)
            plt.ylabel('Number of Samples (Training Set)', fontsize=12)
            plt.title('CICIDS2017 Preprocessed Class Distribution', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.xticks(rotation=45)
            plt.savefig("graphs/class_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Feature Correlation Heatmap (using subset to fit reasonably in plot)
            plt.figure(figsize=(10, 8))
            top_features = self.X.columns[:10]  # First 10 features for visibility
            corr_subset = self.X[top_features].corr()
            
            try:
                import seaborn as sns
                sns.heatmap(corr_subset, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            except ImportError:
                # Manual heatmap using Matplotlib if Seaborn is missing
                im = plt.imshow(corr_subset, cmap='coolwarm')
                plt.colorbar(im)
                plt.xticks(np.arange(len(corr_subset.columns)), corr_subset.columns, rotation=45, ha="right")
                plt.yticks(np.arange(len(corr_subset.columns)), corr_subset.columns)
                for i in range(len(corr_subset.columns)):
                    for j in range(len(corr_subset.columns)):
                        plt.text(j, i, f"{corr_subset.iloc[i, j]:.2f}", ha="center", va="center", color="black")
                        
            plt.title('Feature Correlation Heatmap (Subset)', fontsize=14, fontweight='bold')
            plt.savefig("graphs/feature_correlation.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Feature Distribution of Important Features
            plt.figure(figsize=(10, 6))
            plot_features = []
            for col in self.X.columns:
                if "duration" in col.lower() or "mean" in col.lower() or "packets" in col.lower():
                    plot_features.append(col)
                if len(plot_features) == 2:
                    break
            if not plot_features:
                plot_features = list(self.X.columns[:2])
                
            try:
                import seaborn as sns
                for col in plot_features:
                    sns.kdeplot(self.X[col], label=col.strip(), fill=True, alpha=0.4)
            except ImportError:
                for col in plot_features:
                    plt.hist(self.X[col], bins=30, alpha=0.5, label=col.strip(), density=True)
                    
            plt.xlabel('Scaled Value', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.title('Distribution of Important Numerical Features', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend()
            plt.savefig("graphs/feature_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("[PREPROCESSING] Preprocessing plots generated.")
        except Exception as e:
            raise RuntimeError(f"Error in generate_visualizations: {str(e)}")

    def print_statistics(self):
        """
        Prints data shape and split information to the console.
        """
        try:
            print("\n" + "="*50)
            print("          WSN DATASET PREPROCESSING SUMMARY        ")
            print("="*50)
            print(f"Original Dataset Size      : {self.original_rows} samples")
            print(f"Final Dataset Size         : {self.final_rows} samples")
            print(f"Features                   : {self.final_cols - 1}")
            print(f"Training Samples           : {len(self.X_train)}")
            print(f"Validation Samples         : {len(self.X_val)}")
            print(f"Testing Samples            : {len(self.X_test)}")
            print(f"Number of Classes          : {len(self.le.classes_)}")
            print(f"Scaling Method             : {config.SCALER}")
            print("="*50 + "\n")
        except Exception as e:
            raise RuntimeError(f"Error in print_statistics: {str(e)}")

    def preprocess(self):
        """
        Orchestration method to run the entire preprocessing workflow.
        """
        self.load_dataset()
        self.clean_dataset()
        self.encode_labels()
        self.scale_features()
        self.split_dataset()
        self.save_processed_data()
        self.generate_visualizations()
        self.print_statistics()


class PRCNN_IDS:
    """
    Parallel Recurrent Convolutional Neural Network (PRCNN) Intrusion Detection System.
    Loads processed dataset splits, compiles the functional Keras network,
    trains it with optimization callbacks, evaluates results on a test split,
    saves the models, and visualizes accuracy, loss, confusion matrix, ROC, and PR curves.
    """
    def __init__(self):
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None
        self.history = None
        
        # Dimensions
        self.num_samples = 0
        self.num_features = 0
        self.num_classes = 0
        self.class_names = []
        
        # Metrics
        self.train_accuracy = 0.0
        self.val_accuracy = 0.0
        self.test_accuracy = 0.0
        self.test_precision = 0.0
        self.test_recall = 0.0
        self.test_f1 = 0.0
        self.test_specificity = 0.0
        self.test_roc_auc = 0.0
        
        # Times
        self.training_time = 0.0
        self.inference_time = 0.0
        
        # Reports
        self.confusion_matrix = None
        self.class_report_str = ""
        self.class_report_dict = {}

    def load_dataset(self):
        """
        Loads the preprocessed splits from the datasets/ directory.
        """
        try:
            self.X_train = pd.read_csv("datasets/X_train.csv")
            self.X_val = pd.read_csv("datasets/X_validation.csv")
            self.X_test = pd.read_csv("datasets/X_test.csv")
            self.y_train = pd.read_csv("datasets/y_train.csv").values.ravel()
            self.y_val = pd.read_csv("datasets/y_validation.csv").values.ravel()
            self.y_test = pd.read_csv("datasets/y_test.csv").values.ravel()
            
            self.num_samples = len(self.X_train) + len(self.X_val) + len(self.X_test)
            self.num_features = self.X_train.shape[1]
            
            # Retrieve classes list by loading encoder
            encoder_path = "models/label_encoder.pkl"
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    le = pickle.load(f)
                    self.class_names = list(le.classes_)
                    self.num_classes = len(self.class_names)
            else:
                unique_labels = np.unique(np.concatenate([self.y_train, self.y_val, self.y_test]))
                self.class_names = [f"Class {int(c)}" for c in unique_labels]
                self.num_classes = len(self.class_names)
                
            print(f"[DATASET] Loaded preprocessed splits successfully.")
            print(f"[DATASET] Total Samples : {self.num_samples}")
            print(f"[DATASET] Features      : {self.num_features}")
            print(f"[DATASET] Classes Count : {self.num_classes} ({self.class_names})")
        except Exception as e:
            raise RuntimeError(f"Error in load_dataset: {str(e)}")

    def build_model(self):
        """
        Builds the PRCNN model using the Keras Functional API.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Reshape, Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, LSTM, Dense
            
            inputs = Input(shape=(self.num_features,))
            
            # Reshape tabular data to 3D tensor: (batch_size, num_features, 1) for Conv1D
            x = Reshape((self.num_features, 1))(inputs)
            
            # Conv1D Layer: extracts local adjacent spatial features
            x = Conv1D(filters=config.CNN_FILTERS, kernel_size=config.CNN_KERNEL_SIZE, padding='same')(x)
            
            # Batch Normalization: stabilizes gradient flows and accelerates training
            x = BatchNormalization()(x)
            
            # ReLU Activation: introduces non-linearity to the network
            x = Activation('relu')(x)
            
            # MaxPooling1D: performs downsampling on feature mapping
            x = MaxPooling1D(pool_size=config.POOL_SIZE, padding='same')(x)
            
            # Dropout: prevents co-adaptation and reduces overfitting
            x = Dropout(config.DROPOUT_RATE)(x)
            
            # LSTM Layer: handles sequential dependencies across features
            x = LSTM(config.LSTM_UNITS)(x)
            
            # Dropout
            x = Dropout(config.DROPOUT_RATE)(x)
            
            # Dense Layer: aggregates feature mappings
            x = Dense(config.DENSE_UNITS, activation='relu')(x)
            
            # Dense Output Layer with Softmax for probability distribution
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            self.model = Model(inputs=inputs, outputs=outputs, name="PRCNN_IDS")
            print("[MODEL] Architecture built successfully.")
            self.model.summary()
            return self.model
        except Exception as e:
            raise RuntimeError(f"Error in build_model: {str(e)}")

    def compile_model(self):
        """
        Compiles the model using Adam optimizer and SparseCategoricalCrossentropy loss.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.losses import SparseCategoricalCrossentropy
            
            optimizer = Adam(learning_rate=config.LEARNING_RATE)
            loss_fn = SparseCategoricalCrossentropy()
            
            self.model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy']
            )
            print("[MODEL] Model compiled successfully.")
        except Exception as e:
            raise RuntimeError(f"Error in compile_model: {str(e)}")

    def train_model(self):
        """
        Trains the compiled PRCNN model using optimizer callbacks and records training duration.
        """
        try:
            import time
            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
            
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            checkpoint = ModelCheckpoint(
                filepath=config.MODEL_SAVE_PATH,
                monitor='val_loss',
                save_best_only=True
            )
            
            lr_reduce = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
            
            print("[TRAINING] Starting neural network training...")
            start_time = time.time()
            
            self.history = self.model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=config.PRCNN_EPOCHS,
                batch_size=config.PRCNN_BATCH_SIZE,
                callbacks=[early_stop, checkpoint, lr_reduce],
                verbose=1
            )
            
            self.training_time = time.time() - start_time
            print(f"[TRAINING] Model training completed in {self.training_time:.4f} seconds.")
            
            # Store final training accuracies
            self.train_accuracy = self.history.history['accuracy'][-1]
            self.val_accuracy = self.history.history['val_accuracy'][-1]
            return self.history
        except Exception as e:
            raise RuntimeError(f"Error in train_model: {str(e)}")

    def evaluate_model(self):
        """
        Evaluates the trained model on the test split and calculates classification metrics.
        """
        try:
            import time
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
            
            print("[EVALUATION] Testing model on unseen test split...")
            start_time = time.time()
            
            # Predict probabilities
            y_pred_probs = self.model.predict(self.X_test)
            self.inference_time = time.time() - start_time
            
            # Convert probabilities to class labels
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Basic metrics
            self.test_accuracy = accuracy_score(self.y_test, y_pred)
            self.test_precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            self.test_recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            self.test_f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
            
            # Specificity calculation per-class
            cm = self.confusion_matrix
            specificities = []
            for i in range(self.num_classes):
                tp = cm[i, i]
                fn = sum(cm[i, :]) - tp
                fp = sum(cm[:, i]) - tp
                tn = sum(sum(cm)) - tp - fn - fp
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificities.append(spec)
            self.test_specificity = np.mean(specificities)
            
            # Classification reports
            self.class_report_str = classification_report(self.y_test, y_pred, target_names=self.class_names, zero_division=0)
            self.class_report_dict = classification_report(self.y_test, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)
            
            # ROC-AUC calculation if all target classes are represented in y_test
            try:
                if len(np.unique(self.y_test)) == self.num_classes:
                    self.test_roc_auc = roc_auc_score(self.y_test, y_pred_probs, multi_class='ovr', average='weighted')
                else:
                    self.test_roc_auc = 0.0
            except Exception:
                self.test_roc_auc = 0.0
                
            print("[EVALUATION] Completed evaluating model.")
        except Exception as e:
            raise RuntimeError(f"Error in evaluate_model: {str(e)}")

    def save_model(self):
        """
        Saves the model to MODEL_SAVE_PATH and serializes the training history.
        """
        try:
            os.makedirs("models", exist_ok=True)
            self.model.save(config.MODEL_SAVE_PATH)
            
            history_path = "models/training_history.pkl"
            with open(history_path, 'wb') as f:
                pickle.dump(self.history.history, f)
                
            print(f"[MODEL] Model weights saved to {config.MODEL_SAVE_PATH}")
            print(f"[MODEL] Model history saved to {history_path}")
        except Exception as e:
            raise RuntimeError(f"Error in save_model: {str(e)}")

    def generate_graphs(self):
        """
        Generates accuracy, loss, confusion matrix, ROC-AUC, and Precision-Recall plots.
        """
        try:
            os.makedirs("graphs", exist_ok=True)
            from sklearn.metrics import roc_curve, auc, precision_recall_curve
            
            epochs_range = np.arange(1, len(self.history.history['accuracy']) + 1)
            
            # 1. Training Accuracy Plot
            plt.figure(figsize=(8, 6))
            plt.plot(epochs_range, self.history.history['accuracy'], 'b-', label='Training Accuracy')
            plt.plot(epochs_range, self.history.history['val_accuracy'], 'r--', label='Validation Accuracy')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title('PRCNN Training and Validation Accuracy', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='lower right')
            plt.savefig("graphs/training_accuracy.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Training Loss Plot
            plt.figure(figsize=(8, 6))
            plt.plot(epochs_range, self.history.history['loss'], 'b-', label='Training Loss')
            plt.plot(epochs_range, self.history.history['val_loss'], 'r--', label='Validation Loss')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title('PRCNN Training and Validation Loss', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='upper right')
            plt.savefig("graphs/training_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Confusion Matrix Plot
            plt.figure(figsize=(8, 6))
            cm = self.confusion_matrix
            try:
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                            xticklabels=self.class_names, yticklabels=self.class_names)
            except ImportError:
                im = plt.imshow(cm, cmap="Blues")
                plt.colorbar(im)
                plt.xticks(np.arange(len(self.class_names)), self.class_names, rotation=45)
                plt.yticks(np.arange(len(self.class_names)), self.class_names)
                for i in range(len(self.class_names)):
                    for j in range(len(self.class_names)):
                        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
            plt.savefig("graphs/confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Predict probabilities
            y_pred_probs = self.model.predict(self.X_test)
            
            # 4. ROC Curve (One-vs-Rest for multi-class)
            plt.figure(figsize=(8, 6))
            for i in range(self.num_classes):
                y_true_bin = (self.y_test == i).astype(int)
                y_prob = y_pred_probs[:, i]
                
                if sum(y_true_bin) > 0:
                    fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'Class {self.class_names[i]} (AUC = {roc_auc:.2f})')
                    
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(loc="lower right")
            plt.savefig("graphs/roc_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 5. Precision-Recall Curve
            plt.figure(figsize=(8, 6))
            for i in range(self.num_classes):
                y_true_bin = (self.y_test == i).astype(int)
                y_prob = y_pred_probs[:, i]
                
                if sum(y_true_bin) > 0:
                    precision, recall, _ = precision_recall_curve(y_true_bin, y_prob)
                    plt.plot(recall, precision, label=f'Class {self.class_names[i]}')
                    
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(loc="lower left")
            plt.savefig("graphs/precision_recall_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("[GRAPHS] Evaluation charts created.")
        except Exception as e:
            raise RuntimeError(f"Error in generate_graphs: {str(e)}")

    def generate_reports(self):
        """
        Writes aggregated evaluation metrics and detailed classification reports to CSV files.
        """
        try:
            os.makedirs("results", exist_ok=True)
            
            # 1. results/prcnn_metrics.csv
            metrics_data = {
                "Metric": [
                    "Training Accuracy",
                    "Validation Accuracy",
                    "Test Accuracy",
                    "Precision (Weighted)",
                    "Recall (Weighted)",
                    "F1 Score (Weighted)",
                    "Specificity",
                    "ROC-AUC (Weighted)",
                    "Training Time (s)",
                    "Inference Time (s)"
                ],
                "Value": [
                    round(self.train_accuracy, 6),
                    round(self.val_accuracy, 6),
                    round(self.test_accuracy, 6),
                    round(self.test_precision, 6),
                    round(self.test_recall, 6),
                    round(self.test_f1, 6),
                    round(self.test_specificity, 6),
                    round(self.test_roc_auc, 6),
                    round(self.training_time, 4),
                    round(self.inference_time, 4)
                ]
            }
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_csv("results/prcnn_metrics.csv", index=False)
            
            # 2. results/classification_report.csv
            df_report = pd.DataFrame(self.class_report_dict).transpose()
            df_report.to_csv("results/classification_report.csv", index=True)
            print("[REPORTS] Reports generated and written to CSV.")
        except Exception as e:
            raise RuntimeError(f"Error in generate_reports: {str(e)}")

    def print_statistics(self):
        """
        Outputs training, testing, and evaluation metrics onto the console.
        """
        try:
            print("\n" + "="*50)
            print("          WSN PRCNN IDS EVALUATION STATISTICS       ")
            print("="*50)
            print(f"Training Accuracy   : {self.train_accuracy:.4f}")
            print(f"Validation Accuracy : {self.val_accuracy:.4f}")
            print(f"Test Accuracy       : {self.test_accuracy:.4f}")
            print(f"Precision (Weighted): {self.test_precision:.4f}")
            print(f"Recall (Weighted)   : {self.test_recall:.4f}")
            print(f"F1 Score (Weighted) : {self.test_f1:.4f}")
            print(f"Specificity         : {self.test_specificity:.4f}")
            print(f"ROC-AUC (Weighted)  : {self.test_roc_auc:.4f}")
            print(f"Training Time       : {self.training_time:.4f} seconds")
            print(f"Inference Time      : {self.inference_time:.4f} seconds")
            print("="*50)
            print("\nClassification Report:")
            print(self.class_report_str)
            print("="*50 + "\n")
        except Exception as e:
            raise RuntimeError(f"Error in print_statistics: {str(e)}")

    def run(self):
        """
        Orchestration method to run the entire model pipeline.
        """
        self.load_dataset()
        self.build_model()
        self.compile_model()
        self.train_model()
        self.evaluate_model()
        self.save_model()
        self.generate_graphs()
        self.generate_reports()
        self.print_statistics()
