#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Title:          Cross-Validation, Statistical Testing, and Ablation Study
                for Insider Threat Detection Model Evaluation

Author:         Okonny, Kitoye Ebire
Institution:    Centre for Information and Telecommunication Engineering,
                University of Port Harcourt, Nigeria

Supervisors:    Prof. Eseosa Omorogiuwa, Dr. Matthew Ehikhamenle

Purpose:        This script implements the validation experiments described
                in Section 3.6 (Evaluation Metrics and Implementation) and
                reported in Sections 4.4 and 4.5 of the associated paper:

                "A Rigorous Data Preprocessing Pipeline and Interpretable
                Detection Framework for Insider Threat Classification in
                High-Dimensional Behavioural Telemetry"

Experiments:    1. Five-Fold Stratified Cross-Validation (Table 5)
                2. Pairwise Wilcoxon Signed-Rank Tests  (Table 6)
                3. AE-MLP Ablation Study                (Table 7)
                4. Confusion Matrices                   (Table 4)

Dataset:        Insider Threat Dataset for Classified Environments (Bdil, 2025)
                Preprocessed balanced version: 29,880 records, 37 features,
                50:50 class distribution (is_emp_malicious)

Environment:    Python 3.12, scikit-learn 1.8.0, scipy 1.17.0
                (Also includes a Google Colab version using XGBoost + TensorFlow
                 for replication with thesis-identical libraries — see Part B)

References:     - Breiman, L. (2001). Random forests. Machine Learning, 45(1).
                - Chen, T. & Guestrin, C. (2016). XGBoost. ACM SIGKDD.
                - Goodfellow, I. et al. (2016). Deep Learning. MIT Press.
                - Hinton, G.E. & Salakhutdinov, R.R. (2006). Science, 313.
                - Kingma, D.P. & Ba, J. (2015). Adam optimizer. ICLR.

Version:        1.0
Date:           March 2026
===============================================================================
"""


# =============================================================================
# SECTION 0: LIBRARY IMPORTS
# =============================================================================
# Standard libraries for data manipulation and numerical computation
import pandas as pd                         # DataFrame operations
import numpy as np                          # Numerical computing
import warnings                             # Suppress non-critical warnings
import json                                 # Save results in structured format
import time                                 # Track experiment duration
warnings.filterwarnings("ignore")           # Suppress sklearn convergence warnings

# Scikit-learn: Model selection and cross-validation
from sklearn.model_selection import StratifiedKFold     # K-fold with class balance
from sklearn.model_selection import train_test_split    # Single train-test split

# Scikit-learn: Classification algorithms
from sklearn.ensemble import RandomForestClassifier             # Breiman (2001)
from sklearn.ensemble import HistGradientBoostingClassifier     # Sklearn XGBoost equivalent
from sklearn.neural_network import MLPClassifier                # Feedforward neural network

# Scikit-learn: Preprocessing
from sklearn.preprocessing import StandardScaler    # Z-score standardisation

# Scikit-learn: Evaluation metrics
from sklearn.metrics import accuracy_score      # (TP + TN) / Total
from sklearn.metrics import precision_score     # TP / (TP + FP)
from sklearn.metrics import recall_score        # TP / (TP + FN) — critical for insider threats
from sklearn.metrics import f1_score            # Harmonic mean of precision and recall
from sklearn.metrics import roc_auc_score       # Area under ROC curve
from sklearn.metrics import confusion_matrix    # TP, FP, TN, FN counts

# SciPy: Statistical testing
from scipy.stats import wilcoxon    # Non-parametric paired comparison test


# =============================================================================
# SECTION 1: CONFIGURATION PARAMETERS
# =============================================================================
# All parameters are set once here for reproducibility and transparency.

# Path to the preprocessed balanced dataset (29,880 records)
# UPDATE THIS PATH when running in a different environment:
DATA_PATH = "/mnt/user-data/uploads/Final_ITD.csv"

# Random seed — ensures identical results on every run.
# Matches the thesis (Appendix D-G) which uses random_state=42 throughout.
RANDOM_STATE = 42

# Number of cross-validation folds.
# 5 folds = 80% training / 20% testing per fold, matching the thesis split ratio.
N_FOLDS = 5

# Bottleneck sizes for the AE-MLP ablation study.
# d=16 is the thesis default; d=8,24,32 test the Recall-Compression Paradox gradient.
ABLATION_BOTTLENECK_SIZES = [8, 16, 24, 32]

# Set global random seed for NumPy operations
np.random.seed(RANDOM_STATE)


# =============================================================================
# SECTION 2: DATA LOADING AND INITIAL INSPECTION
# =============================================================================
print("=" * 70)
print("INSIDER THREAT DETECTION — VALIDATION EXPERIMENTS")
print("=" * 70)

# Load the preprocessed balanced dataset
# Expected: 29,880 rows, 38 columns (37 features + 1 target)
df = pd.read_csv(DATA_PATH)

# Separate features (X) from target variable (y)
# Target column: 'is_emp_malicious' (1 = malicious, 0 = non-malicious)
X = df.drop(columns=["is_emp_malicious"])
y = df["is_emp_malicious"]

# Print dataset summary for verification
print(f"Dataset loaded:       {DATA_PATH}")
print(f"Total records:        {df.shape[0]}")
print(f"Feature count:        {X.shape[1]}")
print(f"Class distribution:   {y.value_counts().to_dict()}")
print(f"Missing values:       {df.isnull().sum().sum()}")
print(f"Random seed:          {RANDOM_STATE}")
print(f"Cross-validation:     {N_FOLDS}-fold stratified")
print()


# =============================================================================
# SECTION 3: AUTOENCODER IMPLEMENTATION (NUMPY)
# =============================================================================
# This implements Stage 1 of the AE-MLP architecture from the thesis (Section 3.4).
#
# Architecture (matching thesis Table 3.3 and Appendix G):
#   Encoder: Input(37) → Dense(32, ReLU) → Dense(encoding_dim, ReLU)
#   Decoder: Dense(32, ReLU) → Dense(37, Linear)
#
# Training:
#   Loss function:  Mean Squared Error (MSE) — reconstruction loss
#   Optimiser:      Mini-batch gradient descent with gradient clipping
#   Learning rate:  0.0005
#   Batch size:     64
#   Max epochs:     50
#   Early stopping: Patience = 5 epochs
#
# Note: This is a pure NumPy implementation because TensorFlow was not
# available in the execution environment. A TensorFlow version matching
# the thesis exactly is provided in Part B (Google Colab version) at the
# bottom of this file.
# =============================================================================

class NumpyAutoencoder:
    """
    Autoencoder for unsupervised feature compression.

    Implements the autoencoder component of the AE-MLP architecture
    described in thesis Section 3.4 (Classification Architectures).

    The autoencoder learns a compressed latent representation of
    employee behavioural features by minimising reconstruction error.
    The encoder portion is then used to transform features before
    passing them to an MLP classifier (Stage 2).

    Parameters
    ----------
    input_dim : int
        Number of input features (37 in the thesis dataset).
    encoding_dim : int, default=16
        Size of the latent bottleneck layer. Controls the compression
        ratio. Thesis default is 16 (compression ratio ~1:2.3).
    lr : float, default=0.0005
        Learning rate for gradient descent updates.
    epochs : int, default=50
        Maximum number of training epochs.
    batch_size : int, default=64
        Number of samples per mini-batch.

    Architecture
    ------------
    Layer 1 (Encoder):  input_dim → 32 neurons, ReLU activation
    Layer 2 (Encoder):  32 → encoding_dim neurons, ReLU activation
    Layer 3 (Decoder):  encoding_dim → 32 neurons, ReLU activation
    Layer 4 (Decoder):  32 → input_dim neurons, Linear activation
    """

    def __init__(self, input_dim, encoding_dim=16, lr=0.0005, epochs=50, batch_size=64):
        self.encoding_dim = encoding_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Weight initialisation: small random values to prevent gradient explosion
        # during early training epochs. Scale of 0.01 is conservative but stable.
        scale = 0.01

        # Encoder weights and biases
        self.W1 = np.random.randn(input_dim, 32) * scale       # Input → Hidden(32)
        self.b1 = np.zeros(32)                                  # Bias for Hidden(32)
        self.W2 = np.random.randn(32, encoding_dim) * scale     # Hidden(32) → Bottleneck
        self.b2 = np.zeros(encoding_dim)                        # Bias for Bottleneck

        # Decoder weights and biases (symmetric architecture)
        self.W3 = np.random.randn(encoding_dim, 32) * scale     # Bottleneck → Hidden(32)
        self.b3 = np.zeros(32)                                  # Bias for Hidden(32)
        self.W4 = np.random.randn(32, input_dim) * scale        # Hidden(32) → Output
        self.b4 = np.zeros(input_dim)                           # Bias for Output

    def _relu(self, x):
        """
        Rectified Linear Unit activation function.
        Returns max(0, x) element-wise.
        Used in all hidden layers to introduce non-linearity.
        Reference: Goodfellow et al. (2016), Chapter 6.
        """
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        """
        Derivative of ReLU for backpropagation.
        Returns 1 where x > 0, else 0.
        """
        return (x > 0).astype(float)

    def _clip(self, gradient, max_norm=1.0):
        """
        Gradient clipping to prevent numerical instability.

        If the L2 norm of a gradient exceeds max_norm, the gradient
        is scaled down proportionally. This prevents exploding gradients
        that can cause NaN values during training.

        Parameters
        ----------
        gradient : np.ndarray
            The gradient tensor to clip.
        max_norm : float, default=1.0
            Maximum allowed L2 norm.

        Returns
        -------
        np.ndarray
            Clipped gradient with L2 norm ≤ max_norm.
        """
        norm = np.linalg.norm(gradient)
        if norm > max_norm:
            return gradient * max_norm / norm
        return gradient

    def encode(self, X):
        """
        Forward pass through the encoder only.

        Used after training to extract compressed feature representations
        for the downstream MLP classifier (Stage 2 of AE-MLP).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_dim)
            Input feature matrix.

        Returns
        -------
        np.ndarray, shape (n_samples, encoding_dim)
            Compressed latent representation.
        """
        h1 = self._relu(X @ self.W1 + self.b1)     # Input → Hidden(32)
        h2 = self._relu(h1 @ self.W2 + self.b2)     # Hidden(32) → Bottleneck
        return h2

    def fit(self, X):
        """
        Train the autoencoder via mini-batch backpropagation.

        The autoencoder learns to reconstruct its input by minimising
        Mean Squared Error (MSE) between the input and the reconstruction.
        Training uses early stopping: if the loss does not improve for
        5 consecutive epochs, training halts to prevent overfitting.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_dim)
            Training data (StandardScaler-transformed features).

        Returns
        -------
        self
            The trained autoencoder instance.
        """
        n = X.shape[0]          # Total number of training samples
        best_loss = float('inf')  # Track best loss for early stopping
        patience = 5             # Number of epochs to wait before stopping
        wait = 0                 # Counter for epochs without improvement

        for epoch in range(self.epochs):
            # Shuffle training data at the start of each epoch
            indices = np.random.permutation(n)
            epoch_loss = 0

            # Process data in mini-batches
            for i in range(0, n, self.batch_size):
                batch = X[indices[i:i + self.batch_size]]
                bs = batch.shape[0]     # Actual batch size (may be smaller for last batch)

                # ---- FORWARD PASS ----
                # Encoder
                z1 = batch @ self.W1 + self.b1      # Pre-activation, Layer 1
                h1 = self._relu(z1)                  # Post-activation, Layer 1
                z2 = h1 @ self.W2 + self.b2          # Pre-activation, Layer 2 (bottleneck)
                h2 = self._relu(z2)                  # Post-activation, Layer 2 (bottleneck)

                # Decoder
                z3 = h2 @ self.W3 + self.b3          # Pre-activation, Layer 3
                h3 = self._relu(z3)                  # Post-activation, Layer 3
                z4 = h3 @ self.W4 + self.b4          # Reconstruction (linear output)

                # ---- BACKWARD PASS (Backpropagation) ----
                # Loss gradient: d(MSE)/d(z4) = 2 * (reconstruction - input) / batch_size
                dz4 = 2 * (z4 - batch) / bs

                # Layer 4 gradients (Decoder output)
                dW4 = self._clip(h3.T @ dz4)        # Weight gradient
                db4 = self._clip(dz4.sum(0))         # Bias gradient
                dh3 = dz4 @ self.W4.T                # Propagate to Layer 3

                # Layer 3 gradients (Decoder hidden)
                dz3 = dh3 * self._relu_deriv(z3)     # Apply ReLU derivative
                dW3 = self._clip(h2.T @ dz3)
                db3 = self._clip(dz3.sum(0))
                dh2 = dz3 @ self.W3.T                # Propagate to Layer 2

                # Layer 2 gradients (Encoder bottleneck)
                dz2 = dh2 * self._relu_deriv(z2)
                dW2 = self._clip(h1.T @ dz2)
                db2 = self._clip(dz2.sum(0))
                dh1 = dz2 @ self.W2.T                # Propagate to Layer 1

                # Layer 1 gradients (Encoder input)
                dz1 = dh1 * self._relu_deriv(z1)
                dW1 = self._clip(batch.T @ dz1)
                db1 = self._clip(dz1.sum(0))

                # ---- WEIGHT UPDATES (Gradient Descent) ----
                self.W4 -= self.lr * dW4
                self.b4 -= self.lr * db4
                self.W3 -= self.lr * dW3
                self.b3 -= self.lr * db3
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

                # Accumulate batch loss for epoch summary
                epoch_loss += np.mean((z4 - batch) ** 2) * bs

            # Compute average loss across all batches in this epoch
            epoch_loss /= n

            # ---- EARLY STOPPING CHECK ----
            # If loss improved by at least 0.0001, reset patience counter.
            # Otherwise, increment counter. Stop if no improvement for 5 epochs.
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        return self


# =============================================================================
# SECTION 4: AE-MLP PIPELINE FUNCTION
# =============================================================================
# This function implements the complete two-stage AE-MLP architecture:
#   Stage 1: Train autoencoder → extract compressed features
#   Stage 2: Train MLP classifier on compressed features → predict
# =============================================================================

def build_ae_mlp(X_train, y_train, X_test, encoding_dim=16):
    """
    Build and evaluate the Autoencoder-Enhanced MLP (AE-MLP).

    Implements the two-stage architecture described in thesis Section 3.4
    and Appendix G. Stage 1 trains an autoencoder to learn compressed
    representations. Stage 2 trains an MLP classifier on those representations.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features (raw, before scaling).
    y_train : pd.Series or np.ndarray
        Training labels (0 = non-malicious, 1 = malicious).
    X_test : pd.DataFrame or np.ndarray
        Test features (raw, before scaling).
    encoding_dim : int, default=16
        Bottleneck size. Varied in ablation study: {8, 16, 24, 32}.

    Returns
    -------
    y_pred : np.ndarray
        Binary predictions (0 or 1) for the test set.
    y_prob : np.ndarray
        Probability scores for the positive class (malicious).
    """
    # Standardise features (zero mean, unit variance)
    # Required for stable autoencoder training convergence
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)    # Fit on training data only
    X_te = scaler.transform(X_test)         # Transform test data using training statistics

    # ---- STAGE 1: Autoencoder Training ----
    ae = NumpyAutoencoder(X_tr.shape[1], encoding_dim=encoding_dim)
    ae.fit(X_tr)

    # Extract compressed features from the encoder
    X_tr_enc = ae.encode(X_tr)      # Training set: shape (n_train, encoding_dim)
    X_te_enc = ae.encode(X_te)      # Test set: shape (n_test, encoding_dim)

    # Safety check: replace any NaN or infinity values that may result
    # from numerical instability during encoding
    X_tr_enc = np.nan_to_num(X_tr_enc, nan=0.0, posinf=1.0, neginf=-1.0)
    X_te_enc = np.nan_to_num(X_te_enc, nan=0.0, posinf=1.0, neginf=-1.0)

    # ---- STAGE 2: MLP Classification ----
    # Architecture: encoding_dim → 32 neurons (ReLU) → 16 neurons (ReLU) → 1 output
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),        # Two hidden layers
        activation="relu",                  # ReLU activation (Goodfellow et al., 2016)
        solver="adam",                      # Adam optimiser (Kingma & Ba, 2015)
        learning_rate_init=0.001,           # Learning rate (thesis Table 3.3)
        max_iter=200,                       # Maximum training epochs
        early_stopping=True,                # Stop if validation loss plateaus
        validation_fraction=0.1,            # 10% of training data for validation
        random_state=RANDOM_STATE           # Reproducibility
    )
    clf.fit(X_tr_enc, y_train)

    # Generate predictions and probability scores
    y_pred = clf.predict(X_te_enc)                      # Binary: 0 or 1
    y_prob = clf.predict_proba(X_te_enc)[:, 1]          # P(malicious)

    return y_pred, y_prob


# =============================================================================
# SECTION 5: MODEL BUILDER FUNCTIONS
# =============================================================================
# Each function returns a fresh, untrained model instance with hyperparameters
# matching the thesis (Section 3.4, Tables 3.2 and 3.3). A new instance is
# created for each fold to prevent information leakage between folds.
# =============================================================================

def build_rf():
    """
    Build a Random Forest classifier.

    Hyperparameters match thesis Table 3.2 (best configuration from GridSearchCV):
        - 200 trees (n_estimators)
        - No maximum depth constraint (max_depth=None)
        - Minimum 2 samples to split a node
        - Minimum 1 sample per leaf
        - sqrt(n_features) features considered per split

    Reference: Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
    """
    return RandomForestClassifier(
        n_estimators=200,           # Number of decision trees in the forest
        max_depth=None,             # Trees grow until all leaves are pure
        min_samples_split=2,        # Minimum samples needed to split a node
        min_samples_leaf=1,         # Minimum samples in each leaf node
        max_features="sqrt",        # sqrt(37) ≈ 6 features per split
        random_state=RANDOM_STATE,  # Reproducibility
        n_jobs=-1                   # Use all available CPU cores
    )


def build_xgb():
    """
    Build a gradient boosting classifier.

    Uses sklearn's HistGradientBoostingClassifier, which implements the same
    histogram-based gradient boosting algorithm as XGBoost (Chen & Guestrin, 2016).
    Hyperparameters approximate thesis Table 3.2 (best XGBoost configuration).

    Note: The Google Colab version (Part B) uses the actual XGBoost library
    for exact replication of thesis results.

    Reference: Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree
               boosting system. ACM SIGKDD, 785-794.
    """
    return HistGradientBoostingClassifier(
        max_iter=200,               # Number of boosting rounds (≈ n_estimators)
        max_depth=5,                # Maximum tree depth
        learning_rate=0.1,          # Step size shrinkage (thesis: 0.1)
        min_samples_leaf=20,        # Minimum samples per leaf
        l2_regularization=0.1,      # L2 penalty (≈ gamma in XGBoost)
        random_state=RANDOM_STATE   # Reproducibility
    )


def build_mlp():
    """
    Build a Multilayer Perceptron classifier.

    Hyperparameters match thesis Table 3.3 and Appendix F:
        - Single hidden layer with 128 neurons
        - ReLU activation
        - Adam optimiser with learning rate 0.001
        - Early stopping with 10% validation split

    Reference: Goodfellow, I. et al. (2016). Deep Learning. MIT Press.
               Kingma, D.P. & Ba, J. (2015). Adam optimizer. ICLR.
    """
    return MLPClassifier(
        hidden_layer_sizes=(128,),          # One hidden layer, 128 neurons
        activation="relu",                  # ReLU activation function
        solver="adam",                      # Adam optimiser
        alpha=0.0001,                       # L2 regularisation strength
        learning_rate="constant",           # Fixed learning rate
        learning_rate_init=0.001,           # Learning rate value (thesis: 0.001)
        max_iter=500,                       # Maximum training iterations
        early_stopping=True,                # Stop when validation score plateaus
        validation_fraction=0.1,            # 10% held out for early stopping
        random_state=RANDOM_STATE           # Reproducibility
    )


# =============================================================================
# SECTION 6: EVALUATION HELPER FUNCTION
# =============================================================================

def evaluate(y_true, y_pred, y_prob):
    """
    Compute all five classification metrics used in the paper.

    These metrics are defined in thesis Section 3.6 and reported in
    Tables 1, 3, 5, and 7 of the paper.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (0 or 1).
    y_pred : array-like
        Predicted binary labels (0 or 1).
    y_prob : array-like
        Predicted probability of the positive class (malicious).

    Returns
    -------
    dict
        Dictionary with keys: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

    Notes
    -----
    - Recall is prioritised interpretively because in insider threat detection,
      missing a malicious insider (false negative) is more costly than
      generating a false alarm (false positive).
    - ROC-AUC measures discriminatory ability across all classification thresholds.
    """
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall":    recall_score(y_true, y_pred),
        "F1-Score":  f1_score(y_true, y_pred),
        "ROC-AUC":   roc_auc_score(y_true, y_prob),
    }


# =============================================================================
# SECTION 7: EXPERIMENT 1 — FIVE-FOLD STRATIFIED CROSS-VALIDATION
# =============================================================================
# Purpose: Validate that the performance hierarchy (RF ≈ XGBoost > MLP > AE-MLP)
#          is robust across different data partitions, not an artefact of a
#          single lucky train-test split.
#
# Method:  StratifiedKFold splits the dataset into 5 equal-sized folds while
#          preserving the 50:50 class balance in each fold. Each fold takes a
#          turn as the test set while the other 4 folds form the training set.
#          This ensures every record is tested exactly once.
#
# Output:  Mean ± standard deviation of each metric across 5 folds (Table 5).
# =============================================================================

print("=" * 70)
print(f"EXPERIMENT 1: {N_FOLDS}-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)

# Create the 5-fold splitter
# - n_splits=5:       Split data into 5 equal piles
# - shuffle=True:     Randomise which records go into which pile
# - random_state=42:  Same shuffle every time for reproducibility
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Metric names (same order as thesis Table 4.1)
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

# Model names (same order as thesis: ensemble models first, then deep learning)
model_names = ["XGBoost", "Random Forest", "MLP", "AE-MLP"]

# Storage: results[model_name][metric_name] = list of 5 fold values
results = {m: {met: [] for met in metrics_names} for m in model_names}

# Record start time
t0 = time.time()

# ---- MAIN CROSS-VALIDATION LOOP ----
# skf.split(X, y) yields 5 pairs of (train_indices, test_indices).
# The 'y' argument ensures stratification (class balance preserved per fold).
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")

    # Extract training and testing subsets for this fold
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]   # Features
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]    # Labels

    # ---- Random Forest ----
    # Build fresh model (no data leakage between folds)
    rf = build_rf()
    rf.fit(X_tr, y_tr)                                   # Train on this fold's training set
    m = evaluate(y_te, rf.predict(X_te), rf.predict_proba(X_te)[:, 1])
    for k, v in m.items():
        results["Random Forest"][k].append(v)            # Store this fold's scores
    print(f"  RF:     Acc={m['Accuracy']:.4f}  Prec={m['Precision']:.4f}  "
          f"Rec={m['Recall']:.4f}  F1={m['F1-Score']:.4f}  AUC={m['ROC-AUC']:.4f}")

    # ---- XGBoost (HistGradientBoosting) ----
    xgb = build_xgb()
    xgb.fit(X_tr, y_tr)
    y_prob_xgb = xgb.predict_proba(X_te)[:, 1]
    m = evaluate(y_te, xgb.predict(X_te), y_prob_xgb)
    for k, v in m.items():
        results["XGBoost"][k].append(v)
    print(f"  XGB:    Acc={m['Accuracy']:.4f}  Prec={m['Precision']:.4f}  "
          f"Rec={m['Recall']:.4f}  F1={m['F1-Score']:.4f}  AUC={m['ROC-AUC']:.4f}")

    # ---- MLP ----
    # MLP requires standardised input for stable gradient descent convergence.
    # StandardScaler is fit ONLY on training data to prevent information leakage.
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)     # Fit and transform training data
    X_te_s = scaler.transform(X_te)         # Transform test data using training statistics
    mlp = build_mlp()
    mlp.fit(X_tr_s, y_tr)
    m = evaluate(y_te, mlp.predict(X_te_s), mlp.predict_proba(X_te_s)[:, 1])
    for k, v in m.items():
        results["MLP"][k].append(v)
    print(f"  MLP:    Acc={m['Accuracy']:.4f}  Prec={m['Precision']:.4f}  "
          f"Rec={m['Recall']:.4f}  F1={m['F1-Score']:.4f}  AUC={m['ROC-AUC']:.4f}")

    # ---- AE-MLP (d=16, thesis default bottleneck) ----
    y_pred_ae, y_prob_ae = build_ae_mlp(X_tr, y_tr, X_te, encoding_dim=16)
    m = evaluate(y_te, y_pred_ae, y_prob_ae)
    for k, v in m.items():
        results["AE-MLP"][k].append(v)
    print(f"  AE-MLP: Acc={m['Accuracy']:.4f}  Prec={m['Precision']:.4f}  "
          f"Rec={m['Recall']:.4f}  F1={m['F1-Score']:.4f}  AUC={m['ROC-AUC']:.4f}")

cv_time = time.time() - t0
print(f"\nCross-validation completed in {cv_time:.1f} seconds")


# ---- Print Table 5: Cross-Validation Summary ----
print("\n" + "=" * 70)
print("TABLE 5: 5-FOLD CROSS-VALIDATION RESULTS (Mean ± Std)")
print("=" * 70)

header = f"{'Model':<16}"
for met in metrics_names:
    header += f" {met:<20}"
print(header)
print("-" * 118)

cv_summary = {}
for model in model_names:
    row = f"{model:<16}"
    cv_summary[model] = {}
    for met in metrics_names:
        vals = results[model][met]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        cv_summary[model][met] = (mean_val, std_val)
        row += f" {mean_val:.4f} ± {std_val:.4f}    "
    print(row)

# ---- Print fold-level detail for full transparency ----
print("\n--- Fold-Level Detail ---")
for model in model_names:
    print(f"\n{model}:")
    for met in metrics_names:
        vals = results[model][met]
        vals_str = ", ".join([f"{v:.4f}" for v in vals])
        print(f"  {met:<12} Folds: [{vals_str}]  Mean={np.mean(vals):.4f}  Std={np.std(vals):.4f}")


# =============================================================================
# SECTION 8: EXPERIMENT 2 — PAIRWISE WILCOXON SIGNED-RANK TESTS
# =============================================================================
# Purpose: Determine whether the performance differences between model pairs
#          are statistically significant (not due to random data partitioning).
#
# Method:  The Wilcoxon signed-rank test is a non-parametric test for paired
#          samples. It compares the fold-level scores of two models and tests
#          the null hypothesis that their median difference is zero.
#
# Interpretation:
#   - W statistic: Sum of ranks where the lower-performing model won.
#     W = 0 means the higher model won every single fold.
#   - p-value: Probability of observing this result if models were equal.
#     p < 0.05 → reject null hypothesis → difference is significant.
#
# Known limitation: With only N=5 folds, the minimum achievable two-sided
#   p-value is 2/2^5 = 0.0625, which exceeds α = 0.05. This is a statistical
#   power limitation, not evidence of equivalence. The paper addresses this
#   by also reporting a one-sided sign test (p = (1/2)^5 = 0.03125 < 0.05).
#
# Reference: Demšar, J. (2006). Statistical comparisons of classifiers over
#            multiple datasets. JMLR, 7, 1-30.
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 2: PAIRWISE WILCOXON SIGNED-RANK TESTS (α = 0.05)")
print("=" * 70)

# Define all 6 pairwise comparisons (4 models → C(4,2) = 6 pairs)
pairs = [
    ("XGBoost", "Random Forest"),       # Both ensemble — closest comparison
    ("XGBoost", "MLP"),                 # Best ensemble vs standard deep learning
    ("XGBoost", "AE-MLP"),             # Best ensemble vs autoencoder deep learning
    ("Random Forest", "MLP"),           # Second ensemble vs standard deep learning
    ("Random Forest", "AE-MLP"),        # Second ensemble vs autoencoder deep learning
    ("MLP", "AE-MLP"),                 # Standard DL vs autoencoder DL
]

print(f"\n{'Comparison':<30} {'W(ROC)':<10} {'p(ROC)':<12} {'W(F1)':<10} {'p(F1)':<12} {'Sig?':<8}")
print("-" * 82)

stat_results = []
for m1, m2 in pairs:
    # Extract fold-level scores for both models
    roc1 = results[m1]["ROC-AUC"]       # List of 5 ROC-AUC values for model 1
    roc2 = results[m2]["ROC-AUC"]       # List of 5 ROC-AUC values for model 2
    f1_1 = results[m1]["F1-Score"]      # List of 5 F1 values for model 1
    f1_2 = results[m2]["F1-Score"]      # List of 5 F1 values for model 2

    # Wilcoxon test on ROC-AUC scores
    try:
        w_roc, p_roc = wilcoxon(roc1, roc2)
    except Exception:
        # Fails if all differences are zero (identical scores)
        w_roc, p_roc = "N/A", 1.0

    # Wilcoxon test on F1-Score scores
    try:
        w_f1, p_f1 = wilcoxon(f1_1, f1_2)
    except Exception:
        w_f1, p_f1 = "N/A", 1.0

    # Significant if either metric's p-value < 0.05
    sig = "Yes" if (isinstance(p_roc, float) and p_roc < 0.05) or \
                    (isinstance(p_f1, float) and p_f1 < 0.05) else "No"

    print(f"{m1} vs {m2:<16} {str(w_roc):<10} "
          f"{str(round(p_roc, 4) if isinstance(p_roc, float) else p_roc):<12} "
          f"{str(w_f1):<10} "
          f"{str(round(p_f1, 4) if isinstance(p_f1, float) else p_f1):<12} {sig:<8}")

    stat_results.append({
        "pair": f"{m1} vs {m2}",
        "w_roc": str(w_roc), "p_roc": round(p_roc, 4) if isinstance(p_roc, float) else p_roc,
        "w_f1": str(w_f1), "p_f1": round(p_f1, 4) if isinstance(p_f1, float) else p_f1,
        "sig": sig
    })

# ---- One-sided sign test interpretation ----
print("\n--- Statistical Power Note ---")
print(f"With N={N_FOLDS} folds, minimum two-sided Wilcoxon p = 2/2^{N_FOLDS} = {2/2**N_FOLDS:.4f}")
print(f"One-sided sign test p = (1/2)^{N_FOLDS} = {(1/2)**N_FOLDS:.5f}")
print("When W=0.0, Model A beat Model B on EVERY fold.")
print(f"Probability of this by chance: {(1/2)**N_FOLDS:.5f} < 0.05 → significant.")


# =============================================================================
# SECTION 9: EXPERIMENT 3 — AE-MLP ABLATION STUDY
# =============================================================================
# Purpose: Characterise how the autoencoder bottleneck size affects insider
#          threat detection performance, providing empirical evidence for the
#          Recall-Compression Paradox (thesis Section 5.5, Contribution ii).
#
# Method:  Vary the latent space dimensionality d across {8, 16, 24, 32} while
#          keeping all other hyperparameters constant. Each configuration is
#          evaluated using the same 5-fold cross-validation protocol.
#
# Hypothesis: As d decreases (more aggressive compression), recall should
#             decrease because the autoencoder's reconstruction-error objective
#             prioritises majority-class patterns, suppressing the rare
#             behavioural signals associated with malicious insiders.
#
# Configurations:
#   d=8  → compression ratio 1:4.6 (most aggressive)
#   d=16 → compression ratio 1:2.3 (thesis default)
#   d=24 → compression ratio 1:1.5
#   d=32 → compression ratio 1:1.2 (least aggressive)
#
# Output: Performance metrics per bottleneck size (Table 7).
# =============================================================================

print("\n" + "=" * 70)
print(f"EXPERIMENT 3: AE-MLP ABLATION STUDY")
print(f"Bottleneck sizes: {ABLATION_BOTTLENECK_SIZES}")
print(f"Input dimensionality: {X.shape[1]} features")
print(f"Using same {N_FOLDS}-fold splits as Experiment 1")
print("=" * 70)

# Storage: ablation_results[bottleneck_size][metric_name] = list of 5 fold values
ablation_results = {d: {met: [] for met in metrics_names} for d in ABLATION_BOTTLENECK_SIZES}

t0 = time.time()

# Test each bottleneck configuration
for encoding_dim in ABLATION_BOTTLENECK_SIZES:
    ratio = X.shape[1] / encoding_dim
    print(f"\n  Testing d={encoding_dim} (compression ratio 1:{ratio:.1f})...")

    # Use the SAME fold splits as Experiment 1 for fair comparison
    # (StratifiedKFold with same random_state produces identical splits)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Build and evaluate AE-MLP with this bottleneck size
        y_pred_ae, y_prob_ae = build_ae_mlp(X_tr, y_tr, X_te, encoding_dim=encoding_dim)
        m = evaluate(y_te, y_pred_ae, y_prob_ae)

        for k, v in m.items():
            ablation_results[encoding_dim][k].append(v)

        print(f"    Fold {fold_idx + 1}: "
              f"Acc={m['Accuracy']:.4f}  Rec={m['Recall']:.4f}  AUC={m['ROC-AUC']:.4f}")

ab_time = time.time() - t0
print(f"\nAblation study completed in {ab_time:.1f} seconds")

# ---- Print Table 7: Ablation Results ----
print("\n" + "=" * 70)
print("TABLE 7: AE-MLP ABLATION RESULTS")
print("=" * 70)

header = f"{'d':<6} {'Ratio':<10}"
for met in metrics_names:
    header += f" {met:<20}"
print(header)
print("-" * 116)

input_dim = X.shape[1]
for d in ABLATION_BOTTLENECK_SIZES:
    ratio = f"1:{input_dim / d:.1f}"
    row = f"{d:<6} {ratio:<10}"
    for met in metrics_names:
        vals = ablation_results[d][met]
        row += f" {np.mean(vals):.4f} ± {np.std(vals):.4f}    "
    print(row)

# ---- Ablation fold-level detail ----
print("\n--- Ablation Fold-Level Detail ---")
for d in ABLATION_BOTTLENECK_SIZES:
    print(f"\nd={d} (ratio 1:{input_dim/d:.1f}):")
    for met in metrics_names:
        vals = ablation_results[d][met]
        vals_str = ", ".join([f"{v:.4f}" for v in vals])
        print(f"  {met:<12} Folds: [{vals_str}]  Mean={np.mean(vals):.4f}")


# =============================================================================
# SECTION 10: EXPERIMENT 4 — CONFUSION MATRICES (SINGLE 80:20 SPLIT)
# =============================================================================
# Purpose: Provide TP, FP, TN, FN counts for Table 4 in the paper.
#          Uses a single 80:20 stratified split matching the thesis protocol.
#
# Interpretation:
#   TP (True Positive):  Correctly identified as malicious
#   FP (False Positive): Non-malicious incorrectly flagged as malicious
#   TN (True Negative):  Correctly identified as non-malicious
#   FN (False Negative): Malicious employee missed — the most dangerous error
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 4: CONFUSION MATRICES (80:20 Single Split)")
print("=" * 70)

# Single stratified split matching thesis Section 3.7.3
X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
    X, y,
    test_size=0.2,                  # 20% for testing
    random_state=RANDOM_STATE,      # Same seed as thesis
    stratify=y                      # Preserve class balance
)

print(f"Training set:  {X_train_single.shape[0]} records")
print(f"Test set:      {X_test_single.shape[0]} records")
print(f"Test balance:  {y_test_single.value_counts().to_dict()}")

# ---- Train and evaluate each model on the single split ----

# Random Forest
rf = build_rf()
rf.fit(X_train_single, y_train_single)
cm_rf = confusion_matrix(y_test_single, rf.predict(X_test_single))

# XGBoost
xgb = build_xgb()
xgb.fit(X_train_single, y_train_single)
cm_xgb = confusion_matrix(y_test_single, xgb.predict(X_test_single))

# MLP (requires scaling)
scaler_single = StandardScaler()
X_train_s = scaler_single.fit_transform(X_train_single)
X_test_s = scaler_single.transform(X_test_single)
mlp = build_mlp()
mlp.fit(X_train_s, y_train_single)
cm_mlp = confusion_matrix(y_test_single, mlp.predict(X_test_s))

# AE-MLP
y_pred_ae_s, _ = build_ae_mlp(X_train_single, y_train_single, X_test_single, encoding_dim=16)
cm_ae = confusion_matrix(y_test_single, y_pred_ae_s)

# ---- Print confusion matrix summary table ----
print(f"\n{'Model':<20} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Total':<8}")
print("-" * 52)
for name, cm in [("Random Forest", cm_rf), ("XGBoost", cm_xgb),
                  ("MLP", cm_mlp), ("AE-MLP (d=16)", cm_ae)]:
    # confusion_matrix layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    print(f"{name:<20} {tp:<8} {fp:<8} {tn:<8} {fn:<8} {cm.sum():<8}")

# ---- Print individual confusion matrices in standard format ----
for name, cm in [("Random Forest", cm_rf), ("XGBoost", cm_xgb),
                  ("MLP", cm_mlp), ("AE-MLP (d=16)", cm_ae)]:
    print(f"\n{name} Confusion Matrix:")
    print(f"                 Predicted: 0    Predicted: 1")
    print(f"  Actual: 0      TN={cm[0,0]:<8}   FP={cm[0,1]:<8}")
    print(f"  Actual: 1      FN={cm[1,0]:<8}   TP={cm[1,1]:<8}")


# =============================================================================
# SECTION 11: SAVE ALL RESULTS TO JSON
# =============================================================================
# Saves all experimental results in a structured JSON file for:
#   1. Programmatic insertion into the paper document
#   2. Archival and reproducibility verification
#   3. Supervisor/examiner review
# =============================================================================

output = {
    "metadata": {
        "script": "Cross-Validation, Statistical Testing, and Ablation Study",
        "dataset": DATA_PATH,
        "records": int(df.shape[0]),
        "features": int(X.shape[1]),
        "random_state": RANDOM_STATE,
        "n_folds": N_FOLDS,
        "ablation_sizes": ABLATION_BOTTLENECK_SIZES,
    },
    "cv_results": {
        m: {met: {
                "mean": round(cv_summary[m][met][0], 4),
                "std": round(cv_summary[m][met][1], 4),
                "fold_values": [round(v, 4) for v in results[m][met]]
            } for met in metrics_names
        } for m in model_names
    },
    "stat_tests": stat_results,
    "ablation": {
        str(d): {met: {
                "mean": round(np.mean(ablation_results[d][met]), 4),
                "std": round(np.std(ablation_results[d][met]), 4),
                "fold_values": [round(v, 4) for v in ablation_results[d][met]]
            } for met in metrics_names
        } for d in ABLATION_BOTTLENECK_SIZES
    },
    "confusion_matrices": {
        "Random Forest": {"TP": int(cm_rf[1,1]), "FP": int(cm_rf[0,1]),
                          "TN": int(cm_rf[0,0]), "FN": int(cm_rf[1,0])},
        "XGBoost":       {"TP": int(cm_xgb[1,1]), "FP": int(cm_xgb[0,1]),
                          "TN": int(cm_xgb[0,0]), "FN": int(cm_xgb[1,0])},
        "MLP":           {"TP": int(cm_mlp[1,1]), "FP": int(cm_mlp[0,1]),
                          "TN": int(cm_mlp[0,0]), "FN": int(cm_mlp[1,0])},
        "AE-MLP":        {"TP": int(cm_ae[1,1]), "FP": int(cm_ae[0,1]),
                          "TN": int(cm_ae[0,0]), "FN": int(cm_ae[1,0])},
    },
}

# Save to JSON file
output_path = "experiment_results.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n{'=' * 70}")
print(f"ALL EXPERIMENTS COMPLETE")
print(f"Results saved to: {output_path}")
print(f"Total execution time: {time.time() - t0:.1f} seconds")
print(f"{'=' * 70}")


# =============================================================================
# END OF PART A
# =============================================================================
# Part B (Google Colab version with XGBoost + TensorFlow) is available in
# the file: complete_experiments.py (already provided separately).
# =============================================================================
