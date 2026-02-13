"""
NARMA-L2 (Nonlinear Autoregressive Moving Average - Level 2) controller.

Implements the feedback linearization approach from:
    y(k+d) = f(.) + g(.) * u(k+1)

where f and g are approximated by neural networks trained on synthetic ABS data.

Training strategy:
    1. Train g-network on an initial approximation of system sensitivity
    2. Use g-network predictions to compute corrected f targets: f_k = y(k+1) - g_k * u(k+1)
    3. Train f-network with corrected targets
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class NARMAL2Controller:
    """NARMA-L2 neural network controller for ABS slip regulation.

    Uses two neural networks (f-network and g-network) to approximate
    the nonlinear system dynamics and compute optimal control inputs.

    Parameters
    ----------
    na : int
        Number of past outputs used as features (default: 2).
    nb : int
        Number of past inputs used as features (default: 2).
    d : int
        System delay (default: 1).
    hidden_neurons : int
        Number of neurons in the hidden layer (default: 5, as in the paper).
    """

    def __init__(self, na=2, nb=2, d=1, hidden_neurons=5):
        self.na = na
        self.nb = nb
        self.d = d
        self.hidden_neurons = hidden_neurons

        self.input_scaler = StandardScaler()
        self.output_scaler_f = StandardScaler()
        self.output_scaler_g = StandardScaler()

        self.f_network = None
        self.g_network = None

        self.X_test = None
        self.y_test = None

    def _create_network(self, input_dim, output_dim=1):
        """Create a single-hidden-layer neural network.

        Architecture follows the paper: tanh hidden layer, linear output,
        trained with Adam optimizer (approximation of Levenberg-Marquardt).
        """
        model = Sequential([
            layers.Dense(self.hidden_neurons, activation="tanh", input_shape=(input_dim,)),
            layers.Dense(output_dim, activation="linear"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse")
        return model

    def prepare_training_data(self, df):
        """Prepare input features and targets from simulation data.

        Creates feature vectors: [y(k), y(k-1), y(k-2), u(k), u(k-1), u(k-2)]

        Parameters
        ----------
        df : pd.DataFrame
            Simulation data from ABSSystemSimulator.

        Returns
        -------
        tuple of np.ndarray
            (X, y_f, y_g, u_kplus1) arrays.
        """
        X, y_f, y_g, u_kplus1 = [], [], [], []

        for i in range(len(df) - self.d - 1):
            y_k = df.iloc[i]["slip_ratio"]
            y_k1 = df.iloc[i]["slip_k_1"]
            y_k2 = df.iloc[i]["slip_k_2"]

            u_k = df.iloc[i]["control_input"]
            u_k1 = df.iloc[i]["control_k_1"]
            u_k2 = df.iloc[i]["control_k_2"]

            feature = [y_k, y_k1, y_k2, u_k, u_k1, u_k2]
            X.append(feature)

            y_target = df.iloc[i + 1]["slip_ratio"]
            u_next = df.iloc[i + 1]["control_input"]

            y_f.append(y_target)
            u_kplus1.append(u_next)
            y_g.append(0.1)  # Initial g approximation

        return np.array(X), np.array(y_f), np.array(y_g), np.array(u_kplus1)

    def train(self, df, epochs=100, batch_size=32, test_size=0.2, verbose=1):
        """Train the NARMA-L2 controller using a two-step process.

        Step 1: Train g-network on initial sensitivity approximation.
        Step 2: Use g predictions to compute corrected f targets, then train f-network.

        Parameters
        ----------
        df : pd.DataFrame
            Training data from ABSSystemSimulator.
        epochs : int
            Number of training epochs for the f-network.
        batch_size : int
            Training batch size.
        test_size : float
            Fraction of data reserved for testing.
        verbose : int
            Keras verbosity level.

        Returns
        -------
        tuple
            (history_f, history_g) training history objects.
        """
        print("Preparing training data...")
        X, y_f_raw, y_g, u_kplus1 = self.prepare_training_data(df)

        X_train, X_test, y_f_train, y_f_test, u_train, u_test, y_g_train, y_g_test = (
            train_test_split(X, y_f_raw, u_kplus1, y_g, test_size=test_size, shuffle=False)
        )

        X_train_scaled = self.input_scaler.fit_transform(X_train)
        X_test_scaled = self.input_scaler.transform(X_test)

        print(f"Training data shape: {X_train_scaled.shape}")
        print(f"Test data shape: {X_test_scaled.shape}")

        # Step 1: Train g-network
        print("\nStep 1: Training g-network (system sensitivity)...")
        self.g_network = self._create_network(X_train.shape[1])
        y_g_train_scaled = self.output_scaler_g.fit_transform(
            np.array(y_g_train).reshape(-1, 1)
        ).flatten()

        history_g = self.g_network.fit(
            X_train_scaled, y_g_train_scaled,
            epochs=epochs // 2,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=0.2,
        )

        # Step 2: Compute corrected f targets
        print("\nStep 2: Calculating corrected targets for f-network...")
        g_train_pred = self.g_network.predict(X_train_scaled, verbose=0).flatten()
        g_train_pred = self.output_scaler_g.inverse_transform(
            g_train_pred.reshape(-1, 1)
        ).flatten()

        y_f_corrected = y_f_train - g_train_pred * u_train
        y_f_scaled = self.output_scaler_f.fit_transform(
            y_f_corrected.reshape(-1, 1)
        ).flatten()

        print(f"f-target statistics: mean={np.mean(y_f_corrected):.4f}, std={np.std(y_f_corrected):.4f}")

        # Step 3: Train f-network
        print("\nStep 3: Training f-network...")
        self.f_network = self._create_network(X_train.shape[1])
        history_f = self.f_network.fit(
            X_train_scaled, y_f_scaled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=0.2,
        )

        self.X_test = X_test_scaled
        self.y_test = y_f_test
        self.u_test = u_test

        print("\nNARMA-L2 training completed!")

        self._plot_training_history(history_f, history_g)

        return history_f, history_g

    def _plot_training_history(self, history_f, history_g):
        """Plot training and validation loss curves for both networks."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(history_f.history["loss"], label="Training Loss", linewidth=2)
        ax1.plot(history_f.history["val_loss"], label="Validation Loss", linewidth=2)
        ax1.set_title("f-Network Training History", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (MSE)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        ax2.plot(history_g.history["loss"], label="Training Loss", linewidth=2)
        ax2.plot(history_g.history["val_loss"], label="Validation Loss", linewidth=2)
        ax2.set_title("g-Network Training History", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss (MSE)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

        plt.tight_layout()
        plt.show()

    def predict(self, X):
        """Predict the f-component for given input features."""
        X_scaled = self.input_scaler.transform(X)
        f_pred_scaled = self.f_network.predict(X_scaled, verbose=0)
        return self.output_scaler_f.inverse_transform(f_pred_scaled).flatten()

    def compute_control_input(self, current_state, reference_slip=0.25):
        """Compute the NARMA-L2 control input for a given state.

        Uses the feedback linearization control law:
            u(k+1) = (y_ref - f_pred) / g_pred

        Parameters
        ----------
        current_state : np.ndarray
            Feature vector [y(k), y(k-1), y(k-2), u(k), u(k-1), u(k-2)].
        reference_slip : float
            Desired slip ratio (default: 0.25).

        Returns
        -------
        float
            Control input (brake pressure), clipped to [0, 1800] Pa.
        """
        if self.f_network is None or self.g_network is None:
            raise ValueError("Model not trained.")

        X_scaled = self.input_scaler.transform(current_state.reshape(1, -1))

        f_pred_scaled = self.f_network.predict(X_scaled, verbose=0)[0, 0]
        g_pred_scaled = self.g_network.predict(X_scaled, verbose=0)[0, 0]

        f_pred = self.output_scaler_f.inverse_transform([[f_pred_scaled]])[0, 0]
        g_pred = self.output_scaler_g.inverse_transform([[g_pred_scaled]])[0, 0]

        g_pred = max(g_pred, 0.05)

        u_next = (reference_slip - f_pred) / (g_pred + 1e-8)
        u_next = np.clip(u_next, 0, 1800)

        return u_next
