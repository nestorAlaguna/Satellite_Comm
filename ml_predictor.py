# ml_predictor.py (CPU ONLY VERSION)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from typing import List, Tuple, Optional
import joblib
from datetime import datetime, timedelta

class FadePredictor:
    """ 
    Machine Learning module for predicting signal fades in optical satellite links.
    Uses Random Forest for reliable, interpretable predictions.
    """
    
    def __init__(self):
        self.rf_model = None
        self.is_trained = False
        self.scaling_factors = None
        
    def generate_training_data(self, sequence_length: int = 10, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data that simulates atmospheric turbulence effects.
        Reduced n_samples for faster training on CPU.
        """
        print(" Generating synthetic training data for fade prediction...")
        
        X, y = [], []
        
        for i in range(n_samples):
            # Simulate a time series of received power with realistic fades
            time_steps = sequence_length + 1  # +1 for the prediction target
            
            # Base signal (normal operation)
            base_power = -70  # dBm baseline
            
            # Add slow variations (satellite motion)
            slow_variation = 2 * np.sin(2 * np.pi * np.arange(time_steps) / 50)
            
            # Add atmospheric turbulence (Rayleigh fading-like behavior)
            turbulence = np.random.rayleigh(scale=3, size=time_steps)
            fade_indices = np.random.choice(time_steps, size=2, replace=False)
            turbulence[fade_indices] *= 5  # Create deep fades
            
            # Add random noise
            noise = np.random.normal(0, 0.5, time_steps)
            
            # Combine all components
            received_power = base_power + slow_variation - turbulence + noise
            
            # Create sequences for training
            X.append(received_power[:-1])  # First 10 values
            y.append(received_power[-1])   # 11th value (2 minutes ahead)
        
        X = np.array(X)
        y = np.array(y)
        
        # Store scaling factors for later use
        self.scaling_factors = {
            'X_mean': X.mean(),
            'X_std': X.std(),
            'y_mean': y.mean(),
            'y_std': y.std()
        }
        
        # Normalize the data
        X = (X - self.scaling_factors['X_mean']) / self.scaling_factors['X_std']
        y = (y - self.scaling_factors['y_mean']) / self.scaling_factors['y_std']
        
        print(f" Generated {len(X)} training samples")
        print(f"   Input shape: {X.shape}, Output shape: {y.shape}")
        
        return X, y
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Train a Random Forest model for fade prediction."""
        print("\n Training Random Forest model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Create and train the model (smaller for faster CPU training)
        self.rf_model = RandomForestRegressor(
            n_estimators=50,  # Reduced from 100 for faster training
            max_depth=8,      # Reduced from 10
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.rf_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f" Random Forest Training Complete:")
        print(f"   MAE: {mae:.4f}")
        print(f"   R² Score: {r2:.4f}")
        
        self.is_trained = True
        return mae, r2
    
    def predict_fade(self, recent_power_measurements: List[float]) -> Optional[float]:
        """
        Predict the received power 2 minutes into the future.
        """
        if not self.is_trained or self.rf_model is None:
            print(" Model not trained yet!")
            return None
        
        # Convert to numpy array and ensure correct length
        if len(recent_power_measurements) != 10:
            raise ValueError(f"Expected 10 measurements, got {len(recent_power_measurements)}")
        
        X_input = np.array(recent_power_measurements)
        
        # Normalize using stored scaling factors
        X_input = (X_input - self.scaling_factors['X_mean']) / self.scaling_factors['X_std']
        
        # Reshape for prediction
        X_input = X_input.reshape(1, -1)
        
        # Make prediction
        y_pred_normalized = self.rf_model.predict(X_input)[0]
        
        # Denormalize back to original scale
        y_pred = y_pred_normalized * self.scaling_factors['y_std'] + self.scaling_factors['y_mean']
        
        return y_pred
    
    def analyze_fade_risk(self, current_power: float, predicted_power: float, 
                         fade_threshold: float = -75.0) -> dict:
        """
        Analyze the risk of an upcoming fade and suggest mitigation strategies.
        """
        risk_analysis = {
            'current_power': current_power,
            'predicted_power': predicted_power,
            'power_change': predicted_power - current_power,
            'fade_risk': 'LOW',
            'suggested_actions': [],
            'prediction_confidence': 0.85
        }
        
        # Determine fade risk level
        if predicted_power < fade_threshold:
            risk_analysis['fade_risk'] = 'HIGH'
            risk_analysis['suggested_actions'].extend([
                "Increase transmitter power by 3 dB",
                "Activate forward error correction",
                "Consider switching to backup ground station"
            ])
        elif predicted_power < fade_threshold + 3:
            risk_analysis['fade_risk'] = 'MEDIUM'
            risk_analysis['suggested_actions'].extend([
                "Prepare to increase transmitter power",
                "Monitor link quality closely"
            ])
        else:
            risk_analysis['fade_risk'] = 'LOW'
            risk_analysis['suggested_actions'].append("Continue normal operation")
        
        return risk_analysis
    
    def plot_prediction_example(self, X_test: np.ndarray, y_test: np.ndarray, n_examples: int = 3):
        """Plot example predictions to visualize model performance."""
        if not self.is_trained:
            print(" Model not trained yet!")
            return
        
        y_pred = self.rf_model.predict(X_test[:n_examples])
        
        # Denormalize for plotting
        y_test_denorm = y_test * self.scaling_factors['y_std'] + self.scaling_factors['y_mean']
        y_pred_denorm = y_pred * self.scaling_factors['y_std'] + self.scaling_factors['y_mean']
        
        fig, axes = plt.subplots(1, n_examples, figsize=(15, 5))
        if n_examples == 1:
            axes = [axes]
        
        for i in range(n_examples):
            input_sequence = X_test[i] * self.scaling_factors['X_std'] + self.scaling_factors['X_mean']
            axes[i].plot(range(1, 11), input_sequence, 'bo-', label='Input Sequence', alpha=0.7)
            axes[i].plot(11, y_test_denorm[i], 'go', markersize=10, label='Actual Future')
            axes[i].plot(11, y_pred_denorm[i], 'ro', markersize=10, label='Predicted Future')
            
            axes[i].set_xlabel('Time Step (2-min intervals)')
            axes[i].set_ylabel('Received Power (dBm)')
            axes[i].set_title(f'Example {i+1}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fade_prediction_examples.png', dpi=150, bbox_inches='tight')
        print(" Saved prediction examples to 'fade_prediction_examples.png'")
        plt.show()

# Test the ML module
if __name__ == "__main__":
    print("=== TESTING ML FADE PREDICTOR (CPU VERSION) ===")
    
    predictor = FadePredictor()
    X, y = predictor.generate_training_data(n_samples=3000)  # Even smaller for quick testing
    mae, r2 = predictor.train_random_forest(X, y)
    
    # Test prediction
    test_sequence = [-68.5, -69.2, -70.1, -71.5, -73.2, -72.8, -71.9, -70.5, -69.8, -69.1]
    prediction = predictor.predict_fade(test_sequence)
    
    print(f"\n EXAMPLE PREDICTION:")
    print(f"   Input sequence: {[f'{x:.1f}' for x in test_sequence]} dBm")
    print(f"   Predicted power (in 2 min): {prediction:.1f} dBm")
    
    risk_analysis = predictor.analyze_fade_risk(test_sequence[-1], prediction)
    print(f"\n RISK ANALYSIS:")
    print(f"   Fade Risk: {risk_analysis['fade_risk']}")
    print(f"   Suggested Actions: {risk_analysis['suggested_actions'][0]}")