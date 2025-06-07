import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size=4, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

def generate_synthetic_data(n_samples=1000, n_features=4, noise=10.0, random_state=42):
    """Generate synthetic regression data"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
        bias=50.0,
        n_informative=4,  # All features are informative
        effective_rank=None,  # Use all features
        tail_strength=0.5
    )
    
    # Normalize features
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    # Normalize target to reasonable range
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X, y, scaler_X, scaler_y

def train_model(model, train_loader, val_loader, epochs=200, lr=0.001):
    """Train the linear regression model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 30
    
    print(f"🏋️  Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 25 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch+1:3d}/{epochs}: Train = {avg_train_loss:.4f}, Val = {avg_val_loss:.4f}, LR = {current_lr:.6f}")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"   Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
            break
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    """Evaluate the trained model"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(batch_y.tolist())
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions.tolist(),
        'actuals': actuals.tolist()
    }

def save_training_artifacts(model, scaler_X, scaler_y, metrics, train_losses, val_losses):
    """Save model, scalers, and training metadata"""
    artifacts_dir = "../artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(artifacts_dir, "trained_model.pth"))
    
    # Save scalers
    import pickle
    with open(os.path.join(artifacts_dir, "scaler_X.pkl"), "wb") as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(artifacts_dir, "scaler_y.pkl"), "wb") as f:
        pickle.dump(scaler_y, f)
    
    # Save training metadata
    metadata = {
        'model_architecture': {
            'input_size': model.linear.in_features,
            'output_size': model.linear.out_features,
            'weight_shape': list(model.linear.weight.shape),
            'bias_shape': list(model.linear.bias.shape)
        },
        'training_metrics': metrics,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }
    }
    
    with open(os.path.join(artifacts_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n💾 Training artifacts saved to: {os.path.abspath(artifacts_dir)}")

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(train_losses, label='Training Loss', color='blue')
    plt.semilogy(val_losses, label='Validation Loss', color='red')
    plt.title('Training History (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    artifacts_dir = "../artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    plt.savefig(os.path.join(artifacts_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
    print("📊 Training curves saved to artifacts/training_curves.png")
    
    # Don't show plot in headless environments
    try:
        plt.show()
    except:
        pass

def main():
    print("🎯 Linear Regression Training Pipeline")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data with better parameters
    print("📊 Generating synthetic regression data...")
    X, y, scaler_X, scaler_y = generate_synthetic_data(
        n_samples=2000, 
        n_features=4, 
        noise=5.0  # Reduced noise for better performance
    )
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"   📈 Training set: {X_train.shape[0]} samples")
    print(f"   📊 Validation set: {X_val.shape[0]} samples")
    print(f"   🧪 Test set: {X_test.shape[0]} samples")
    print(f"   📏 Feature range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   🎯 Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = LinearRegressionModel(input_size=4, output_size=1)
    print(f"\n🧠 Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model with better hyperparameters
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=300, lr=0.001
    )
    
    # Evaluate model
    print("\n🔍 Evaluating trained model...")
    metrics = evaluate_model(model, test_loader)
    
    print(f"   📊 Test Results:")
    print(f"      MSE:  {metrics['mse']:.4f}")
    print(f"      RMSE: {metrics['rmse']:.4f}")
    print(f"      MAE:  {metrics['mae']:.4f}")
    print(f"      R²:   {metrics['r2']:.4f}")
    
    # Check if model performance is good
    if metrics['r2'] > 0.8:
        print("   ✅ Excellent performance! (R² > 0.8)")
    elif metrics['r2'] > 0.6:
        print("   👍 Good performance! (R² > 0.6)")
    elif metrics['r2'] > 0.3:
        print("   📈 Moderate performance (R² > 0.3)")
    else:
        print("   ⚠️  Poor performance - consider tuning hyperparameters")
    
    # Save artifacts
    save_training_artifacts(model, scaler_X, scaler_y, metrics, train_losses, val_losses)
    
    # Plot training curves
    try:
        plot_training_curves(train_losses, val_losses)
    except ImportError:
        print("⚠️  matplotlib not available, skipping training curve plots")
    
    # Show model weights
    print(f"\n🎯 Trained Model Weights:")
    print(f"   Weight matrix: {model.linear.weight.data.numpy()}")
    print(f"   Bias: {model.linear.bias.data.numpy()}")
    
    # Example prediction with denormalization
    print(f"\n🧪 Example Prediction:")
    model.eval()
    sample_input = torch.randn(1, 4)
    with torch.no_grad():
        prediction_normalized = model(sample_input)
        # Denormalize prediction
        prediction_real = scaler_y.inverse_transform(prediction_normalized.numpy())[0, 0]
    
    print(f"   Input (normalized): {sample_input.numpy()[0]}")
    print(f"   Predicted output (normalized): {prediction_normalized.item():.4f}")
    print(f"   Predicted output (real scale): {prediction_real:.4f}")
    
    print(f"\n✅ Training complete! Model ready for export.")
    print(f"📁 Run 'python export_trained_model.py' to export for inference.")

if __name__ == "__main__":
    main()