"""
Main training pipeline for customer churn prediction.

This script orchestrates the entire ML training workflow:
1. Load and preprocess data
2. Train multiple models
3. Evaluate and compare models
4. Save the best performing model

Usage:
    python main.py
"""
from preprocess import preprocess_data
from train_models import train_models
from evaluate_models import evaluate
from save_model import save_best_model


def main() -> None:
    """
    Execute the complete training pipeline.
    """
    print("=" * 50)
    print("Customer Churn Prediction - Training Pipeline")
    print("=" * 50)
    
    # Step 1: Preprocess data
    print("\n[1/4] Loading and preprocessing data...")
    (X_train, X_test, y_train, y_test), scaler = preprocess_data("data/telco_churn.csv")
    print(f"      Training samples: {len(X_train)}")
    print(f"      Test samples: {len(X_test)}")

    # Step 2: Train models
    print("\n[2/4] Training models...")
    models = train_models(X_train, y_train)
    print(f"      Trained {len(models)} models")

    # Step 3: Evaluate models
    print("\n[3/4] Evaluating models...")
    results = evaluate(models, X_test, y_test)
    
    print("\n" + "-" * 50)
    print("Model Performance Comparison")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    # Step 4: Save best model (Random Forest)
    print("\n[4/4] Saving best model (Random Forest)...")
    best_model = models["Random Forest"]
    save_best_model(best_model, scaler)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
