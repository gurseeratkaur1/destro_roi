import pandas as pd
from enhanced_dataset import EnhancedConversionDataGenerator
from persistent_predictor import PersistentConversionPredictor

def train_and_save_model():
    """Train a new model and save it for future use"""
    
    print("=" * 60)
    print("TRAINING AND SAVING PERSISTENT MODEL")
    print("=" * 60)
    
    # Step 1: Generate training data
    print("\n1. Generating training data...")
    generator = EnhancedConversionDataGenerator(num_users=1000)
    datasets = generator.generate_complete_dataset()
    
    # Step 2: Initialize predictor
    print("\n2. Initializing predictor...")
    predictor = PersistentConversionPredictor(model_name="conversion_predictor_v1")
    
    # Step 3: Prepare data
    print("\n3. Preparing features...")
    user_features = predictor.load_and_prepare_data(
        datasets['interactions'],
        datasets['consent'],
        datasets['communications'],
        datasets['conversions']
    )
    
    # Step 4: Train model
    print("\n4. Training model...")
    X, y = predictor.prepare_features(user_features)
    results = predictor.train_model(X, y)
    
    # Step 5: Save the trained model
    print("\n5. Saving model...")
    predictor.save_model()
    
    # Step 6: Test the saved model by loading it
    print("\n6. Testing saved model...")
    test_predictor = PersistentConversionPredictor(model_name="conversion_predictor_v1")
    
    if test_predictor.load_model():
        print("✅ Model save/load test successful!")
        
        # Show model info
        print(f"\nModel Information:")
        for key, value in test_predictor.get_model_info().items():
            if key != 'feature_names':  # Skip long feature list
                print(f"  • {key}: {value}")
    else:
        print("❌ Model save/load test failed!")
    
    print(f"\n✅ Training complete! Model saved and ready for use.")
    
    return predictor, user_features

if __name__ == "__main__":
    train_and_save_model()
