#!/usr/bin/env python3
import os
import requests
import torch
from classifier_model import CNNBiLSTMClassifier

def verify_model(model_path):
    """Verify that the model file is valid and contains necessary data"""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist")
        return False
    
    try:
        print(f"Verifying model file: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Check for class names
        if "class_names" in checkpoint:
            class_names = checkpoint["class_names"]
            print(f"✓ Model contains {len(class_names)} activity classes")
            print(f"  Classes: {', '.join(class_names)}")
        else:
            print("✗ Model missing 'class_names' attribute")
        
        # Check for state_dict
        if "state_dict" in checkpoint:
            print(f"✓ Model contains state dictionary with {len(checkpoint['state_dict'])} weights")
        else:
            print("✗ Model missing 'state_dict' attribute")
            
        # Try loading the weights into the model
        n_classes = len(checkpoint.get("class_names", []))
        if n_classes > 0:
            model = CNNBiLSTMClassifier(n_classes=n_classes)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            print("✓ Successfully loaded weights into model")
            
            # Do a test forward pass
            dummy_input = torch.randn(1, 100, 6)  # Batch=1, seq_len=100, features=6
            with torch.no_grad():
                output = model(dummy_input)
            print(f"✓ Forward pass successful, output shape: {list(output.shape)}")
            
            return True
        else:
            print("✗ Could not determine number of classes")
            return False
            
    except Exception as e:
        print(f"Error verifying model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_dummy_model(model_path):
    """Create a dummy model with random weights if existing model is invalid"""
    print("Creating dummy model with random weights...")
    
    # Define activities list (should match what's in the kafka_consumer.py)
    activities = [
        'Walking', 'Jogging', 'Stairs', 'Sitting', 'Standing', 'Typing',
        'Brushing Teeth', 'Eating Soup', 'Eating Chips', 'Eating Pasta',
        'Drinking', 'Eating Sandwich', 'Kicking', 'Catch Tennis Ball',
        'Dribbling', 'Writing', 'Clapping', 'Folding Clothes'
    ]
    
    # Create model
    model = CNNBiLSTMClassifier(n_classes=len(activities))
    
    # Create checkpoint
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_names': activities
    }
    
    # Save model
    torch.save(checkpoint, model_path)
    print(f"Dummy model created and saved to {model_path}")

def main():
    # Set the local path to match the consumer's expected path
    local_path = "cnn_bilstm_classifier_final.pt"
    
    # Check if model exists
    if os.path.exists(local_path):
        print(f"\nFound existing model: {local_path} (size: {os.path.getsize(local_path) / 1024:.1f} KB)")
        
        # Verify that the model is valid
        if verify_model(local_path):
            print("\n✅ Model is valid and ready to use with the Kafka consumer!")
        else:
            print("\n⚠️ Model file exists but has problems. Creating a dummy model instead.")
            create_dummy_model(local_path + ".dummy")
            print(f"\n⚠️ Dummy model created. To use it, rename it from {local_path}.dummy to {local_path}")
    else:
        print(f"\n⚠️ Model file {local_path} not found")
        create_dummy_model(local_path)
        print("\n✅ Created dummy model ready for use with the Kafka consumer")

if __name__ == "__main__":
    main() 