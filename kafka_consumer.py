import json
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aiokafka import AIOKafkaConsumer
import os
import matplotlib.pyplot as plt

# Model definition (from classifier_model.py)
class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, input_size=6, cnn_ch=64, lstm_h=128, lstm_layers=2, n_classes=18):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, cnn_ch, 3, padding=1),
            nn.BatchNorm1d(cnn_ch), nn.ReLU(),
            nn.Conv1d(cnn_ch, cnn_ch, 3, padding=1),
            nn.BatchNorm1d(cnn_ch), nn.ReLU()
        )
        self.lstm = nn.LSTM(cnn_ch, lstm_h, lstm_layers,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = nn.Linear(lstm_h*2, 1)
        self.drop = nn.Dropout(0.5)
        self.fc   = nn.Linear(lstm_h*2, n_classes)

    def forward(self, x):                # x: (B,T,F)
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)    # (B,T,C)
        out,_ = self.lstm(x)                               # (B,T,2H)
        w = torch.softmax(self.attn(out).squeeze(-1), 1)   # (B,T)
        ctx = torch.sum(out * w.unsqueeze(-1), 1)          # (B,2H)
        return self.fc(self.drop(ctx))

FREQ_HZ = 50
WINDOW_SIZE = 100
STEP_SIZE = 50

class ActivityClassifierConsumer:
    def __init__(self, bootstrap_servers='localhost:9092', topic_name='sensor_data', 
                 model_path='cnn_bilstm_classifier_final.pt'):
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.consumer = None
        
        # Activity mapping: Letter code -> Full name
        self.activity_map = {
            'A': 'Walking',
            'B': 'Jogging',
            'C': 'Stairs',
            'D': 'Sitting',
            'E': 'Standing',
            'F': 'Typing',
            'G': 'Brushing Teeth',
            'H': 'Eating Soup',
            'I': 'Eating Chips',
            'J': 'Eating Pasta',
            'K': 'Drinking',
            'L': 'Eating Sandwich',
            'M': 'Kicking',
            'O': 'Catch Tennis Ball',
            'P': 'Dribbling',
            'Q': 'Writing',
            'R': 'Clapping',
            'S': 'Folding Clothes'
        }
        
        # Create reverse mapping: Full name -> Letter code
        self.reverse_activity_map = {v: k for k, v in self.activity_map.items()}
        
        # Load classifier model
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            # Load model and metadata
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Get class names from checkpoint
                if "class_names" in checkpoint:
                    self.class_names = checkpoint["class_names"]
                    print(f"Loaded {len(self.class_names)} activity classes")
                    
                    # Map class indices to letters directly
                    self.class_to_letter = {}
                    
                    for i, name in enumerate(self.class_names):
                        letter = None
                        
                        # Check if the class name is exactly a letter (e.g., "A", "B", etc.)
                        if name in self.activity_map:
                            letter = name
                        # Check if the class name is a full activity name (e.g., "Walking")
                        elif name in self.reverse_activity_map:
                            letter = self.reverse_activity_map[name]
                        # Check if any activity name is contained in the class name
                        else:
                            for code, activity in self.activity_map.items():
                                if activity in name or name in activity:
                                    letter = code
                                    break
                        
                        # If we found a letter, use it; otherwise use a placeholder
                        if letter:
                            self.class_to_letter[i] = letter
                        else:
                            self.class_to_letter[i] = f"Class-{i}"
                else:
                    # Use activities as fallback
                    self.class_names = list(self.activity_map.values())
                    self.class_to_letter = {i: k for i, (k, v) in enumerate(self.activity_map.items())}
                    print(f"Using {len(self.class_names)} activity classes from mapping")
        
                # Create model
                n_classes = len(self.class_names)
                self.model = CNNBiLSTMClassifier(n_classes=n_classes).to(self.device)
                
                # Load weights
                if "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    print("Warning: Model checkpoint doesn't contain state_dict")
                    
                self.model.eval()
                print("Model loaded successfully")
            else:
                print(f"Warning: Model file {model_path} not found.")
                print("Creating untrained model - predictions will be random")
                self.class_names = list(self.activity_map.values())
                self.class_to_letter = {i: k for i, (k, v) in enumerate(self.activity_map.items())}
                self.model = CNNBiLSTMClassifier(n_classes=len(self.class_names)).to(self.device)
                self.model.eval()
                
            # Initialize stats
            self.total_messages = 0
            self.correct_predictions = 0
                
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def process_message(self, message):
        try:
            data = message.value
            device_type = data['device']
            features = np.array(data['features'])
            activity_code = data['activity'].strip()
            subject_id = data['subject_id']
            
            # Normalize the features (per sequence) as in the notebook
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True) + 1e-5
            normalized_features = (features - mean) / std
            
            # Convert features to tensor and reshape
            features_tensor = torch.FloatTensor(normalized_features).reshape(1, WINDOW_SIZE, -1).to(self.device)
            
            # Get ground truth activity - both code and name
            ground_truth_name = self.activity_map.get(activity_code, f"Unknown ({activity_code})")
            
            # Forward pass through model for classification
            self.model.eval()
            with torch.no_grad():
                # Get prediction
                logits = self.model(features_tensor)
                probs = F.softmax(logits, dim=1)
                
                # Get predicted class and confidence
                confidence, pred_idx = torch.max(probs, dim=1)
                confidence = confidence.item()
                pred_idx = pred_idx.item()
                
                # Extract the letter code based on the predicted class index
                predicted_letter = self.class_to_letter.get(pred_idx, "?")
                
                # If the predicted_letter still has "Class-" in it,
                # check if the class name has the letter directly in it
                if predicted_letter.startswith("Class-"):
                    cls_name = self.class_names[pred_idx] if pred_idx < len(self.class_names) else "Unknown"
                    
                    # The class name might directly be the letter (e.g., "I" for class 8)
                    if cls_name in self.activity_map:
                        predicted_letter = cls_name
                
                # Get the full activity name based on the letter
                predicted_name = self.activity_map.get(predicted_letter, self.class_names[pred_idx] if pred_idx < len(self.class_names) else "Unknown")
                
                # Check if prediction is correct using the extracted letter
                is_correct = predicted_letter == activity_code
                if is_correct:
                    self.correct_predictions += 1
            
            self.total_messages += 1
            accuracy = self.correct_predictions / self.total_messages if self.total_messages > 0 else 0
            
            # Print results
            print(f"\n========================================")
            print(f"Subject: {subject_id}, Device: {device_type}")
            print(f"Predicted → {predicted_letter} ({predicted_name}) (conf: {confidence:.2f})")
            print(f"Ground Truth: {activity_code} ({ground_truth_name})")
            print(f"Correct: {'✓' if is_correct else '✗'}")
            print(f"Accuracy: {accuracy:.2%} ({self.correct_predictions}/{self.total_messages})")
            print(f"========================================")

        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()

    async def start(self):
        try:
            self.consumer = AIOKafkaConsumer(
                self.topic_name,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            await self.consumer.start()
            print(f"Consumer started. Listening on topic: {self.topic_name}")
        except Exception as e:
            print(f"Error starting consumer: {e}")
            raise

    async def consume(self):
        try:
            print("Waiting for messages...")
            async for message in self.consumer:
                await self.process_message(message)
        except Exception as e:
            print(f"Error in consume: {e}")
            raise

async def main():
    consumer = ActivityClassifierConsumer(topic_name='wisdm_predictions')
    try:
        print("Starting activity classifier consumer...")
        await consumer.start()
        print("Consumer started successfully. Waiting for messages...")
        await consumer.consume()
    except KeyboardInterrupt:
        print("\nShutting down consumer...")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(consumer, 'consumer') and consumer.consumer:
            await consumer.consumer.stop()
            print("Consumer stopped")

if __name__ == "__main__":
    asyncio.run(main()) 