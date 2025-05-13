import json
import asyncio
import numpy as np
import torch
from aiokafka import AIOKafkaConsumer
from model import LSTMAutoencoder
import os

FREQ_HZ = 50
WINDOW_SIZE = 100
STEP_SIZE = 50

class LSTMConsumer:
    def __init__(self, bootstrap_servers='localhost:9092', topic_name='sensor_data', 
                 model_path='lstmae_best.pt', threshold_path='val_recon_errors.npz'):
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.consumer = None
        
        # Activity mapping
        self.activities = {
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
        
        # Load LSTM model and threshold
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            
            # Initialize model
            self.model = LSTMAutoencoder().to(self.device)
            
            # Load model weights
            if os.path.exists(model_path):
                print(f"Loading model weights from {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            else:
                print(f"Warning: Model weights file {model_path} not found.")
                print("Using untrained model - predictions will be random")
                self.model.eval()
            
            # Load threshold
            if os.path.exists(threshold_path):
                print(f"Loading threshold from {threshold_path}")
                threshold_data = np.load(threshold_path)
                self.threshold = float(threshold_data['threshold']) / 1000.0  # Scale down threshold
                print(f"Using threshold: {self.threshold:.4f}")
            else:
                print(f"Warning: Threshold file {threshold_path} not found.")
                print("Using default threshold of 0.5")
                self.threshold = 0.5
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    async def process_message(self, message):
        try:
            data = message.value
            device_type = data['device']
            features = np.array(data['features'])
            activity = data['activity'].strip()
            subject_id = data['subject_id']
            
            # Convert features to tensor and reshape
            features_tensor = torch.FloatTensor(features).reshape(1, WINDOW_SIZE, -1).to(self.device)
            
            # Get prediction
            is_normal, recon_error = self.model.predict(features_tensor, self.threshold)
            
            # Get activity name
            activity_name = self.activities.get(activity, f"Unknown ({activity})")
            
            # Print results
            print(f"\nSubject: {subject_id}, Device: {device_type}")
            print(f"Activity: {activity_name}")
            print(f"Reconstruction Error: {recon_error:.4f}")
            print(f"Status: {'Normal' if is_normal else 'Anomaly'}")

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
    consumer = LSTMConsumer()
    await consumer.start()
    await consumer.consume()

if __name__ == "__main__":
    asyncio.run(main()) 