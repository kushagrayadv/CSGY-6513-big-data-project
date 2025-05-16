# WISDM Activity Recognition System

A Kafka-based streaming system for real-time human activity recognition using sensor data.

## Setup

1. Create and activate virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure Kafka is running locally:
   ```
   # Example commands for running Kafka with Docker
   docker run -d --name zookeeper -p 2181:2181 wurstmeister/zookeeper
   docker run -d --name kafka -p 9092:9092 --link zookeeper -e KAFKA_ADVERTISED_HOST_NAME=localhost -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 wurstmeister/kafka
   ```

## Running the System

1. **Consumer and Streamlit Integration** - Start in a terminal (make sure to follow the steps of using a Virtual Environment (venv) if you used it for previous tasks):
   ```
   streamlit run app.py
   ```

2. **Producer** - Start in another terminal (make sure to follow the steps of using a Virtual Environment (venv) if you used it for previous tasks):
   ```
   python producer.py
   ```

## Key Files

- `producer.py`: Standalone Kafka producer that downloads the WISDM dataset, processes it and sends data to Kafka
- `app.py`: Consumes sensor data from Kafka and runs real-time activity classification and presents it on a Real-Time Web-Page
- `model.py`: Contains model definitions used by the project
- `data_pipeline_full_fixed_100.ipynb` : Contains the training of the CNN-BiLSTM Model
- `requirements.txt`: Project dependencies
- `BIG DATA PROJECT REPORT.pdf` is the final report of our project
