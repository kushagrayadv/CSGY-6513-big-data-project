# import asyncio
# from preprocessing import DataProducer

# async def main():
#     producer = DataProducer()
#     try:
#         # Start the producer
#         print("Starting producer - will download fresh dataset from source")
#         if await producer.start():
#             print("Producer started successfully")
#             print("Processing raw data directly from freshly downloaded files...")
#             # Send the data
#             await producer.send_data()
#             print("All data processed and sent successfully")
#             print("Temporary data files will be removed automatically")
#     except KeyboardInterrupt:
#         print("\nShutting down...")
#     except Exception as e:
#         print(f"Error: {str(e)}")
#     finally:
#         # Ensure producer is stopped
#         if producer.producer:
#             await producer.producer.stop()
#             print("Producer stopped")
#         if producer.spark:
#             producer.spark.stop()
#             print("Spark session stopped")

# if __name__ == "__main__":
#     asyncio.run(main()) 


# run_producer.py
import asyncio, json, traceback
from aiokafka import AIOKafkaProducer
from preprocessing import DataProducer   # your Spark-based pipeline

BROKER    = "localhost:9092"
RAW_TOPIC = "wisdm_predictions"                  # topic that holds raw / windowed data

async def main():
    """
    1. Build DataProducer (downloads & preprocesses WISDM).
    2. Replace its internal producer with one that serialises dict→JSON bytes.
    3. Call send_data() to push everything into `wisdm_raw`.
    """
    producer = DataProducer()

    try:
        print("Starting producer – downloading fresh dataset from source …")
        if await producer.start():                       # spins up Spark etc.
            print("Producer infrastructure ready.")

            # ─── Ensure Kafka producer writes JSON bytes ──────────────────
            if hasattr(producer, "producer") and producer.producer:
                try:
                    await producer.producer.stop()
                except Exception:
                    pass

            producer.producer = AIOKafkaProducer(
                bootstrap_servers=BROKER,
                value_serializer=lambda obj: json.dumps(obj).encode()
            )
            await producer.producer.start()
            print(f"Producer connected → will write to topic “{RAW_TOPIC}”")

            # ─── Now push data ------------------------------------------------
            print("Processing raw data directly from freshly downloaded files …")
            await producer.send_data()      # must internally call:
                                            # await self.producer.send_and_wait(RAW_TOPIC, payload_dict)
            print("All data processed and sent successfully.")
            print("Temporary data files will be removed automatically.")

    except KeyboardInterrupt:
        print("\nShutting down producer …")
    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()
    finally:
        # Clean shutdown
        if hasattr(producer, "producer") and producer.producer:
            await producer.producer.stop()
            print("Kafka producer stopped.")
        if hasattr(producer, "spark") and producer.spark:
            producer.spark.stop()
            print("Spark session stopped.")

if __name__ == "__main__":
    asyncio.run(main())

