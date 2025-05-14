# import asyncio
# from kafka_consumer import ActivityClassifierConsumer

# async def main():
#     consumer = ActivityClassifierConsumer()
#     try:
#         print("Starting activity classifier consumer...")
#         await consumer.start()
#         print("Consumer started successfully. Waiting for messages...")
#         await consumer.consume()
#     except KeyboardInterrupt:
#         print("\nShutting down consumer...")
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         if hasattr(consumer, 'consumer') and consumer.consumer:
#             await consumer.consumer.stop()
#             print("Consumer stopped")

# if __name__ == "__main__":
#     asyncio.run(main()) 

# run_consumer.py
import asyncio, json, traceback
from aiokafka import AIOKafkaProducer
from kafka_consumer import ActivityClassifierConsumer   # your existing class

BROKER = "localhost:9092"        # must match producer, dashboard, and broker
PRED_TOPIC = "wisdm_predictions" # ensure this spelling is the same everywhere

async def main():
    """
    1. Start ActivityClassifierConsumer (reads wisdm_raw, makes predictions)
    2. Replace / wrap its producer so every prediction is JSON-encoded bytes
       → avoids the 'no data on Streamlit' issue.
    """
    consumer = ActivityClassifierConsumer()

    try:
        print("Starting activity classifier consumer …")
        await consumer.start()            # initialises internal consumer/producer

        # ─── Ensure the producer serialises as JSON UTF-8 bytes ───────────
        if hasattr(consumer, "producer") and consumer.producer:
            # Stop the old producer (it may not have a serializer)
            try:
                await consumer.producer.stop()
            except Exception:
                pass

        consumer.producer = AIOKafkaProducer(
            bootstrap_servers=BROKER,
            value_serializer=lambda obj: json.dumps(obj).encode()
        )
        await consumer.producer.start()
        print(f"Producer ready → will publish to topic “{PRED_TOPIC}”")

        print("Consumer started successfully. Waiting for messages …")
        await consumer.consume()          # your existing method: loops forever

    except KeyboardInterrupt:
        print("\nShutting down consumer …")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if hasattr(consumer, "consumer") and consumer.consumer:
            await consumer.consumer.stop()
        if hasattr(consumer, "producer") and consumer.producer:
            await consumer.producer.stop()
        print("Consumer stopped gracefully.")

if __name__ == "__main__":
    asyncio.run(main())

