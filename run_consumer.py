import asyncio
from kafka_consumer import ActivityClassifierConsumer

async def main():
    consumer = ActivityClassifierConsumer()
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