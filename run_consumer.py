import asyncio
from kafka_consumer import LSTMConsumer

async def main():
    consumer = LSTMConsumer()
    await consumer.start()
    try:
        await consumer.consume()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await consumer.consumer.stop()

if __name__ == "__main__":
    asyncio.run(main()) 