import asyncio
from preprocessing import DataProducer

async def main():
    producer = DataProducer()
    try:
        # Start the producer
        if await producer.start():
            print("Producer started successfully")
            # Send the data
            await producer.send_data()
            print("All data sent successfully")
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Ensure producer is stopped
        if producer.producer:
            await producer.producer.stop()
            print("Producer stopped")
        if producer.spark:
            producer.spark.stop()
            print("Spark session stopped")

if __name__ == "__main__":
    asyncio.run(main()) 