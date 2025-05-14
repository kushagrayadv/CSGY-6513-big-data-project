import asyncio
from preprocessing import DataProducer

async def main():
    producer = DataProducer()
    try:
        # Start the producer
        print("Starting producer - will download fresh dataset from source")
        if await producer.start():
            print("Producer started successfully")
            print("Processing raw data directly from freshly downloaded files...")
            # Send the data
            await producer.send_data()
            print("All data processed and sent successfully")
            print("Temporary data files will be removed automatically")
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