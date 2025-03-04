import aiohttp
import asyncio
import time
from statistics import mean, median, stdev
from typing import List, Dict
import json


async def make_request(
    session: aiohttp.ClientSession, payload: Dict
) -> tuple[float, bool]:
    """Make a single request and return the response time and success status"""
    start_time = time.time()
    try:
        async with session.post(
            "http://localhost:5001/v1/safety/run-shield", json=payload
        ) as response:
            await response.json()
            return time.time() - start_time, response.status == 200
    except Exception as e:
        print(f"Request failed: {e}")
        return time.time() - start_time, False


async def run_load_test(
    num_requests: int, concurrency: int
) -> List[tuple[float, bool]]:
    """Run load test with specified number of requests and concurrency"""
    payload = {
        "shield_id": "email_hap",
        "messages": [{"content": "My email is test@example.com", "role": "system"}],
    }

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def controlled_request():
            async with semaphore:
                return await make_request(session, payload)

        tasks = [controlled_request() for _ in range(num_requests)]
        return await asyncio.gather(*tasks)


def print_stats(results: List[tuple[float, bool]], concurrency: int):
    """Print statistics from the load test results"""
    times = [t for t, success in results]
    successes = sum(1 for _, success in results if success)

    print(f"\nLoad Test Results (Concurrency: {concurrency})")
    print("-" * 50)
    print(f"Total Requests: {len(results)}")
    print(f"Successful Requests: {successes}")
    print(f"Failed Requests: {len(results) - successes}")
    print(f"Success Rate: {(successes/len(results))*100:.2f}%")
    print("\nTiming Statistics (seconds):")
    print(f"Mean Response Time: {mean(times):.3f}")
    print(f"Median Response Time: {median(times):.3f}")
    try:
        print(f"Standard Deviation: {stdev(times):.3f}")
    except:
        print("Standard Deviation: N/A")
    print(f"Min Response Time: {min(times):.3f}")
    print(f"Max Response Time: {max(times):.3f}")
    print(f"Total Test Duration: {sum(times):.3f}")
    print(f"Requests per Second: {len(results)/sum(times):.2f}")


# Run the load tests with different concurrency levels
async def main():
    print("Starting Load Tests...")

    # Test configurations
    configurations = [
        {"requests": 100, "concurrency": 1},  # Sequential
        {"requests": 100, "concurrency": 10},  # Medium concurrency
        {"requests": 100, "concurrency": 25},  # High concurrency
    ]

    for config in configurations:
        results = await run_load_test(config["requests"], config["concurrency"])
        print_stats(results, config["concurrency"])


# Run the test
if __name__ == "__main__":
    asyncio.run(main())
