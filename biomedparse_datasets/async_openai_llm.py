import asyncio
import os
from click import prompt
from openai import AsyncAzureOpenAI
import time

# 1. Setup Client
client = AsyncAzureOpenAI(
    azure_endpoint="https://medevalkit.openai.azure.com/",
    api_key=os.environ["AZURE_API_KEY"],
    api_version="2024-12-01-preview"
)

# 2. Limit concurrency (e.g., only 5 requests at a time)
# This prevents you from getting '429 Rate Limit' errors immediately
sem = asyncio.Semaphore(128) ## 64 is 0.14, 128 is 0.09, 256 was sometimes not possible, 512 is not possible. 

async def run_session(session_id, prompt):
    async with sem:  # This waits here if 5 requests are already running
        print(f"Starting session {session_id}...")
        try:
            response = await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                # max_completion_tokens=,
            )
            result = response.choices[0].message.content
            print(f"Finished session {session_id}.")
            return f"Session {session_id} Result: {result[:50]}..."
        except Exception as e:
            return f"Session {session_id} failed: {e}"

async def main():
    # 3. Create a list of tasks
    _prompts = [
        "What is the capital of France?",
        "Write a 3-line poem about AI.",
        "Explain quantum physics to a cat.",
        "How do I boil an egg?",
        "What is 2+2?"
    ]

    prompts = _prompts * 50  # Create more prompts for testing
    
    tasks = [run_session(i, p) for i, p in enumerate(prompts)]
    
    # 4. Run them all in parallel
    overall_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    overall_end = time.perf_counter()
    print(f"Total time for all sessions: {overall_end - overall_start:.2f} seconds")
    print(f"average time per session: {(overall_end - overall_start) / len(prompts):.2f} seconds")
    
    print("\n--- All Results ---")
    for r in results:
        print(r)

if __name__ == "__main__":
    asyncio.run(main())