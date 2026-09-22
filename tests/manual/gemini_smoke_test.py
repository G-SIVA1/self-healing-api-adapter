from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from self_healing_agent.agent.llm_client import (
    AsyncLLMClient,
    GeminiTransport,
)


async def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured"
        )

    model = os.getenv(
        "LLM_MODEL",
        "gemini-2.5-flash",
    )

    transport = GeminiTransport(
        api_key=api_key,
    )

    client = AsyncLLMClient(
        transport=transport,
        model=model,
        timeout_seconds=30,
        max_retries=2,
        base_delay_seconds=0.5,
    )

    response = await client.generate(
        """
Return exactly this JSON object:

{
  "status": "ok",
  "message": "Gemini connection successful"
}

Do not include markdown fences.
""".strip()
    )

    print("\n=== REAL GEMINI RESPONSE ===")
    print(response)
    print("============================\n")


if __name__ == "__main__":
    asyncio.run(main())