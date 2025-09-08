import os
import multiprocessing as mp
import time
from typing import Annotated

from marker.services.openai import OpenAIService


class SimpleRateLimiter:
    """
    Multiprocess-safe rate limiter for API calls.

    Uses multiprocessing.Value for shared state across processes
    to ensure rate limiting works correctly when marker runs multiple workers.
    """
    def __init__(self, max_per_minute=40):
        self.max = max_per_minute
        self.count = mp.Value('i', 0)
        self.reset_time = mp.Value('d', time.time() + 60)
        self.lock = mp.Lock()

    def wait(self):
        """Wait until allowed to proceed"""
        while True:
            with self.lock:
                now = time.time()

                # Reset counter every minute
                if now >= self.reset_time.value:
                    self.count.value = 0
                    self.reset_time.value = now + 60

                # Check if under limit
                if self.count.value < self.max:
                    self.count.value += 1
                    return

            time.sleep(0.1)  # Wait a bit and try again


class NvidiaService(OpenAIService):
    """
    NVIDIA NIM (NVIDIA Inference Microservice) service with rate limiting.

    Provides access to NVIDIA's hosted AI models through their OpenAI-compatible API.
    Supports both text and multimodal (vision) models with automatic rate limiting
    to stay within API limits (40 requests/minute by default).

    Popular NVIDIA models:
    - meta/llama-4-maverick-17b-128e-instruct (default)
    - nvidia/llama-3-chatqa-1.5-8b
    - microsoft/phi-3-medium-128k-instruct
    - meta/llama-3.1-8b-instruct
    - nvidia/nemotron-4-340b-instruct

    Usage:
        # Via command line
        marker_single file.pdf --use_llm --llm_service=marker.services.nim.NvidiaService

        # Via Python
        from marker.services.nim import NvidiaService
        service = NvidiaService(config={
            "openai_model": "meta/llama-3.1-8b-instruct",
            "rate_limit_per_minute": 40
        })

    Environment Variables:
        NVIDIA_API_KEY: Required API key from NVIDIA (https://build.nvidia.com/)

    Rate Limiting:
        Automatically limits requests to 40 per minute (configurable) across all
        marker processes using multiprocess-safe synchronization.
    """

    nim_api_key: Annotated[
        str, "NVIDIA_API_KEY environment variable for NVIDIA NIM service."
    ] = os.getenv('NVIDIA_API_KEY')

    openai_base_url: Annotated[
        str, "NVIDIA NIM API endpoint (OpenAI-compatible)."
    ] = "https://integrate.api.nvidia.com/v1"

    openai_model: Annotated[
        str, "NVIDIA model to use for inference. See class docstring for popular options."
    ] = "meta/llama-4-maverick-17b-128e-instruct"

    rate_limit_per_minute: Annotated[
        int, "Maximum requests per minute to NVIDIA API (default: 40)."
    ] = 40

    def __init__(self, config=None):
        """
        Initialize NVIDIA NIM service with rate limiting.

        Maps nim_api_key to openai_api_key for OpenAI service compatibility.
        This allows the inherited OpenAI service methods to work with NVIDIA's API.
        Rate limiter ensures we stay within NVIDIA's API limits across all processes.
        """
        super().__init__(config)
        if self.nim_api_key:
            self.openai_api_key = self.nim_api_key

        # Initialize rate limiter
        self.rate_limiter = SimpleRateLimiter(max_per_minute=self.rate_limit_per_minute)

    def __call__(self, prompt, image, block, response_schema, max_retries=None, timeout=None):
        """
        Make API call with rate limiting.

        Applies rate limiting before calling parent OpenAI service method.
        This ensures we respect NVIDIA's API rate limits across all marker processes.
        """
        # Apply rate limiting before making the API call
        self.rate_limiter.wait()

        # Call parent method with rate limiting applied
        return super().__call__(prompt, image, block, response_schema, max_retries, timeout)