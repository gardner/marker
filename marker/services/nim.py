import os
from typing import Annotated

from marker.services.openai import OpenAIService


class NvidiaService(OpenAIService):
    """
    NVIDIA NIM (NVIDIA Inference Microservice) service.
    
    Provides access to NVIDIA's hosted AI models through their OpenAI-compatible API.
    Supports both text and multimodal (vision) models.
    
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
        service = NvidiaService(config={"openai_model": "meta/llama-3.1-8b-instruct"})
        
    Environment Variables:
        NVIDIA_API_KEY: Required API key from NVIDIA (https://build.nvidia.com/)
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
    
    def __init__(self, config=None):
        """
        Initialize NVIDIA NIM service.
        
        AIDEV-NOTE: Maps nim_api_key to openai_api_key for OpenAI service compatibility.
        This allows the inherited OpenAI service methods to work with NVIDIA's API.
        """
        super().__init__(config)
        if self.nim_api_key:
            self.openai_api_key = self.nim_api_key