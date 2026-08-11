"""
Editable model registry for contrastive QA API evaluation.

Model IDs are intentionally easy to change. Keep API keys in .env, not here.
"""

MODEL_CONFIGS = {
    "gemini_2_5_flash": {
        "display_name": "Gemini 2.5 Flash",
        "provider": "openai_compatible",
        "model_id": "google/gemini-2.5-flash",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "enabled": True,
    },
    "gemini_2_5_pro": {
        "display_name": "Gemini 2.5 Pro",
        "provider": "openai_compatible",
        "model_id": "google/gemini-2.5-pro",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "enabled": True,
    },
    "llama_3_3_70b": {
        "display_name": "Llama 3.3 70B",
        "provider": "openai_compatible",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "enabled": True,
    },
    "llama_3_1_8b": {
        "display_name": "Llama 3.1 8B",
        "provider": "openai_compatible",
        "model_id": "meta-llama/llama-3.1-8b-instruct",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "enabled": True,
    },
    "deepseek_r1_distill": {
        "display_name": "DeepSeek-R1 Distill",
        "provider": "openai_compatible",
        "model_id": "deepseek/deepseek-r1-distill-llama-70b",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "enabled": False,
    },
    "qwen3_32b": {
        "display_name": "Qwen3 32B",
        "provider": "openai_compatible",
        "model_id": "qwen/qwen3-32b",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "enabled": True,
    },
    "mixtral_8x22b": {
        "display_name": "Mixtral 8x22B",
        "provider": "openai_compatible",
        "model_id": "mistralai/mixtral-8x22b-instruct",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "enabled": True,
    },
    "gemma_2_27b": {
        "display_name": "Gemma 2 27B",
        "provider": "openai_compatible",
        "model_id": "google/gemma-2-27b-it",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "enabled": True,
    },
}


