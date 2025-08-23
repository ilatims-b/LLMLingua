# Updated utils.py for Groq Support

```python
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from time import sleep
import os
import json
from datetime import datetime
import tiktoken

# Import both OpenAI and Groq clients
try:
    import openai
except ImportError:
    print("OpenAI not installed. Install with: pip install openai")

try:
    from groq import Groq
except ImportError:
    print("Groq not installed. Install with: pip install groq")

MAX_BUDGET = 2.00  # $2.00 limit
BUDGET_FILE = "usage_tracker.json"


def query_llm(
    prompt,
    model,
    model_name,
    max_tokens,
    tokenizer=None,
    chat_completion=False,
    use_groq=False,
    **kwargs,
):
    SLEEP_TIME_FAILED = 62

    request = {
        "temperature": kwargs["temperature"] if "temperature" in kwargs else 0.0,
        "top_p": kwargs["top_p"] if "top_p" in kwargs else 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    
    if chat_completion:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        request["messages"] = messages
    else:
        request["prompt"] = prompt

    # Add seed for non-Groq models (Groq doesn't support seed parameter)
    if not use_groq:
        request["seed"] = kwargs["seed"] if "seed" in kwargs else 42
        request["n"] = 1

    answer = None
    response = None
    while answer is None:
        try:
            if use_groq:
                if chat_completion:
                    response = model.chat.completions.create(
                        model=model_name,
                        messages=request["messages"],
                        temperature=request["temperature"],
                        top_p=request["top_p"],
                        max_tokens=request["max_tokens"],
                        stream=request["stream"]
                    )
                    answer = response.choices[0].message.content
                else:
                    # Groq doesn't support non-chat completions for Llama models
                    # Convert to chat format
                    response = model.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": request["prompt"]}],
                        temperature=request["temperature"],
                        top_p=request["top_p"],
                        max_tokens=request["max_tokens"],
                        stream=request["stream"]
                    )
                    answer = response.choices[0].message.content
            else:
                # OpenAI API
                response = model.create(model=model_name, **request)
                answer = (
                    response["choices"][0]["message"]["content"]
                    if chat_completion
                    else response["choices"][0]["text"]
                )
        except Exception as e:
            answer = None
            print(f"error: {e}, response: {response}")
            sleep(SLEEP_TIME_FAILED)
    return answer


def load_model_and_tokenizer(model_name_or_path, chat_completion=False, use_groq=False):
    """
    Load model and tokenizer based on the API choice (OpenAI vs Groq)
    
    Args:
        model_name_or_path: Model name/path
        chat_completion: Whether to use chat completion format
        use_groq: Whether to use Groq API instead of OpenAI
    
    Returns:
        model: API client object
        tokenizer: Tokenizer for token counting
    """
    
    if use_groq:
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        model = Groq(api_key=api_key)
        
        # Use GPT-4 tokenizer for token counting (close approximation for Llama)
        # You could also use tiktoken.encoding_for_model("gpt-3.5-turbo") for different behavior
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        
    else:
        # Initialize OpenAI client (original behavior)
        openai.api_key = os.getenv("OPENAI_API_KEY", "your_api_key")
        
        if chat_completion:
            model = openai.ChatCompletion
        else:
            model = openai.Completion

        tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    return model, tokenizer


def estimate_groq_tokens(text, tokenizer):
    """
    Estimate token count for Groq models using GPT-4 tokenizer as approximation
    Note: This is an approximation since Groq uses different tokenization
    """
    return len(tokenizer.encode(text))


def validate_groq_model(model_name):
    """
    Validate that the model name is supported by Groq
    """
    supported_models = [
        "llama-3.3-70b-versatile",
        "llama-3.2-90b-text-preview", 
        "llama-3.2-11b-text-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-1b-preview",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "gemma2-9b-it"
    ]
    
    if model_name not in supported_models:
        print(f"Warning: Model {model_name} may not be supported by Groq.")
        print(f"Supported models include: {', '.join(supported_models)}")
    
    return model_name in supported_models


# Usage tracking functions (optional - for budget management)
def track_usage(model_name, input_tokens, output_tokens, use_groq=False):
    """Track API usage for budget management"""
    try:
        if os.path.exists(BUDGET_FILE):
            with open(BUDGET_FILE, 'r') as f:
                usage_data = json.load(f)
        else:
            usage_data = {"total_cost": 0.0, "sessions": []}
        
        # Rough cost estimates (update with current pricing)
        if use_groq:
            # Groq pricing for Llama-3.3-70b-versatile (example rates)
            input_cost_per_1k = 0.00059  # $0.59 per 1M input tokens
            output_cost_per_1k = 0.00079  # $0.79 per 1M output tokens
        else:
            # OpenAI GPT-4 pricing (example rates)
            input_cost_per_1k = 0.03  # $30 per 1M tokens
            output_cost_per_1k = 0.06  # $60 per 1M tokens
        
        session_cost = (input_tokens * input_cost_per_1k / 1000) + (output_tokens * output_cost_per_1k / 1000)
        usage_data["total_cost"] += session_cost
        
        usage_data["sessions"].append({
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": session_cost,
            "provider": "groq" if use_groq else "openai"
        })
        
        with open(BUDGET_FILE, 'w') as f:
            json.dump(usage_data, f, indent=2)
            
        if usage_data["total_cost"] > MAX_BUDGET:
            print(f"Warning: Budget exceeded! Total cost: ${usage_data['total_cost']:.4f}")
            
    except Exception as e:
        print(f"Error tracking usage: {e}")
