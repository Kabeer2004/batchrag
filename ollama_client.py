import requests
import json
from typing import List, Dict, Any, Optional, Generator

# Configuration (adjust if your Ollama runs elsewhere)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 60 # Timeout in seconds

def get_embedding(text: str, model: str) -> List[float]:
    """Generates an embedding for the given text using Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status() # Raise exception for bad status codes
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Ollama connection error for embeddings: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to get embedding from Ollama model '{model}': {e}") from e


def generate_completion(
    prompt: str,
    model: str,
    system_message: Optional[str] = None,
    stream: bool = True, # Default to streaming
) -> Generator[str, None, None]:
    """Generates a completion for the given prompt using Ollama, optionally streaming."""
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }
        if system_message:
             payload["system"] = system_message # Add system message if provided

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=stream, # Important for streaming
            timeout=OLLAMA_TIMEOUT * 5 # Longer timeout for generation
        )
        response.raise_for_status()

        if stream:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        yield chunk.get("response", "")
                        if chunk.get("done"):
                            break # Stop iteration when Ollama signals completion
                    except json.JSONDecodeError:
                        # Might receive non-JSON data occasionally or incomplete lines
                        # print(f"Warning: Could not decode JSON line: {line}")
                        continue
        else:
            # Non-streaming case (kept for potential future use, but streaming is preferred)
            full_response = response.json()
            yield full_response.get("response", "")

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Ollama connection error for generation: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to generate completion from Ollama model '{model}': {e}") from e

def check_model_availability(model: str) -> bool:
    """Checks if a model is available in Ollama by trying to get its info."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/show",
            json={"name": model},
            timeout=10 # Short timeout for check
        )
        # Model exists if status is 200 OK
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # Connection error likely means Ollama isn't running
        return False
    except Exception:
        # Any other error probably means model doesn't exist or other issue
        return False