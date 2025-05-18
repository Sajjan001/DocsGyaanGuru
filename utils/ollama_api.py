import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

def get_ollama_models():
    try:
        res = requests.get("http://localhost:11434/api/tags")
        if res.status_code == 200:
            return [m["name"] for m in res.json()["models"]]
    except:
        pass
    return ["llama3"]

def query_ollama(prompt, model, context="", system_prompt=""):
    if not system_prompt:
        system_prompt = (
            "You are a helpful assistant answering questions based on the context from documents.\n"
            "If unsure, say 'I don't know'."
        )
    full_prompt = f"""Context:
{context}

Question:
{prompt}
"""

    payload = {
        "model": model,
        "prompt": full_prompt,
        "system": system_prompt,
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_API_URL, json=payload)
        if res.status_code == 200:
            return res.json()["response"]
        else:
            return f"Error: {res.text}"
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"
