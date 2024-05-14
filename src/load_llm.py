import os
from langchain_openai import OpenAI
from langchain_community.llms import Ollama

# Configurações de ambiente
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def load_llm(model_name, provider=None):
    """
    Carrega um modelo de linguagem grande (LLM) especificado pelo nome e pelo provedor.

    Args:
        model_name (str): Nome do modelo.
        provider (str): Provedor do LLM ('openai', 'ollama'). Default é 'ollama'.

    Returns:
        llm (Ollama or OpenAI): Instância do modelo carregado.
    """

    # Exemplos:
    # chatgpt = load_llm("gpt-3.5-turbo-instruct", provider="openai")
    # llama3 = load_llm("llama3:instruct")
    # phi3 = load_llm("phi3:instruct")
    # gemma = load_llm("gemma:instruct")
    # mistral = load_llm("mistral:instruct")

    if provider == "openai":
        llm = OpenAI(model_name=model_name, openai_api_key="")  # Gere uma key para uso da API
    else:
        llm = Ollama(model=model_name)
    return llm

