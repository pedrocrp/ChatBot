import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"


def load_chatgpt():
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key="") # Gere uma key para uso da API 
    return llm


def load_llama3():
    llm = Ollama(model="llama3:instruct")
    return llm


def load_phi3():
    llm = Ollama(model="phi3:instruct")
    return llm


def load_gemma():
    llm = Ollama(model="gemma:instruct")
    return llm


def load_gemma():
    llm = Ollama(model="mistral:instruct")
    return llm
