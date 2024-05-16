import argparse
from typing import Optional
from pathlib import Path
from pydantic import BaseModel
from pydantic import Field

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Path as PathParam, Query, File, UploadFile, HTTPException, status


from load_data import load_and_process_file, initialize_splitter
from load_llm import load_llm
from prompts import create_prompt
from utils import read_file
from VectorStore import VectorDB
from langchain_community.embeddings import HuggingFaceHubEmbeddings, OllamaEmbeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ml_models = {}
db_name = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    llm = load_llm("llama3-chatqa")
    ml_models["answer_to_query"] = llm
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(
    title="UnB RAG APP",
    description='APP de "Retrival Augmented Generation" que permite ao usuário fazer upload de um arquivo e obter a resposta para a pergunta usando LLMs',
    lifespan=lifespan
)

@app.get("/")
def index():
    return {"message": "Por favor vá para /docs"}



# the model initialized when the app gets loaded but we can configure it if we want
@app.get("/init_llm")
def init_llama_llm(n_gpu_layers: int = Query(500, description="Number of layers to load in GPU"),
                n_batch: int = Query(32, description="Number of tokens to process in parallel. Should be a number between 1 and n_ctx."),
                max_tokens: int = Query(300, description="The maximum number of tokens to generate."),
                n_ctx: int = Query(4096, description="Token context window."),
                temperature: int = Query(0, description="Temperature for sampling. Higher values means more random samples.")):
    model_path = model_args["model_path"]
    model_args = {'model_path' : model_path,
                  'n_gpu_layers': n_gpu_layers,
                  'n_batch': n_batch,
                  'max_tokens': max_tokens,
                  'n_ctx': n_ctx,
                  'temperature': temperature,
                  'device': device}
    llm = load_llm("phi3:instruct")
    ml_models["answer_to_query"] = llm
    return {"message": "LLM initialized"}


@app.post("/upload_document", status_code=status.HTTP_201_CREATED)
def upload_file(file: UploadFile = File(...), collection_name: Optional[str] = "general_collection"):
    data_path = Path('../data')
    file_path = data_path / file.filename

    try:
        # Save uploaded file
        contents = file.file.read()
        file_path.write_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to save the file: {e}")

    try:
        # Load and process the file based on its type
        data = load_and_process_file(file_path)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"Failed to process the file: {e}")

    try:
        # Integrate with VectorDB
        vector_db = VectorDB()
        vector_db.add_documents(collection_name, data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to integrate with VectorDB: {e}")

    return {"message": "File uploaded and processed successfully", "collection": collection_name}


@app.get("/create_collection", status_code=status.HTTP_201_CREATED)
def create_collection_endpoint(collection_name: str = Query(..., description="The name of the collection to create")):
    try:
        # Assume VectorDBInstance is an instance of a class that has the create_collection method.
        vector_db = VectorDB()
        collection = vector_db.create_collection(collection_name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An error occurred: {e}")

    return {"message": f"Collection '{collection_name}' created successfully", "collection": collection}


@app.get("/query")
def query(query : str, n_results : Optional[int] = 2, collection_name : Optional[str] = "general_collection"):
    vector_db = VectorDB()
    search = vector_db.search(query = query, k = n_results, collection_name= collection_name)
    prompt = create_prompt(query, search)
    output = ml_models["answer_to_query"](prompt)
    return {"message": f"Query is {query}",
            "relavent_docs" : search,
            "llm_output" : output}

