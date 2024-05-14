from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredHTMLLoader, CSVLoader, 
    UnstructuredExcelLoader, TextLoader, Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

def load_and_process_file(file_path):
    file_path = Path(file_path)
    loaders = {
        '.html': UnstructuredHTMLLoader,
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
        '.csv': CSVLoader,
        '.xlsx': UnstructuredExcelLoader,
    }
    file_extension = file_path.suffix.lower()
    loader_class = loaders.get(file_extension)
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    text_splitter = initialize_splitter()
    loader = loader_class(file_path)
    return loader.load_and_split(text_splitter)

def initialize_splitter(chunk_size=1000, chunk_overlap=100):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

