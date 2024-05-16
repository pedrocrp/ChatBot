from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceHubEmbeddings, OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import chromadb
from langchain_community.document_loaders import PyPDFLoader
import re
import uuid


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


class CustomEmbeddings(OllamaEmbeddings):
    def __init__(self, model):
        super().__init__(model=model)
        
    def __call__(self, input):
        return self.embed_documents(input)


class VectorDB:
    def __init__(self, data_path="../data"):
        self.embedding_function = CustomEmbeddings(model='nomic-embed-text')
        self.db = chromadb.PersistentClient(path=data_path)


    def validate_collection_name(self, name):
        """
        Validates a collection name based on specific rules.

        Parameters:
        name (str): The name of the collection to be validated.

        Raises:
        ValueError: If the collection name does not meet the required criteria.

        Criteria for a valid collection name:
        1. Length Constraint: The name must be between 3 and 63 characters. This ensures that the name is neither too short to be meaningful nor too long for practical use.
        2. Character Constraint: The name must start and end with a lowercase letter or digit. This rule helps avoid naming inconsistencies and issues in systems that are case-sensitive or that might treat certain symbols specially.
        3. Dot Usage: The name must not contain consecutive dots ('..'). This rule prevents confusion in systems where dot notation might be used for navigation or other semantic meanings.
        4. IP Address Exclusion: The name must not be a valid IP address. This ensures that names are not confused with IP addresses, which might have special handling in network-related contexts.

        Example:
        >>> validate_collection_name("validName1")
        None
        >>> validate_collection_name("invalid..name")
        ValueError: Collection name must not contain consecutive dots.
        """
        if not (3 <= len(name) <= 63):
            raise ValueError("Collection name must be between 3 and 63 characters.")
        if not re.match(r"^[a-z0-9].*[a-z0-9]$", name):
            raise ValueError("Collection name must start and end with a lowercase letter or digit.")
        if ".." in name:
            raise ValueError("Collection name must not contain consecutive dots.")
        if re.match(r"\d+\.\d+\.\d+\.\d+", name):
            raise ValueError("Collection name must not be a valid IP address.")
        

    def create_collection(self, collection_name):
        if self.is_collection(collection_name=collection_name):
            raise ValueError(f"Collection '{collection_name}' already exists.")
        
        self.validate_collection_name(collection_name)
        collection = self.db.create_collection(name=collection_name, embedding_function=self.embedding_function)
        self.register_collection(collection_name)
        return collection
    

    def delete_collection(self, collection_name):
        if not self.is_collection(collection_name=collection_name):
            raise ValueError(f"Collection '{collection_name}' doesn't exists.")
        
        self.db.delete_collection(name=collection_name)
        self.register_collection(collection_name)
        
        
    def add_documents(self, collection_name, documents):
        if self.is_collection(collection_name=collection_name):
            collection = self.db.get_collection(name=collection_name)
            ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents]
            unique_ids = list(set(ids))

            seen_ids = set()
            unique_docs = [doc for doc, id in zip(documents, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

            page_contents = [doc.page_content for doc in unique_docs]
            metadatas = [doc.metadata for doc in unique_docs]

            collection.add(documents=page_contents, metadatas=metadatas, ids=unique_ids)
            
        else:
            raise KeyError("Collection name doesn't exist.")
    

    def update(self, name, ids, embeddings=None, metadatas=None, documents=None): # IMPLEMENTAR UPDATE COLLECTION DATA
        collection = self.get_collection(name)
        for id_, emb, meta, doc in zip(ids, embeddings, metadatas, documents):
            if id_ not in collection["items"]:
                continue  # or raise error
            collection["items"][id_].update({"embedding": emb, "metadata": meta, "document": doc})


    def delete(self, name, ids, where=None): # IMPLEMENTAR DELETE COLLECTION DATA
        collection = self.get_collection(name)
        for id_ in ids:
            if id_ in collection["items"]:
                del collection["items"][id_]
            else:
                raise KeyError("ID not found in collection.")
    

    def search(self, query, k, collection_name):
        if self.is_collection(collection_name=collection_name):
            collection = self.db.get_collection(name=collection_name)
            return collection.query(query_texts=[query], n_results = k)
        else:
            raise KeyError("Collection name doesn't exist.")


    def register_collection(self, collection_name):
        with open("COLLECTIONS.txt", "a") as f:
            f.write(collection_name + "\n")
    

    def unregister_collection(self, collection_name):
        with open("COLLECTIONS.txt", "r") as f:
            lines = f.readlines()

        # Remove the collection name from the list; include the newline character to match the format
        lines = [line for line in lines if line.strip() != collection_name]

        with open("COLLECTIONS.txt", "w") as f:
            f.writelines(lines)

    
    def is_collection(self, file_path = 'COLLECTIONS.txt', collection_name = None):
        with open(file_path, 'r', encoding='utf-8') as file:
            # Cria uma lista com cada linha do arquivo, removendo espaços em branco e quebras de linha
            collections = [line.strip() for line in file]
        return collection_name in collections


