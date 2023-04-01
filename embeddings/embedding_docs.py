# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:41:10 2023

@author: Jeru
"""

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_DB_DIRECTORY = "./db"

class EmbeddingDocs:
    """Wrapper around text splitter, embedding and chroma as vector db.
    Example:
        .. code-block:: python
                embeddingDocs = EmbeddingDocs('my_collection_name')
                embeddingDocs.embed_docs(docs)
    """
    
    model_name: str = DEFAULT_MODEL_NAME
    db_directory: str = DEFAULT_DB_DIRECTORY
    
    
    def __init__(
        self, 
        collection_name,
    ):
        self.collection_name = collection_name
        self.persist_directory = f'{self.db_directory}/{collection_name}'
        
        """Initialize embedding function."""
        self.embeddings = HuggingFaceEmbeddings(model_name = self.model_name)
        
    
    def _split_docs(self, docs):
        text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
        texts = text_splitter.split_documents(docs)
        return texts
    
    def embed_docs(self, docs)-> Chroma:
        chroma = Chroma(collection_name = self.collection_name)
        texts = self._split_docs(docs)
        vector_db = chroma.from_documents(texts, embedding_function = self.embeddings, persist_directory = self.persist_directory)
        vector_db.persist()
        return vector_db
        
    def load_db(self) -> Chroma:
        vector_db = Chroma(persist_directory = self.persist_directory, embedding_function = self.embeddings)
        return vector_db