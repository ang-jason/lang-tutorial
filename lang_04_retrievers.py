from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec




import os
os.environ["PINECONE_API_KEY"] = "pcsk_6msY7k_GzToTwEG4zVR783Uu2ysQ4eDtQcppEjfYHcLZddWuvPegSb7XiAXWVYYuxaoHte"
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

vector_store = PineconeVectorStore(
    index_name="harmonious-sequoia",
    embedding=embeddings
)


from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.runnables import chain

@chain
def retriever(query: str) -> Dict[str, List[Document]]:
    docs = vector_store.similarity_search(query, k=1)
    return {"documents": docs}

results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ]
)

print(results)