from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


file_path = "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()


# embeddings = OllamaEmbeddings(model="gemma3")
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))


vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

import os
os.environ["PINECONE_API_KEY"] = "YOUR_API_KEY_HERE"

pc = Pinecone(api_key="YOUR_API_KEY_HERE")
index = pc.Index("harmonious-sequoia")

vector_store = PineconeVectorStore.from_documents(
    documents=all_splits,
    embedding=embeddings,
    index_name="harmonious-sequoia",  # must match the one you just created
)

# index_name="harmonious-sequoia"  # must match the one you just created
# existing = [i["name"] for i in pc.list_indexes().indexes]
# if index_name not in existing:
#     pc.create_index(
#         name=index_name,
#         # dimension=len(int(768)),  # safer than hard-coding 768
#         dimension=len(all_splits),
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1",
#         )
#     ),

# 5. Build vector store and add documents
vector_store = PineconeVectorStore.from_documents(
    documents=all_splits,
    embedding=embeddings,
    index_name="harmonious-sequoia",
)




# Embeddings typically represent text as a “dense” vector such that texts with similar meanings are geometrically close. This lets us retrieve relevant information just by passing in a question, without knowledge of any specific key-terms used in the document.
# Return documents based on similarity to a string query:

# 6. Query
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?", k=3
)

print(results[0])


# Async query:
import asyncio

async def run_query():
    results = await vector_store.asimilarity_search(
        "When was Nike incorporated?",
        k=3
    )
    print(results[0])

asyncio.run(run_query())


# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)


embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])