from langchain_community.document_loaders import PyPDFLoader

file_path = "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()


print("Loading documents \n")

print(len(docs))

print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)
print("\n\n\n")

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print("Splitting documents \n")
print(len(all_splits))