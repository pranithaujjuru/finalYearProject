import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def initialize_guideline_db():
    print("Initializing Local Guideline Database (ChromaDB)...")
    
    # Path for the database
    persist_directory = "./chroma_db"
    
    # Guidelines to ingest
    guidelines = [
        "MRI findings of severe herniation with progressive neurological deficit (weakness) require urgent surgical consultation.",
        "Conservative management (PT, NSAIDs) is recommended for mild to moderate herniation without red flags.",
        "Red flags include: Cauda Equina Syndrome (saddle anesthesia, bowel/bladder dysfunction), severe motor weakness, or history of malignancy.",
        "Pharmacological management for sciatica includes neuropathic pain medications like gabapentin or pregabalin if symptoms persist.",
        "Surgical intervention (discectomy) is considered if conservative treatment fails after 6-12 weeks and there is clear radiological correlation."
    ]
    
    # Convert to LangChain documents
    documents = [Document(page_content=text, metadata={"source": "Internal Clinical Guidelines"}) for text in guidelines]
    
    # Initialize embeddings with a local cache folder to avoid 429 rate-limits
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", cache_folder="./models/embeddings")
    
    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✅ Successfully initialized {len(guidelines)} guideline chunks in: {persist_directory}")

if __name__ == "__main__":
    initialize_guideline_db()
