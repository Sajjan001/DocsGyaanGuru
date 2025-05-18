from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return FAISS.from_documents(chunks, embeddings)

def get_relevant_context(query, vector_store, top_k=3):
    docs = vector_store.similarity_search(query, k=top_k)
    return "\n\n".join(doc.page_content for doc in docs)
