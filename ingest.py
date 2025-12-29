from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_db():
    loader = PyPDFLoader("data/iso_27001_standard.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    db.save_local("faiss_index")

if __name__ == "__main__":
    create_vector_db()
