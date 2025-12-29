import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

st.set_page_config(page_title="JARVIS - ISO 27001 Agent")
st.title("🛡️ JARVIS: ISO 27001 AI Agent")

# Load the Brain
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Create the Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

# UI Layout
query = st.text_input("Ask JARVIS about a specific ISO 27001 Clause or Control:")

if query:
    with st.spinner("Analyzing standard..."):
        response = qa_chain.invoke(query)
        st.write("### JARVIS's Guidance:")
        st.info(response["result"])
