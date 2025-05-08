import streamlit as st
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
import tempfile
import os
from langchain.document_loaders import UnstructuredExcelLoader
from dotenv import load_dotenv
import re
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit interface
st.title("Research Interview Data Chat")

# Load CSV files
import os
from langchain.document_loaders import CSVLoader, UnstructuredExcelLoader

data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, 'data'))
predefined_files = [
    os.path.join(data_dir, f) for f in [
        'file1.csv',
        'file2.1.csv',
        'file2.2.csv',
        'file3.1.csv',
        'file3.2.csv',
        'file4.csv',
        'file5.csv',
    ]
]

documents = []
for file_path in predefined_files:
    if file_path.lower().endswith('.csv'):
        loader = CSVLoader(file_path=file_path)
    else:
        continue
    docs = loader.load()
    documents.extend(docs)

# Create vector store
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.from_documents(documents, embeddings)

# Create chat interface
query = st.text_input("Ask a question about your research:")
if query:
    # Create retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        OpenAI(),
        vectorstore.as_retriever(),
        return_source_documents=True
    )

    # Get response
    response = qa_chain({"question": query, "chat_history": []})

    # Display response
    st.write(response["answer"])

    # Display source cells
    st.subheader("Source Data:")
    for doc in response["source_documents"]:
        text = doc.page_content

        # Extract Rolle, Firma, Typ and Aussage
        match = re.search(
            r"Rolle:\s*(?P<Rolle>.*?)\s*Firma:\s*(?P<Firma>.*?)\s*Typ:\s*(?P<Typ>.*?)\s*Aussage:\s*(?P<Aussage>[\s\S]*?)(?:Cluster:|$)",
            text
        )
        if match:
            rolle   = match.group("Rolle")
            firma   = match.group("Firma")
            typ     = match.group("Typ")
            aussage = match.group("Aussage").strip()
        else:
            # Fallback: show full text as Aussage
            rolle = ""
            firma = ""
            typ = ""
            aussage = text.strip()

        # Styled display
        with st.container():
            # Title: Rolle – Firma
            st.markdown(f"### {rolle} – {firma}")
            # Subtitle: Typ
            st.caption(typ)
            # Main text: Aussage
            st.write(aussage)
            # Separator
            st.markdown("---")
