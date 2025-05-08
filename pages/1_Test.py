import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit interface
st.title("Research Interview Data Chat")

# Define data directory and files
# Define the data directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# List of predefined CSV files
csv_files = [
    'file1.csv',
    'file2.1.csv',
    'file2.2.csv',
    'file3.1.csv',
    'file3.2.csv',
    'file4.csv',
    'file5.csv',
]

predefined_files = [os.path.join(data_dir, file) for file in csv_files]

# Load documents from CSV files
documents = []
for file_path in predefined_files:
    if file_path.lower().endswith('.csv'):
        loader = CSVLoader(file_path=file_path)
        documents.extend(loader.load())

# Create vector store
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.from_documents(documents, embeddings)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Chat interface
query = st.text_input("Ask a question about your research:")
if query:
    # Create retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0.7),  # Adjusted temperature for balanced responses
        vectorstore.as_retriever(search_kwargs={"k": 10}),  # Retrieve top 10 relevant documents
        return_source_documents=True
    )

    # Get response
    response = qa_chain({"question": query, "chat_history": st.session_state["chat_history"]})

    # Update chat history
    st.session_state["chat_history"].append((query, response["answer"]))

    # Display detailed response
    st.subheader("Response:")
    st.write(response["answer"])

    # Display files searched
    st.subheader("Files Searched:")
    st.markdown(
        f"""
        <ul>
        {''.join([f"<li>{os.path.basename(file)}</li>" for file in predefined_files])}
        </ul>
        """,
        unsafe_allow_html=True
    )

    # Display source data
    st.subheader("Source Data:")
    for doc in response["source_documents"]:
        text = doc.page_content

        # Extract Rolle, Firma, Typ, and Aussage
        match = re.search(
            r"Rolle:\s*(?P<Rolle>.*?)\s*Firma:\s*(?P<Firma>.*?)\s*Typ:\s*(?P<Typ>.*?)\s*Aussage:\s*(?P<Aussage>[\s\S]*?)(?:Cluster:|$)",
            text
        )
        if match:
            rolle = match.group("Rolle")
            firma = match.group("Firma")
            typ = match.group("Typ")
            aussage = match.group("Aussage").strip()
        else:
            # Fallback: show full text as Aussage
            rolle = ""
            firma = ""
            typ = ""
            aussage = text.strip()

        # Styled display
        with st.container():
            st.markdown(f"#### **{rolle}** – *{firma}*", unsafe_allow_html=True)  # Smaller title: Rolle – Firma with bold and italic styling
            st.caption(f"**Typ:** {typ}")  # Subtitle: Typ with label
            st.markdown(
            f"""
            <div style="
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            ">
            {aussage}
            </div>
            """,
            unsafe_allow_html=True
            )  # Main text: Aussage with styled box
            st.markdown("<hr style='border: 1px solid #e0e0e0; margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)  # Separator with custom styling
