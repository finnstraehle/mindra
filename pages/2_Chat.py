
# Imports
import os
import pandas as pd
import numpy as np
import streamlit as st
import openai
import faiss
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set Streamlit page config for a nicer layout
st.set_page_config(page_title="Interview Insights Chat", layout="wide")

# Title of the app
st.title("💬 Qualitative Interview Insights Chat")

# Description/instructions (optional)
st.markdown("""
Willkommen! Diese Anwendung ermöglicht es, in qualitativen Interviewdaten zu stöbern und Fragen dazu zu stellen.
Wählen Sie zunächst links ggf. Filter (Dateien, Rollen, Firmen, Kategorien) aus oder geben Sie Stichworte ein.
Stellen Sie dann eine Frage im Textfeld unten. Die KI wird Ihre Frage auf Basis der Interviewaussagen beantworten und relevante Zitate anführen.
""")

# Sidebar filters
st.sidebar.header("Filter Optionen")
# Load data once (with caching to avoid re-loading on each interaction)
@st.cache_data
def load_data(path_pattern="data/*.csv"):
    files = sorted([f for f in os.listdir("data") if f.endswith(".csv")])
    df_list = []
    for fname in files:
        fpath = os.path.join("data", fname)
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            st.error(f"Fehler beim Laden von {fname}: {e}")
            continue
        df['source_file'] = fname  # add source file name column
        df_list.append(df)
    if not df_list:
        return pd.DataFrame()  # no data loaded
    df_all = pd.concat(df_list, ignore_index=True)
    # Normalize column names (strip whitespace)
    df_all.columns = df_all.columns.str.strip()
    return df_all, files

df_all, file_list = load_data()

if df_all.empty:
    st.warning("Keine Daten gefunden. Bitte stellen Sie sicher, dass CSV-Dateien im data/-Ordner vorhanden sind.")
    st.stop()

# Prepare options for filters
# Unique values for roles, firms, types
roles = sorted([r for r in df_all['Rolle'].dropna().unique()])
firmen = sorted([f for f in df_all['Firma'].dropna().unique()])
typen = sorted([t for t in df_all['Typ'].dropna().unique()])

# Sidebar multiselects (with default to all for files, and no selection = all for others)
selected_files = st.sidebar.multiselect("Interview-Dateien", file_list, default=file_list)
selected_roles = st.sidebar.multiselect("Rolle", roles)  # empty = no filter (means all)
selected_companies = st.sidebar.multiselect("Firma", firmen)
selected_types = st.sidebar.multiselect("Typ", typen)
keyword = st.sidebar.text_input("Stichwort in Aussage")

# Prepare the filtered dataframe based on selections
def filter_data(df):
    # Start with file filter (if none selected, we consider all; but none selected won't happen here because default=all)
    if selected_files:
        df_filtered = df[df['source_file'].isin(selected_files)].copy()
    else:
        df_filtered = df.copy()
    # Apply role filter if any selected
    if selected_roles:
        df_filtered = df_filtered[df_filtered['Rolle'].isin(selected_roles)]
    # Apply company filter
    if selected_companies:
        df_filtered = df_filtered[df_filtered['Firma'].isin(selected_companies)]
    # Apply type filter
    if selected_types:
        df_filtered = df_filtered[df_filtered['Typ'].isin(selected_types)]
    # Apply keyword filter (check in 'Aussage' field)
    if keyword:
        # For multiple keywords separated by space, ensure all are present
        terms = [t.strip() for t in keyword.split() if t.strip()]
        for term in terms:
            df_filtered = df_filtered[df_filtered['Aussage'].astype(str).str.contains(term, case=False, na=False)]
    return df_filtered

df_filtered = filter_data(df_all)

# Download buttons for exporting filtered data
st.sidebar.markdown("### Export")
csv_data = df_filtered.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("💾 CSV herunterladen", data=csv_data, file_name="filtered_results.csv", mime="text/csv")
# Excel export requires using BytesIO and a writer
def to_excel_bytes(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Ergebnisse')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

xlsx_data = to_excel_bytes(df_filtered)
st.sidebar.download_button("💾 Excel herunterladen", data=xlsx_data, file_name="filtered_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Build or load vector index for semantic search (cache to avoid recompute)
@st.cache_resource
def build_vector_index(df):
    # Combine text fields to embed
    texts = []
    for _, row in df.iterrows():
        # Create a combined text including Aussage and any additional descriptive fields
        parts = []
        # We'll include Cluster and Beschreibung if present, as well as any other non-empty string fields beyond basic meta
        for col in df.columns:
            if col in ["NR", "source_file", "Rolle", "Firma", "Typ"]:
                continue  # skip meta fields in embedding content (we add them to context in prompt separately)
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        # Always include the main statement
        statement = str(row.get("Aussage", "")) if row.get("Aussage", "") is not np.nan else ""
        parts.append(statement)
        full_text = " ".join(parts)
        texts.append(full_text)
    # Get embeddings from OpenAI
    try:
        # Note: the API key must be set in environment or via openai.api_key before calling
        embeds = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
    except Exception as e:
        st.error("Fehler beim Erstellen der Embeddings. Bitte API-Key überprüfen und erneut versuchen.\n\n" + str(e))
        return None, None, None
    # The API returns a list of embedding dicts, extract the vectors
    vectors = [np.array(e["embedding"], dtype=np.float32) for e in embeds["data"]]
    # Convert to 2D numpy array
    vector_matrix = np.vstack(vectors)
    # Normalize for cosine similarity
    faiss.normalize_L2(vector_matrix)
    index = faiss.IndexFlatIP(vector_matrix.shape[1])
    index.add(vector_matrix)
    return index, vector_matrix, texts

# Ensure OpenAI API key is set (prompt user if not set via env variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.sidebar.text_input("OpenAI API-Key", type="password")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    st.warning("Bitte einen gültigen OpenAI API-Key eingeben, um Fragen stellen zu können.")
    # We don't stop execution, user can still use filtering without asking questions.

# Build the FAISS index (once) for the entire dataset
index, vector_matrix, corpus_texts = build_vector_index(df_all)
if index is None:
    st.stop()  # error already shown

# Helper: function to get top N similar statements (indices) for a given query, within filtered subset
def semantic_search(query, top_n=5):
    # Get embedding for query
    try:
        q_embed_response = openai.Embedding.create(input=[query], model="text-embedding-ada-002")
    except Exception as e:
        st.error("Fehler beim Embedding der Anfrage: " + str(e))
        return []
    q_vector = np.array(q_embed_response["data"][0]["embedding"], dtype=np.float32)
    faiss.normalize_L2(q_vector.reshape(1, -1))
    # If no filters (all data allowed), use FAISS index directly
    if (not selected_roles) and (not selected_companies) and (not selected_types) and (len(selected_files) == len(file_list)) and (not keyword):
        # No filter active, search globally
        D, I = index.search(q_vector.reshape(1, -1), top_n)
        top_indices = I[0]
    else:
        # Filter active: restrict search to the filtered subset
        filtered_indices = df_filtered.index.values  # these correspond to positions in df_all (since df_all was concatenated with ignore_index)
        if len(filtered_indices) == 0:
            return []  # no data in subset
        # Compute similarity for each candidate in subset
        # We can use the precomputed matrix: take those rows and do dot with q_vector
        # vector_matrix is normalized; q_vector normalized; dot = cosine sim
        sub_matrix = vector_matrix[filtered_indices]
        sims = np.dot(sub_matrix, q_vector)  # shape (len(subset),)
        # Get indices of top sims
        if len(sims) < top_n:
            top_idx_sub = np.argsort(sims)[::-1]  # all of them, sorted descending
        else:
            top_idx_sub = np.argsort(sims)[::-1][:top_n]
        top_indices = filtered_indices[top_idx_sub]
    return list(top_indices)

# Chat input for user question
st.markdown("----")
st.subheader("Frage stellen")
user_query = st.text_input("Geben Sie hier Ihre Frage ein (und drücken Sie Enter):", "")
if user_query:
    # When a question is asked, perform retrieval and get answer from LLM
    # Retrieve top relevant statements
    top_indices = semantic_search(user_query, top_n=10)
    if not top_indices:
        st.write("*(Keine passenden Aussagen gefunden.)*")
    else:
        # Prepare context for LLM: relevant statements with metadata
        context_messages = []
        context_text = "Kontext:\n"
        for idx in top_indices:
            row = df_all.iloc[idx]
            # Compose a reference string for this statement
            source = f"{row['source_file']} Zeile {idx}"
            role = str(row['Rolle']) if not pd.isna(row['Rolle']) else "Unbekannte Rolle"
            firma = str(row['Firma']) if not pd.isna(row['Firma']) else "Unbekannte Firma"
            typ = str(row['Typ']) if not pd.isna(row['Typ']) else ""
            statement = str(row['Aussage']).strip()
            # Add to context text
            context_text += f"- ({source}) **{role}**, *{firma}* ({typ}): \"{statement}\"\n"
        # Define system prompt
        system_prompt = (
            "Du bist ein Assistent, der bei der Analyse qualitativer Interviews hilft. "
            "Du erhältst eine Nutzerfrage und einige Ausschnitte aus Interviews (Kontext), und sollst die Frage basierend auf diesen Informationen beantworten. "
            "Antworte **auf Deutsch** in vollständigen Sätzen. Gib eine strukturierte und ausführliche Antwort. "
            "Belege dabei wichtige Aussagen mit Quellenangaben in Klammern (Dateiname und Zeile). "
            "Verwende nur Informationen aus dem gegebenen Kontext – wenn die Information nicht in den Interview-Aussagen steht, sage ehrlich, dass du es nicht weißt."
        )
        # Create the message payload for OpenAI ChatCompletion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": context_text},
            {"role": "user", "content": user_query}
        ]
        # Call OpenAI ChatCompletion (GPT-4)
        try:
            response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        except Exception as e:
            st.error("Fehler bei der LLM-Antwort: " + str(e))
            st.stop()
        answer = response["choices"][0]["message"]["content"]
        # Display the question and answer in chat format
        st.markdown(f"**Frage:** {user_query}")
        st.markdown(f"**Antwort:**\n\n{answer}")
        # Display the supporting statements in formatted boxes
        st.markdown("*Relevante Aussagen:*")
        for idx in top_indices:
            row = df_all.iloc[idx]
            source = f"{row['source_file']} Zeile {idx}"
            role = str(row['Rolle']) if not pd.isna(row['Rolle']) else "N/A"
            firma = str(row['Firma']) if not pd.isna(row['Firma']) else "N/A"
            typ = str(row['Typ']) if not pd.isna(row['Typ']) else ""
            statement = str(row['Aussage']).strip()
            # Format role, company, type, statement, source
            formatted = f"**{role}**  _{firma}_  <span style='color:gray;font-size:0.9em'>{typ}</span><br/>"
            formatted += f"{statement}<br/>"
            formatted += f"<span style='color:gray;font-size:0.8em'>Quelle: {source}</span>"
            st.markdown(f"{formatted}", unsafe_allow_html=True)
            st.markdown("---")
else:
    # If no question asked yet, but filters applied, show filtered data (not all if unfiltered)
    if (selected_roles or selected_companies or selected_types or keyword or (len(selected_files) < len(file_list))):
        count = len(df_filtered)
        st.subheader(f"Gefilterte Ergebnisse ({count} Aussagen):")
        if count == 0:
            st.write("*(Keine Einträge entsprechen den Filterkriterien.)*")
        else:
            # Show each filtered statement in the formatted box
            for i, (_, row) in enumerate(df_filtered.iterrows()):
                source = f"{row['source_file']} Zeile {row.name}"
                role = str(row['Rolle']) if not pd.isna(row['Rolle']) else "N/A"
                firma = str(row['Firma']) if not pd.isna(row['Firma']) else "N/A"
                typ = str(row['Typ']) if not pd.isna(row['Typ']) else ""
                statement = str(row['Aussage']).strip()
                formatted = f"**{role}**  _{firma}_  <span style='color:gray;font-size:0.9em'>{typ}</span><br/>"
                formatted += f"{statement}<br/>"
                formatted += f"<span style='color:gray;font-size:0.8em'>Quelle: {source}</span>"
                st.markdown(formatted, unsafe_allow_html=True)
                st.markdown("---")
