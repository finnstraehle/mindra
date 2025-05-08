import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI

import os
from dotenv import load_dotenv

# OpenAI API-Key über .env laden
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("Fehlender OpenAI API-Key. Bitte `.env` Datei erstellen mit `OPENAI_API_KEY=...`")
    st.stop()

client = OpenAI(api_key=openai_api_key)
import faiss
from io import StringIO

# OpenAI API Client (neu)
from openai import OpenAI

# Titel und grundlegende Beschreibung der App
st.title("Interview Knowledge Base Explorer")
st.write("Diese App ermöglicht es, qualitativ kodierte Interviewdaten zu explorieren, zu analysieren und abzufragen.")

# OpenAI API-Key über .env laden
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("Fehlender OpenAI API-Key. Bitte `.env` Datei erstellen mit `OPENAI_API_KEY=...`")
    st.stop()

# OpenAI Python v1 Client initialisieren
client = OpenAI(api_key=openai_api_key)

# 1. Daten einlesen und bereinigen
# -------------------------------
# Lese die sieben CSV-Dateien ein und stelle sicher, dass sie eine konsistente Struktur haben.
# Spalten: NR, Rolle, Firma, Typ, Cluster, Beschreibung, Aussage, Quelle (jede Zeile = eine Aussage).
@st.cache_data(show_spinner=False)
def load_and_clean_data():
    files = {
        "file1.csv": "file1.csv",
        "file2.1.csv": "file2.1.csv",
        "file2.2.csv": "file2.2.csv",
        "file3.1.csv": "file3.1.csv",
        "file3.2.csv": "file3.2.csv",
        "file4.csv": "file4.csv",
        "file5.csv": "file5.csv"
    }
    data_frames = {}
    for label, filename in files.items():
        try:
            # CSV einlesen; dtype=str sorgt dafür, dass alles als Text gelesen wird
            data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
            df = pd.read_csv(os.path.join(data_path, filename), dtype=str, keep_default_na=False, engine='python')
        except Exception as e:
            st.error(f"Fehler beim Lesen von {filename}: {e}")
            continue
        # Whitespace an beiden Enden in allen Zellen entfernen
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # Dateispezifische Bereinigungen:
        if label == "file2.1.csv":
            # "Connected Use Case" und "EDM oder EDC?" in Beschreibung integrieren
            usecase_col = "Connected Use Case (wie wurde das erreicht?) "
            edm_col = "EDM oder EDC? "
            if usecase_col in df.columns and edm_col in df.columns:
                new_besch = []
                for _, row in df.iterrows():
                    besch = row.get("Beschreibung", "")
                    usecase = row.get(usecase_col, "")
                    edm_edc = row.get(edm_col, "")
                    parts = []
                    if usecase:
                        parts.append(f"Use Case: {usecase}")
                    if edm_edc:
                        parts.append(f"Kontext: {edm_edc}")
                    extra = " | ".join(parts) if parts else ""
                    # Bestehende Beschreibung mit Zusatzinfos verknüpfen
                    if besch and extra:
                        new_besch.append(f"{besch} | {extra}")
                    elif besch:
                        new_besch.append(besch)
                    else:
                        new_besch.append(extra)
                df["Beschreibung"] = new_besch
                # Entferne die zwei zusätzlichen Spalten
                df.drop(columns=[usecase_col, edm_col], inplace=True)
        if label == "file2.2.csv":
            # Spezielle Spalten (Einsparungen etc.) in Cluster/Beschreibung integrieren
            # Unbenannte letzte Spalte als "Quelle" interpretieren, falls vorhanden
            for col in list(df.columns):
                if col.startswith("Unnamed"):
                    df.rename(columns={col: "Quelle"}, inplace=True)
            if "Quelle" not in df.columns:
                df["Quelle"] = ""
            # Werte aus "Bereiche...", "konkrete Einsparungen", "Skala", "Allg. Aussagen" verarbeiten
            area_col = "Bereiche in die eingespart wurde"
            savings_col = "konkrete Einsparungen / Zahlen"
            scale_col = "Skala Kosteneinsparungen 1–5"
            measure_col = "Allgemeine Aussagen zur Messung"
            if area_col in df.columns:
                new_cluster = []
                new_desc = []
                for _, row in df.iterrows():
                    area = row.get(area_col, "")
                    concrete = row.get(savings_col, "")
                    scale = row.get(scale_col, "")
                    measure = row.get(measure_col, "")
                    # Cluster: benutze Bereich, falls vorhanden
                    cluster_val = area if area else ""
                    new_cluster.append(cluster_val)
                    # Beschreibung aus restlichen Feldern zusammenbauen
                    parts = []
                    if concrete:
                        parts.append(f"Einsparung: {concrete}")
                    if scale:
                        parts.append(f"Skala: {scale}")
                    if measure:
                        parts.append(f"Messung: {measure}")
                    new_desc.append(" | ".join(parts) if parts else "")
                df["Cluster"] = new_cluster
                df["Beschreibung"] = new_desc
                # Alte Spalten entfernen
                drop_cols = [area_col, savings_col, scale_col, measure_col]
                for c in drop_cols:
                    if c in df.columns:
                        df.drop(columns=c, inplace=True)
        if label == "file3.2.csv":
            # "Dauer min." und "Dauer max." Spalten entfernen (Info bereits in Beschreibung enthalten)
            if "Dauer min." in df.columns:
                df.drop(columns=["Dauer min."], inplace=True)
            if "Dauer max." in df.columns:
                df.drop(columns=["Dauer max."], inplace=True)
            # Unbenannte leere Spalten entfernen
            unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
            if unnamed_cols:
                df.drop(columns=unnamed_cols, inplace=True)
        if label in ["file4.csv", "file5.csv"]:
            # Inhalte aus evtl. unbenannten Spalten mit Beschreibung zusammenführen (falls durch Kommas versprengt)
            unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
            if unnamed_cols:
                new_besch = []
                for _, row in df.iterrows():
                    besch = row.get("Beschreibung", "")
                    extra_parts = [str(row.get(col, "")).strip() for col in unnamed_cols if row.get(col, "")]
                    extra_parts = [part for part in extra_parts if part and part.lower() != 'nan']
                    if extra_parts:
                        extra_text = ", ".join(extra_parts)
                        new_besch.append(f"{besch}, {extra_text}" if besch else extra_text)
                    else:
                        new_besch.append(besch)
                df["Beschreibung"] = new_besch
                df.drop(columns=unnamed_cols, inplace=True)
        # Stelle sicher, dass "Quelle" Spalte existiert
        if "Quelle" not in df.columns:
            df["Quelle"] = ""
        # Spalten in gewünschter Reihenfolge anordnen
        cols = ["NR", "Rolle", "Firma", "Typ", "Cluster", "Beschreibung", "Aussage", "Quelle"]
        df = df.reindex(columns=cols)
        # Originale Zeilennummer im Ursprungs-CSV speichern (1-basierter Index inkl. Header)
        df["Line"] = df.index + 2
        data_frames[label] = df.fillna("")
    # Alle Teil-Datensätze zusammenführen und Dateiname als Spalte hinzufügen
    combined = pd.DataFrame()
    for label, df in data_frames.items():
        df = df.copy()
        df["Datei"] = label
        combined = pd.concat([combined, df], ignore_index=True)
    return combined

data_df = load_and_clean_data()

# 2. Filter in der Hauptseite (nicht Sidebar)
# ------------------------------------------
# Ermöglicht Filterung nach Cluster, Typ, Stichwort (freier Text) und CSV-Datei.
st.subheader("Filter")
cluster_options = sorted({c for c in data_df["Cluster"] if c})
type_options = sorted({t for t in data_df["Typ"] if t})
file_options = sorted(set(data_df["Datei"]))

selected_clusters = st.multiselect("Cluster auswählen", cluster_options)
selected_types = st.multiselect("Typ auswählen", type_options)
selected_files = st.multiselect("Quelle (Datei) auswählen", file_options, default=file_options)
keyword = st.text_input("Stichwortsuche in Aussagen/Beschreibung")

# Filter auf DataFrame anwenden
filtered_df = data_df.copy()
if selected_clusters:
    filtered_df = filtered_df[filtered_df["Cluster"].isin(selected_clusters)]
if selected_types:
    filtered_df = filtered_df[filtered_df["Typ"].isin(selected_types)]
if selected_files:
    filtered_df = filtered_df[filtered_df["Datei"].isin(selected_files)]
if keyword:
    kw = keyword.lower()
    mask = filtered_df["Aussage"].str.lower().str.contains(kw) | filtered_df["Beschreibung"].str.lower().str.contains(kw)
    filtered_df = filtered_df[mask]

# 3. Optionale Visualisierung der Aussagen-Verteilung
# ---------------------------------------------------
# (z. B. nach Cluster oder Typ, abhängig vom aktuellen Filter)
with st.expander("Datenverteilung anzeigen (Cluster/Typ Häufigkeit)"):
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cluster-Verteilung (Anzahl Aussagen)**")
            cluster_counts = filtered_df["Cluster"].replace("", "(kein Cluster)").value_counts()
            st.bar_chart(cluster_counts)
        with col2:
            st.markdown("**Typ-Verteilung (Anzahl Aussagen)**")
            type_counts = filtered_df["Typ"].replace("", "(kein Typ)").value_counts()
            st.bar_chart(type_counts)
    else:
        st.write("Keine Daten für ausgewählte Filter.")

# 4. Frageingabe und Modusauswahl (deskriptiv vs. interpretierend)
# --------------------------------------------------------------
st.subheader("Frage stellen")
query = st.text_area("Ihre Frage an die Interview-Daten:", "")
answer_mode = st.radio("Antwortmodus", ("Nur deskriptiv (zusammenfassend)", "Interpretierend (mit Empfehlungen)"))

# 5. Vorbereitung semantische Suche (FAISS Index mit OpenAI Embeddings)
# --------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def prepare_faiss_index(df: pd.DataFrame):
    texts = []
    metadata = []
    for _, row in df.iterrows():
        # Erstelle zu durchsuchenden Text (kombiniere relevante Felder für bessere Trefferquote)
        text = f"{row['Rolle']} {row['Firma']} {row['Typ']} {row['Cluster']} {row['Beschreibung']} {row['Aussage']}"
        texts.append(text)
        metadata.append((row["Datei"], row["Line"]))
    # Embeddings in Batches abrufen (Model: text-embedding-ada-002)
    batch_size = 100
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        response = client.embeddings.create(model="text-embedding-ada-002", input=batch_texts)
        batch_embs = [np.array(e.embedding, dtype="float32") for e in response.data]
        embeddings_list.extend(batch_embs)
    embeddings_matrix = np.vstack(embeddings_list)
    # FAISS Index mit Kosinus-Ähnlichkeit (via inner product auf normalisierten Vektoren)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    embeddings_normed = embeddings_matrix / (norms + 1e-10)
    index = faiss.IndexFlatIP(embeddings_normed.shape[1])
    index.add(embeddings_normed)
    return index, metadata, embeddings_normed

# 6. Beantwortung der Frage mittels GPT-4 (RAG Pipeline)
# -----------------------------------------------------
if st.button("Antwort generieren"):
    # Sicherstellen, dass Voraussetzungen erfüllt sind
    if not openai_api_key:
        st.error("Bitte zuerst einen gültigen OpenAI API-Key eingeben.")
    elif not query.strip():
        st.error("Bitte eine Frage eingeben.")
    elif filtered_df.empty:
        st.warning("Für die gewählten Filter sind keine Daten vorhanden.")
    else:
        # FAISS Index erstellen (oder aus Cache laden)
        faiss_index, meta_list, embed_matrix = prepare_faiss_index(filtered_df)
        if faiss_index is None:
            st.stop()
        # Anfrage-Embedding erzeugen
        try:
            q_response = client.embeddings.create(model="text-embedding-ada-002", input=[query])
            q_emb = q_response.data[0].embedding
        except Exception as e:
            st.error(f"Fehler bei der Embedding-Erstellung der Frage: {e}")
            st.stop()
        q_vec = np.array(q_emb, dtype="float32")
        # Query-Vektor normalisieren
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)
        # Ähnlichste K Dokumente finden
        K = 5
        D, I = faiss_index.search(q_vec.reshape(1, -1), K)
        top_indices = I[0]
        # Kontext aus den Top-Ergebnissen zusammenstellen
        context_snippets = []
        references = []
        for idx in top_indices:
            if idx < len(filtered_df):
                row = filtered_df.iloc[idx]
                role = row["Rolle"]; company = row["Firma"]; typ = row["Typ"]
                statement = row["Aussage"]
                source_file = row["Datei"]; source_line = row["Line"]
                references.append((source_file, source_line, role, company, typ, statement))
                context_snippets.append(f"[{source_file} Zeile {source_line}] {role} @ {company} ({typ}): {statement}")
        # GPT-4 Prompt erstellen
        system_prompt = (
            "Du bist ein Assistent, der Fragen anhand von Interview-Aussagen beantwortet.\n"
        )
        if answer_mode.startswith("Nur deskriptiv"):
            system_prompt += "Gib eine zusammenfassende, deskriptive Antwort basierend auf den bereitgestellten Aussagen. "
            system_prompt += "Füge keine eigenen Interpretationen oder Empfehlungen hinzu.\n"
        else:
            system_prompt += "Gib eine interpretierende Antwort mit möglichen Erklärungen oder Empfehlungen basierend auf den Aussagen. "
            system_prompt += "Du darfst Schlussfolgerungen ziehen, aber stütze dich auf die Inhalte der Aussagen.\n"
        system_prompt += "Zitiere relevante Quellen in der Form (Datei, Zeile) für jede wichtige Aussage in deiner Antwort."
        # Kontext der Aussagen als zusätzliche System-Nachricht
        context_text = "Relevante Aussagen:\n" + "\n".join(context_snippets)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_text},
            {"role": "user", "content": query.strip()}
        ]
        # GPT-4 Anfrage ausführen
        try:
            completion = client.chat.completions.create(model="gpt-4-turbo", messages=messages, temperature=0.2, max_tokens=4000)
        except Exception as e:
            st.error(f"Fehler bei der GPT-4 Anfrage: {e}")
            st.stop()
        answer_text = completion.choices[0].message.content
        # Antwort anzeigen
        st.subheader("Antwort")
        st.write(answer_text)
        # Quellen (die herangezogenen Aussagen) anzeigen im gewünschten Layout
        st.subheader("Quellen aus den Interviews")
        for (src_file, src_line, role, company, typ, statement) in references:
            st.markdown(f"**{role}, {company} ({typ})**")  # Rolle, Firma, Typ oben
            quote_text = "> " + statement.replace("\n", "\n> ")
            st.markdown(quote_text)  # Aussage im Blockquote (gepolstertes Textfeld)
            st.markdown(f"*Quelle: {src_file}, Zeile {src_line}*")  # Quelle am Ende
        # 7. Export-Funktion: Antwort und Quellen als Markdown oder CSV
        # ------------------------------------------------------------
        output_md = f"**Frage:** {query}\n\n**Antwort:**\n{answer_text}\n\n**Quellen:**\n"
        for (src_file, src_line, role, company, typ, statement) in references:
            output_md += f"- **{role}, {company} ({typ})** – {statement} *(Quelle: {src_file}, Zeile {src_line})*\n"
        output_csv_df = pd.DataFrame([{
            "Frage": query,
            "Antwort": answer_text,
            "Quellen": "; ".join([f"{sfile} Zeile {sline}" for (sfile, sline, *_ ) in references])
        }])
        csv_data = output_csv_df.to_csv(index=False).encode("utf-8")
        st.download_button("Antwort und Quellen als Markdown herunterladen", output_md, file_name="antwort_und_quellen.md")
        st.download_button("Antwort und Quellen als CSV herunterladen", csv_data, file_name="antwort_und_quellen.csv")
