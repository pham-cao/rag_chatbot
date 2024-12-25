import streamlit as st
import time
import tempfile
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from qdrant_client.http.exceptions import UnexpectedResponse
from utils.vector_db import get_list_collection_names, delete_collection
from RAG.VectorStoreDB import load_docs_from_text
import os
from RAG.Ingestor import Ingestor


ingestor = Ingestor()



st.set_page_config(page_title="Document Manager", page_icon="📈")




def show_content(text):
    st.text_area("Content", text, height=500)


def load_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1], dir='tmp') as tmp_file:

        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_file_path)
        else:
            st.error("Unsupported file type. Please upload a PDF or DOCX file.")

    documents = loader.load()

    text = ''.join(page.page_content for page in documents)
    os.remove(temp_file_path)
    return text


tab3, tab2 = st.tabs(["Insert Document", 'Document' ])

with tab2:
    st.title("Document List")
    collection_names = get_list_collection_names()
    if collection_names:
        df = pd.DataFrame(collection_names)
        event = st.dataframe(data=df,
                             on_select="rerun",
                             selection_mode="multi-row",
                             use_container_width=True,
                             hide_index=True)

        remove_df = df.loc[event.selection.get('rows')][['name', 'id']]

        if not remove_df.empty:
            if st.button("Delete Document"):
                with st.spinner('processing...'):
                    for _, row in remove_df.iterrows():
                        print(row)
                        delete_collection(row['name'], row['id'])
                    st.rerun()
with tab3:
    st.title("Upload Document")
    uploaded_file = st.file_uploader("Tải lên file của bạn với định dạng pdf hoặc word:", type=["pdf", "docx"])
    if uploaded_file is not None:

        text = load_document(uploaded_file)
        button1, button2 = st.columns(2)
        if button2.button("Insert to Database"):
            # if description:
            with st.spinner('processing...'):
                try:
                    ingestor.insert(text,uploaded_file.name)
                    # load_docs_from_text(text, uploaded_file.name, description)
                    st.success(" insert completed successfully!")
                    time.sleep(1)
                    st.rerun()
                except UnexpectedResponse as e:
                    if e.status_code == 409:
                        st.error('⚠️ Document already exists!')
                    else:
                        st.error(e)

        if button1.button("show content"):
            show_content(text=text)
