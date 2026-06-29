from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import streamlit as st
import tempfile 
import os



@st.cache_resource
def load_embeddings(embed_model:str):
    return OllamaEmbeddings(model=embed_model)
    
def text_splitter(documents:str,embeddings:OllamaEmbeddings):
    
    semantic_splitter = SemanticChunker(embeddings=embeddings,
                                        buffer_size=1,
                                        breakpoint_threshold_type="percentile",
                                        breakpoint_threshold_amount=95,
                                        min_chunk_size=500,
                                        add_start_index=True
                                        )
    
    return semantic_splitter.split_documents(documents=documents)
    
@st.cache_resource
def initiate_llm_pipeline(model_name:str):
    return OllamaLLM(model=model_name)

def process_pdf(embeddings,llm,uploaded_file=None):
    if uploaded_file is not None:
        # ---Load  và chung PDF mới---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            documents = PyPDFLoader(tmp_file_path).load()
        finally:
            os.unlink(tmp_file_path)
        

        docs = text_splitter(documents, embeddings)

        if not docs:
            raise ValueError("Cannot chunk PDF file")
        
        # add data into database (create new if it doesnot exit, append if db available)
        vector_db =Chroma( collection_name="RAG",
                        embedding_function=embeddings,
                        persist_directory="./RAG"
                        )
        
        vector_db.add_documents(docs) # 
    else:
        #--If upload file is none -> load exist db---                
        if not os.path.exists("./RAG"):
            raise FileNotFoundError("No Database available, please upload an PDF file first")
        
        vector_db =Chroma( collection_name="RAG",
                        embedding_function=embeddings,
                        persist_directory="./RAG"
                        )
        
        if vector_db._collection.count() == 0:
            raise ValueError("Database is empty, please upload PDF first")
        
    # Retrieve data
    retriever = vector_db.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, len(docs)