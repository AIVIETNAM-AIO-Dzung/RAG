import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipelines
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import streamlit as st
import tempfile 



def text_splitter(documents:str,embed_model:str):
    embed_model = HuggingFaceEmbeddings(model_name=embed_model)
    semantic_splitter = SemanticChunker(embeddings=embed_model,
                                        buffer_size=1,
                                        breakpoint_threshold_type="percentile",
                                        breakpoint_threshold_amount=95,
                                        min_chunk_size=500,
                                        add_start_index=True
                                        )
    
    return semantic_splitter.split_documents(documents=documents)

@st.cache_resource
def load_embeddings(embed_model:str):
    return HuggingFaceEmbeddings(model_name=embed_model)
    

    
@st.cache_resource
def initiate_llm_pipeline(model_name:str):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipelines(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    return HuggingFacePipeline(pipeline=model_pipeline)

def process_pdf(uploaded_file, embed_model,llm):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load

    docs = text_splitter(documents, embed_model)
    vector_db =Chroma.from_documents(documents=docs,
                                     embedding=embed_model,
                                     persist_directory="./RAG",
                                     )
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