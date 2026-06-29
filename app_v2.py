import langchain_function as lf
import streamlit as st
import time



LLM_MODEL = "vicuna:7b-v1.5-q5_1"
EMBED_MODEL = "qwen3-embedding:latest"


# initiate variable for session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 


st.set_page_config(page_title="PDF RAG Chatbot", layout="wide",initial_sidebar_state="expanded")
st.title("PDF RAG Assistant")

if not st.session_state.models_loaded:
    st.info("Model loading...")
    st.session_state.embeddings = lf.load_embeddings(EMBED_MODEL)
    st.session_state.llm = lf.initiate_llm_pipeline(LLM_MODEL)
    st.session_state.models_loaded = True
    st.success("Model is ready!")
    time.sleep(1)
    st.rerun()
    

with st.sidebar:
    st.subheader("📄Upload document")
    f = st.file_uploader("Choose",type ="pdf")
    if f and st.button("🔄 Xử lý PDF",use_container_width=True):
        with st.spinner("Processing..."):
            st.session_state.rag_chain,num_chunks =  lf.process_pdf(st.session_state.embeddings,LLM_MODEL,f)  
            st.session_state.pdf_name = f.name
            st.session_state.chat_history = []          
        st.success(f"✅ {num_chunks} chunks")
    st.info(f"📄{st.session_state.pdf_name}" if st.session_state.pdf_name else "📄 Chưa có tài liệu")
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_history=[]
    
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if st.session_state.rag_chain is None:
    st.info("🔄 Upload và xử lý PDF trước khi chat.")
    st.chat_input("Input your question...",disabled=True)
else:
    q = st.chat_input("Input your question...")
    if q:
        st.session_state.chat_history.append({"role":"user", "content":q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    output = st.session_state.rag_chain.invoke(q)
                    ans = output.split("Answer:")[1].strip() if "Answer:" in output else output.strip()
                    st.write(ans)
                    st.session_state.chat_history.append({"role":"assistant", "content": ans})
                except Exception as e:
                    st.error(f"Error when call model:{e}")



