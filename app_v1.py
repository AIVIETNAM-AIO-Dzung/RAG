import ollama_function as fu
import streamlit as st



LLM_MODEL = "vicuna:7b-v1.5-q5_1"
EMBED_MODEL = "sentence-transformers/qwen3-embedding:latest"
PROMPT = """
                Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
                Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
                Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

                Ngữ cảnh:{context}
                
                Câu hỏi: {question}

                Trả lời:
            """

for k,v in {"collection":None, "pdf_name":"", "chat_history":[]}.items():
    st.session_state.setdefault(k,v)

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide",initial_sidebar_state="expanded")
st.title("PDF RAG Assistant")

with st.sidebar:
    st.subheader("📄Upload document")
    f = st.file_uploader("Choose",type ="pdf")
    if f and st.button("🔄 Xử lý PDF",use_container_width=True):
        with st.spinner("Processing..."):
            st.session_state.collection, n =  fu.process_pdf(f,EMBED_MODEL)
            st.session_state.pdf_name = f.name
            st.session_state.chat_history = []
        st.success(f"✅ {n} chunks")
    st.info(f"📄{st.session_state.pdf_name}" if st.session_state.pdf_name else "📄 Chưa có tài liệu")
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_history=[]
    
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if st.session_state.collection is None:
    st.info("🔄 Upload và xử lý PDF trước khi chat.")
    st.chat_input("Input ypur question...",disabled=True)
else:
    q = st.chat_input("Input your question...")
    if q:
        st.session_state.chat_history.append({"role":"user", "content":q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = fu.rag(q,PROMPT,LLM_MODEL,st.session_state.collection,4)
                st.write(ans)
            st.session_state.chat_history.append({"role":"assistant", "content": ans})



