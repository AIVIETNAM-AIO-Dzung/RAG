import pypdf
import chromadb
import ollama
import tempfile, os, time



def chunk_text(text:str, size:int=1000, overlap:int=200)->list:
    """Cắt text thành cấc đoạn nhỏ có độ dài tối đa 'size' ký tự, với 
    'overlap ký từ trùng lặp giữa 2 đoạn liên tiếp."""
    if text:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        chunks, cur = [], ""
        for p in paras:
            if len(cur) + len(p) + 1 <=size:
                cur += p +"\n"
            else:
                if cur:
                    chunks.append(cur.strip())
                cur = (cur[-overlap:] + p + "\n") if overlap else (p + "\n")
        
        if cur.strip():
            chunks.append(cur.strip())
        
        return chunks
    else:
        return None

def embed(texts:str, embed_model:str):
    """Chuyển danh sách chuỗi text thành danh sách vector."""
    return ollama.embed(model=embed_model, input=texts)["embeddings"]


def retrieve(query:str, collection, k:int=4):
    """Tìm k đoạn văn bản liên quan nhất với câu hỏi."""
    res = collection.query(
                            query_embeddings = embed([query]),
                            n_results = k
    )
    return res["documents"][0]


def rag(question:str, prompt:str, llm_model:str, collection, k:int=4):
    """Hàm rag chính: tìm context rồi hỏi LLM
    Arg:
        question: user's question,
        prompt: PROMPT,
        llm_model: LLM Model,
        collection: collection of embedded document,
        k: number of content that has similar meaning
    """
    context = "\n\n".join(retrieve(question,collection,k))
    resp = ollama.chat(
        model = llm_model,
        messages = [
            {'role': 'user',
             'content': prompt.format(context = context, question = question) 
             }
        ],
        options = {"temperature":0},
    )
    return resp["message"]["content"]


# Đọc file PFF
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix="pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)
    os.unlink(path)

    chunks = chunk_text(text,1000,200)
    if chunks:
        # Tạo vector database trong bộ nhớ
        client = chromadb.Client()
        collection = client.get_or_create_collection("rag")
        collection.add(ids=[f"{str(i)}_{int(time.time())}" for i in range(len(chunks))],
                    documents=chunks,
                    embeddings=embed(chunks)
                    )
        
        return collection, len(chunks)
    else:
        return None
    

