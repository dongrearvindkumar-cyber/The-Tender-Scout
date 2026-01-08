import sys
import os
import streamlit as st
import PyPDF2
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import io

# --- 1. SYSTEM CONFIGURATION ---
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

st.set_page_config(
    page_title="TenderScout Enterprise (RAG)",
    page_icon="🧠",
    layout="wide"
)

# --- 2. CACHED VECTOR DATABASE (The "High-End" Engine) ---
@st.cache_resource
def create_vector_db(text_content):
    """
    This function creates the 'Smart Index'.
    It runs only once per file upload (Cached).
    """
    # 1. Split text into manageable chunks (chunks of 1000 characters)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text_content)
    
    # 2. Convert chunks into Vectors (Numbers) using a lightweight AI model
    # We use 'all-MiniLM-L6-v2' which is fast and free
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Create the searchable database
    vector_db = FAISS.from_texts(chunks, embeddings)
    return vector_db

# --- 3. UI HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    try:
        st.image("Brihaspathi_logo.png", width=130)
    except:
        st.warning("Logo Missing")
with col2:
    st.title("TenderScout Enterprise")
    st.markdown("##### RAG-Powered Intelligent Bid Analysis System")

st.divider()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Config")
    
    api_key_input = st.text_input("Groq API Key", type="password")
    if not api_key_input:
        st.warning("⚠️ Enter API Key to activate Neural Engine.")
        st.stop()
    GROQ_API_KEY = api_key_input
    
    st.divider()
    
    with st.expander("🏢 Company Profile", expanded=False):
        default_profile = """
        Company: Brihaspathi Technologies Pvt Ltd.
        Turnover: 45 Crores INR.
        Experience: 12 Years in IT/Surveillance/Networking.
        Certifications: ISO 9001, ISO 27001, CMMI Level 3.
        Solvency: 15 Crores.
        """
        company_profile = st.text_area("Profile Data:", value=default_profile, height=200)
        
    st.subheader("📂 Document Ingestion")
    uploaded_file = st.file_uploader("Upload Tender PDF", type="pdf")

# --- 5. CORE FUNCTIONS ---

def extract_all_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    # In V4, we try to read MORE pages because RAG can handle it.
    # We limit to 100 pages to keep Cloud Memory safe.
    max_pages = min(len(reader.pages), 100)
    for i in range(max_pages):
        try:
            content = reader.pages[i].extract_text()
            if content: text += content
        except: continue
    return text

def retrieve_and_analyze(vector_db, profile, task_type):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    
    # --- SMART RETRIEVAL (The "RAG" Magic) ---
    # Instead of slicing strings, we ask the DB for relevant chunks.
    
    retrieval_query = ""
    if task_type == "Synopsis":
        retrieval_query = "tender number department name emd amount dates submission deadline authority contact"
    elif task_type == "Eligibility":
        retrieval_query = "minimum turnover experience solvency certifications bidder qualification iso criteria blacklisted"
    elif task_type == "Timeline":
        retrieval_query = "delivery period project timeline implementation schedule go-live weeks months execution"
    elif task_type == "Equipment":
        retrieval_query = "bill of materials quantity boq hardware specifications server camera software license"
    elif task_type == "Specs":
        retrieval_query = "technical specifications processor lens resolution sensor make model compliance"
    elif task_type == "Risks":
        retrieval_query = "penalty liquidated damages payment terms termination warranty liability indemnity"
    elif task_type == "Queries":
        retrieval_query = "clarification ambiguous conflict discrepancy bidder query"

    # Search the vector DB for the top 10 most relevant chunks (approx 8000 chars)
    search_results = vector_db.similarity_search(retrieval_query, k=10)
    
    # Combine the found chunks into one context text
    context_text = "\n\n".join([doc.page_content for doc in search_results])
    
    # --- PROMPT ENGINEERING ---
    system_instruction = f"""
    You are an Expert Tender Consultant.
    The user has asked for specific details.
    You have been provided with RELEVANT EXCERPTS from the document below.
    
    RULES:
    1. Answer ONLY using the provided Excerpts.
    2. If the excerpts don't contain the answer, state "Not found in analyzed sections".
    """

    if task_type == "Synopsis":
        prompt = f"""
        {system_instruction}
        TASK: Extract Executive Summary.
        
        REQUIRED FIELDS:
        1. Dept Name & State
        2. Tender Name & Ref No
        3. NIT Date & Bid Deadlines
        4. EMD & Fee
        5. Submission Mode
        
        EXCERPTS:
        {context_text}
        """
    elif task_type == "Eligibility":
        prompt = f"""
        {system_instruction}
        TASK: Compare PQC against Profile.
        MY PROFILE: {profile}
        
        OUTPUT TABLE: | Criteria | Requirement | My Status | Compliance |
        
        EXCERPTS:
        {context_text}
        """
    elif task_type == "Equipment":
        prompt = f"""
        {system_instruction}
        TASK: Extract BoM / Quantities.
        OUTPUT TABLE: | S.No | Item | Qty | Unit |
        
        EXCERPTS:
        {context_text}
        """
    elif task_type == "Risks":
        prompt = f"""
        {system_instruction}
        TASK: Identify Commercial Risks (Payment, LD, Warranty).
        OUTPUT TABLE: | Risk Area | Clause | Severity |
        
        EXCERPTS:
        {context_text}
        """
    # (Other tasks follow similar pattern...)
    else:
        prompt = f"""
        {system_instruction}
        TASK: Analyze the text for {task_type}.
        EXCERPTS: {context_text}
        """

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"API Error: {str(e)}"

def chat_rag_engine(vector_db, question):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    
    # Retrieve top 5 chunks for chat
    docs = vector_db.similarity_search(question, k=5)
    context = "\n".join([d.page_content for d in docs])
    
    prompt = f"""
    Answer the user question using ONLY the context below.
    CONTEXT: {context}
    QUESTION: {question}
    """
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Error: {str(e)}"

# --- 6. MAIN LOGIC ---

if uploaded_file:
    # 1. READ TEXT (Raw)
    if "full_text" not in st.session_state:
        with st.spinner("📄 Reading PDF..."):
            st.session_state.full_text = extract_all_text(uploaded_file)
            st.success("PDF Read.")
            
    # 2. BUILD BRAIN (Vector DB) - This happens once
    if st.session_state.full_text:
        with st.spinner("🧠 Building Neural Index (This may take 20s)..."):
            # The function is cached, so it's fast after first run
            vector_db = create_vector_db(st.session_state.full_text)
            st.success("Neural Index Ready!")

    # 3. TABS
    tabs = st.tabs(["A. Synopsis", "B. Eligibility", "C. Timeline", "D. Equipment", "E. Risks", "H. Chat"])

    # Just implementing key tabs for brevity in V4, logic applies to all
    with tabs[0]:
        if st.button("Generate Synopsis"):
            res = retrieve_and_analyze(vector_db, company_profile, "Synopsis")
            st.markdown(res)
            
    with tabs[1]:
        if st.button("Check Eligibility"):
            res = retrieve_and_analyze(vector_db, company_profile, "Eligibility")
            st.markdown(res)

    with tabs[3]:
        if st.button("Extract BoM"):
            res = retrieve_and_analyze(vector_db, company_profile, "Equipment")
            st.markdown(res)

    with tabs[4]:
        if st.button("Scan Risks"):
            res = retrieve_and_analyze(vector_db, company_profile, "Risks")
            st.markdown(res)

    with tabs[5]: # Chat
        st.subheader("Deep-Search Chat")
        if "chat_history" not in st.session_state: st.session_state.chat_history = []
        
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if query := st.chat_input("Ask deeper questions (e.g. 'What is the penalty for delay?')"):
            st.chat_message("user").write(query)
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            with st.spinner("Retrieving relevant clauses..."):
                ans = chat_rag_engine(vector_db, query)
                st.chat_message("assistant").write(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                
else:
    st.info("👈 Upload PDF to initialize the Neural Engine.")
