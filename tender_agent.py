import sys
import os
import streamlit as st
import PyPDF2
from langchain_groq import ChatGroq
import pandas as pd
import io

# --- 1. SYSTEM CONFIGURATION & FIXES ---
# Force UTF-8 to prevent Windows emoji crashes
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

st.set_page_config(
    page_title="TenderScout Pro - Brihaspathi",
    page_icon="🏢",
    layout="wide"
)

# --- 2. PROFESSIONAL HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    # Tries to load your logo; fails silently if missing
    try:
        st.image("Brihaspathi_logo.png", width=130)
    except:
        st.warning("Logo Missing")
with col2:
    st.title("TenderScout AI")
    st.markdown("##### Enterprise Bid Management & Decision Support System")

st.divider()

# --- 3. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    api_key_input = st.text_input("Groq API Key", type="password")
    if not api_key_input:
        st.warning("⚠️ Enter API Key to proceed.")
        st.stop()
    GROQ_API_KEY = api_key_input
    
    st.divider()
    
    # Company Profile Section
    with st.expander("🏢 Company Profile", expanded=False):
        st.info("AI will compare criteria against this profile.")
        default_profile = """
        Company: Brihaspathi Technologies Pvt Ltd.
        Turnover: 45 Crores INR.
        Experience: 12 Years in IT/Surveillance/Networking.
        Certifications: ISO 9001, ISO 27001, CMMI Level 3.
        Solvency: 15 Crores.
        Key Projects: Smart City Surveillance, Safe City, ZP School Connectivity.
        Legal Status: Not Blacklisted.
        """
        company_profile = st.text_area("Profile Data:", value=default_profile, height=250)
        
    st.divider()
    
    # PDF Upload
    st.subheader("📂 Tender Document")
    uploaded_file = st.file_uploader("Upload PDF / RFP", type="pdf")
    
    if st.button("Clear Session Memory"):
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.success("Memory Cleared.")

# --- 4. CORE TEXT EXTRACTION ---
def extract_pdf_content(pdf_file):
    """
    Smart Extraction: 
    - Reads first 40 pages (Synopsis/Eligibility/Timeline).
    - Reads last 20 pages (BoQ/Specs/Financials).
    """
    reader = PyPDF2.PdfReader(pdf_file)
    text_start = ""
    text_end = ""
    total_pages = len(reader.pages)
    
    # Extract Start (First 40)
    try:
        for i in range(min(40, total_pages)):
            content = reader.pages[i].extract_text()
            if content: text_start += content
    except: pass

    # Extract End (Last 20) - Only if doc is long enough
    try:
        if total_pages > 60:
            for i in range(total_pages - 20, total_pages):
                content = reader.pages[i].extract_text()
                if content: text_end += content
    except: pass
    
    return text_start + "\n...[MIDDLE SKIPPED]...\n" + text_end

# --- 5. AI LOGIC ENGINE ---
def analyze_tender(tender_text, profile, task_type):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    
    # Limit context size to prevent rate limits
    safe_text = tender_text[:25000] # Approx 6k tokens

    # MASTER PERSONA
    system_instruction = """
    You are a Senior Tender Consultant and Expert Technical Copywriter.
    Your tone is Professional, Precise, and Corporate.
    
    CORE RULES:
    1. SOURCE TRUTH: Use ONLY the provided Tender Text.
    2. NO GUESSING: If a data point (like EMD or Date) is not found, write "Not Mentioned".
    3. FORMATTING: Output strictly in the Markdown format requested.
    """

    # --- A. SYNOPSIS PROMPT ---
    if task_type == "Synopsis":
        prompt = f"""
        {system_instruction}
        TASK: Extract the 'Executive Summary' of the Tender.
        
        REQUIRED FIELDS:
        1. Name of the Department
        2. State (Infer from Department location if not explicit)
        3. Name of the Tender / Work Description
        4. Tender / RFP Number
        5. Date of Issuance (NIT Date)
        6. Contact Details (Name, Email, Phone of Authority)
        7. Key Dates (Start, End, Opening)
        8. Prebid Meeting Details (Date, Time, Venue/Link)
        9. EMD Value (Earnest Money Deposit) & Tender Fee
        10. Selection Criteria (e.g., L1, QCBS, 70:30, Reverse Auction)
        11. Mode of Bid Submission (Online/Offline/Hybrid)

        OUTPUT FORMAT:
        Use a Markdown Table with columns: | Parameter | Details |
        
        TENDER TEXT:
        {safe_text}
        """

    # --- B. ELIGIBILITY PROMPT ---
    elif task_type == "Eligibility":
        prompt = f"""
        {system_instruction}
        TASK: Compare Pre-Qualification (PQC) & Eligibility against Company Profile.
        
        MY PROFILE:
        {profile}
        
        OUTPUT FORMAT (Markdown Table):
        | S.No | Criteria Category | Tender Requirement (Exact Text) | My Profile Status | Compliance (Met/Not Met) |
        |---|---|---|---|---|
        | 1 | Turnover | [Extract] | [My Value] | [Met/Not Met] |
        | 2 | Experience | [Extract] | [My Value] | [Met/Not Met] |
        | 3 | Solvency | [Extract] | [My Value] | [Met/Not Met] |
        | 4 | Certifications | [Extract] | [My Value] | [Met/Not Met] |
        
        TENDER TEXT:
        {safe_text}
        """

    # --- C. TIMELINE PROMPT ---
    elif task_type == "Timeline":
        prompt = f"""
        {system_instruction}
        TASK: Extract the Project Execution Timeline / Delivery Schedule.
        
        INSTRUCTIONS:
        Look for clauses related to "Delivery Period", "Implementation Schedule", "Milestones", or "Go-Live".
        
        OUTPUT FORMAT (Markdown Table):
        | Milestone / Phase | Time Period (Days/Weeks from LOI) | Description |
        |---|---|---|
        | Delivery of Material | [e.g., T+4 Weeks] | [Details] |
        | Installation | [e.g., T+8 Weeks] | [Details] |
        
        TENDER TEXT:
        {safe_text}
        """

    # --- D. EQUIPMENT (BoM) PROMPT ---
    elif task_type == "Equipment":
        prompt = f"""
        {system_instruction}
        TASK: Extract the Bill of Materials (BoM) / Quantity Table.
        
        INSTRUCTIONS:
        Find the list of Hardware/Software items and their quantities.
        
        OUTPUT FORMAT (Markdown Table):
        | S.No | Item Name / Description | Quantity | Unit (Nos/Set/Lot) |
        |---|---|---|---|
        | 1 | [Item Name] | [Qty] | [Unit] |
        
        TENDER TEXT:
        {safe_text}
        """

    # --- E. SPECIFICATIONS PROMPT ---
    elif task_type == "Specs":
        prompt = f"""
        {system_instruction}
        TASK: Extract Key Technical Specifications.
        
        INSTRUCTIONS:
        For the major items identified (e.g., Cameras, Servers, Software), summarize the key technical specs required (Processor, Lens, ISO, Make).
        
        OUTPUT FORMAT (Bulleted List by Item):
        ### 1. [Item Name]
        * **Sensor/Processor:** [Spec]
        * **Resolution/Capacity:** [Spec]
        * **Key Feature:** [Spec]
        
        TENDER TEXT:
        {safe_text}
        """

    # --- F. RISKS PROMPT ---
    elif task_type == "Risks":
        prompt = f"""
        {system_instruction}
        TASK: Identify Commercial & Legal Risks.
        
        SCAN FOR:
        - Payment Terms > 60 Days
        - Penalty / Liquidated Damages > 10%
        - Unlimited Liability
        - Warranty Period > 5 Years
        - Ambiguous Scope
        
        OUTPUT FORMAT (Markdown Table):
        | Risk Category | Tender Clause / Condition | Risk Level (High/Med/Low) | Recommendation |
        |---|---|---|---|
        | Payment | [Extract Clause] | [Level] | [Advice] |
        
        TENDER TEXT:
        {safe_text}
        """

    # --- G. QUERIES PROMPT ---
    elif task_type == "Queries":
        prompt = f"""
        {system_instruction}
        TASK: Draft Pre-Bid Queries for ambiguous or restrictive clauses.
        
        OUTPUT FORMAT (Markdown Table):
        | S.No | RFP Section/Page | Existing Clause | Clarification Requested | Justification |
        |---|---|---|---|---|
        | 1 | [Ref] | [Clause Text] | [Question] | [Reason e.g. "To ensure wider participation"] |
        
        TENDER TEXT:
        {safe_text}
        """

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"API Error: {str(e)}"

def chat_engine(tender_text, question):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    safe_text = tender_text[:20000]
    prompt = f"""
    Role: Expert Tender Consultant.
    Context: Answer based strictly on the text provided.
    Text: {safe_text}
    User Question: {question}
    """
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Error: {str(e)}"

# --- 6. MAIN UI LOGIC ---

if uploaded_file:
    # Load Text Once
    if "tender_text" not in st.session_state:
        with st.spinner("📄 Analyzing Document Structure..."):
            st.session_state.tender_text = extract_pdf_content(uploaded_file)
            st.success("Document Indexed Successfully.")

    # Create Tabs A-H
    tabs = st.tabs([
        "A. Synopsis", 
        "B. Eligibility", 
        "C. Timeline", 
        "D. Equipment (BoM)", 
        "E. Specifications", 
        "F. Risks", 
        "G. Pre-Bid Queries", 
        "H. Chat Window"
    ])

    # --- TAB A: SYNOPSIS ---
    with tabs[0]:
        st.caption("Extracts the 11 Key Executive Summary Points.")
        if st.button("Generate Synopsis"):
            with st.spinner("Extracting Executive Summary..."):
                res = analyze_tender(st.session_state.tender_text, company_profile, "Synopsis")
                st.markdown(res)

    # --- TAB B: ELIGIBILITY ---
    with tabs[1]:
        st.caption("Compares Tender PQC vs Company Profile.")
        if st.button("Check Eligibility"):
            with st.spinner("Verifying Credentials..."):
                res = analyze_tender(st.session_state.tender_text, company_profile, "Eligibility")
                st.markdown(res)

    # --- TAB C: TIMELINE ---
    with tabs[2]:
        st.caption("Project Implementation Schedule.")
        if st.button("Extract Timeline"):
            with st.spinner("Analyzing Schedule..."):
                res = analyze_tender(st.session_state.tender_text, company_profile, "Timeline")
                st.markdown(res)

    # --- TAB D: EQUIPMENT ---
    with tabs[3]:
        st.caption("Bill of Materials & Quantity.")
        if st.button("Extract Equipment List"):
            with st.spinner("Scanning BoQ..."):
                res = analyze_tender(st.session_state.tender_text, company_profile, "Equipment")
                st.markdown(res)

    # --- TAB E: SPECIFICATIONS ---
    with tabs[4]:
        st.caption("Detailed Technical Specs for Major Items.")
        if st.button("Extract Specifications"):
            with st.spinner("Reading Technical Annexures..."):
                res = analyze_tender(st.session_state.tender_text, company_profile, "Specs")
                st.markdown(res)

    # --- TAB F: RISKS ---
    with tabs[5]:
        st.caption("Commercial & Legal Risk Scan.")
        if st.button("Identify Risks"):
            with st.spinner("Auditing Clauses..."):
                res = analyze_tender(st.session_state.tender_text, company_profile, "Risks")
                st.markdown(res)

    # --- TAB G: QUERIES ---
    with tabs[6]:
        st.caption("Drafts Questions for Authority.")
        if st.button("Draft Queries"):
            with st.spinner("Formulating Questions..."):
                res = analyze_tender(st.session_state.tender_text, company_profile, "Queries")
                st.markdown(res)

    # --- TAB H: CHAT ---
    with tabs[7]:
        st.subheader("Consultant Chat")
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        if user_input := st.chat_input("Ask a specific question..."):
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Consulting Document..."):
                ans = chat_engine(st.session_state.tender_text, user_input)
                st.chat_message("assistant").markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})

else:
    st.info("👈 Please upload a Tender PDF to begin.")
