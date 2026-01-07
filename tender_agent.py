import sys
import os
import streamlit as st
import PyPDF2
from langchain_groq import ChatGroq
import pandas as pd
import io

# --- 1. SYSTEM FIXES ---
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# --- 2. APP CONFIGURATION ---
st.set_page_config(
    page_title="TenderScout AI - Brihaspathi",
    page_icon="🏢",
    layout="wide"
)

# --- 3. UI HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    try:
        st.image("Brihaspathi_logo.png", width=140)
    except:
        st.warning("Logo file not found.")
with col2:
    st.title("TenderScout AI")
    st.markdown("#### Intelligent Bid/No-Bid Decision System")

st.divider()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key_input = st.text_input("Enter Groq API Key:", type="password")
    if not api_key_input:
        st.warning("⚠️ Enter API Key to proceed.")
        st.stop()
    GROQ_API_KEY = api_key_input

    st.divider()

    with st.expander("🏢 Your Company Profile", expanded=False):
        default_profile = """
        Company Name: Brihaspathi Technologies Pvt Ltd.
        Annual Turnover: 45 Crores INR.
        Years of Experience: 12 Years in IT/Surveillance/Networking.
        Certifications: ISO 9001, ISO 27001, CMMI Level 3.
        Key Projects: Smart City Surveillance, ZP School Connectivity.
        Solvency Certificate: Available for 15 Cr.
        Blacklisted: No.
        """
        company_profile = st.text_area("Profile Data:", value=default_profile, height=300)
    
    st.divider()
    
    st.header("📂 Tender Document")
    uploaded_file = st.file_uploader("Upload PDF here", type="pdf")
    
    if st.button("Clear Chat Memory"):
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.success("Memory Cleared!")

# --- 5. CORE FUNCTIONS ---

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    try:
        # Increased to 60 pages to ensure we catch middle sections
        for i, page in enumerate(reader.pages):
            if i > 60: 
                break
            content = page.extract_text()
            if content:
                text += content
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def markdown_table_to_df(markdown_text):
    try:
        lines = markdown_text.split('\n')
        table_lines = [line for line in lines if '|' in line]
        if len(table_lines) < 3:
            return None
        table_str = "\n".join(table_lines)
        df = pd.read_csv(io.StringIO(table_str), sep="|", engine='python', skipinitialspace=True)
        df = df.dropna(axis=1, how='all')
        df.columns = [c.strip() for c in df.columns]
        return df
    except:
        return None

def analyze_tender(tender_text, my_profile, task_type):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    
    # --- SMART SLICING STRATEGY ---
    # Different tasks need different parts of the document.
    
    if task_type in ["Synopsis", "Eligibility"]:
        # PRIORITY: START OF DOC
        # PQC/Eligibility is almost always in the first 30% of the doc.
        # We take the first 14,000 chars and just 1,000 of the end (for dates).
        if len(tender_text) > 15000:
            safe_text = tender_text[:14000] + "\n...[End Section]...\n" + tender_text[-1000:]
        else:
            safe_text = tender_text

    elif task_type in ["BoM", "Risks"]:
        # PRIORITY: END OF DOC
        # BoM/Price Schedule is almost always in the last 30%.
        if len(tender_text) > 15000:
            safe_text = tender_text[:2000] + "\n...[Middle Skipped]...\n" + tender_text[-13000:]
        else:
            safe_text = tender_text
            
    else:
        # BALANCED (Queries/Chat)
        if len(tender_text) > 15000:
            safe_text = tender_text[:10000] + "\n...[Middle Skipped]...\n" + tender_text[-5000:]
        else:
            safe_text = tender_text

    # --- STRICT SYSTEM PROMPTS ---
    system_header = """
    You are a Strict Bid Compliance Officer. 
    RULES:
    1. QUOTE THE SOURCE: For every criteria found, you must mention the Section Name or Clause No if visible.
    2. SEPARATION OF CONCERNS: 
       - If asked for 'Eligibility', look ONLY for Pre-Qualification Criteria (Turnover, Exp, Legal). IGNORE technical specs.
       - If asked for 'BoM', look ONLY for Technical Specifications.
    3. NO GUESSING: If a criteria is missing, write "NOT FOUND".
    """

    if task_type == "Synopsis":
        prompt = f"""
        {system_header}
        TASK: Extract Key Tender Data.
        
        REQUIRED OUTPUT FORMAT (Markdown Table):
        | Category | Parameter | Tender Requirement (Quote Exact Text) | My Status (Pass/Fail/Info) |
        |---|---|---|---|
        | **Project** | Tender Ref No | [Extract] | Info |
        | **Project** | Authority Name | [Extract] | Info |
        | **Dates** | Bid Submission End Date | [Extract] | Info |
        | **Financial** | EMD Amount | [Extract] | Info |
        | **Eligibility** | Avg Annual Turnover | [Extract Financial Clause] | [Pass/Fail vs {my_profile}] |
        | **Eligibility** | Solvency Amount | [Extract Financial Clause] | [Pass/Fail vs {my_profile}] |
        | **Eligibility** | Past Experience | [Extract "Similar Work" Clause] | [Pass/Fail vs {my_profile}] |
        
        TENDER TEXT:
        {safe_text}
        """

    elif task_type == "Eligibility":
        prompt = f"""
        {system_header}
        TASK: Detailed PRE-QUALIFICATION (PQC) Matrix.
        
        INSTRUCTIONS:
        1. Scan for the section titled "Eligibility Criteria", "Pre-Qualification Criteria", or "Minimum Qualification".
        2. Extract ONLY the mandatory requirements for the BIDDER (Company).
        3. DO NOT extract "Scope of Work" or "Product Specifications" here.
        
        MY PROFILE:
        {my_profile}
        
        REQUIRED OUTPUT FORMAT (Markdown Table):
        | S.No | Criteria Type | Clause / Requirement | Section/Page Ref | My Status |
        |---|---|---|---|---|
        | 1 | Turnover | [e.g. "Avg Turnover of 5Cr in last 3 years"] | [e.g. Clause 4.1] | [Pass/Fail] |
        | 2 | Experience | [e.g. "3 similar projects of 10Cr"] | [e.g. Section III] | [Pass/Fail] |
        | 3 | Certifications | [e.g. "ISO 9001:2015 Mandatory"] | [e.g. Clause 5.2] | [Pass/Fail] |
        | 4 | Legal | [e.g. "Not Blacklisted"] | [e.g. Annexure 1] | [Pass/Fail] |
        
        TENDER TEXT:
        {safe_text}
        """

    elif task_type == "BoM":
        prompt = f"""
        {system_header}
        TASK: Extract Bill of Materials / Scope of Supply.
        
        INSTRUCTIONS:
        1. Scan for "Technical Specifications", "Bill of Quantities (BoQ)", or "Scope of Work".
        2. List the Hardware, Software, or Licenses required.
        
        REQUIRED OUTPUT FORMAT (Markdown Table):
        | S.No | Item Name | Specific Technical Requirements (Make/Model/Specs) | Quantity |
        |---|---|---|---|
        | 1 | [Item Name] | [Extract Key Specs] | [Qty] |
        
        TENDER TEXT:
        {safe_text}
        """

    elif task_type == "Risks":
        prompt = f"""
        {system_header}
        TASK: Identify Commercial Risks.
        
        REQUIRED OUTPUT FORMAT (Bulleted List):
        ### 🚨 Risk Assessment Report
        * **Payment Terms:** [Extract Clause]
        * **Penalty / LD:** [Extract Clause]
        * **PBG / Warranty:** [Extract Clause]
        
        TENDER TEXT:
        {safe_text}
        """

    elif task_type == "Queries":
        prompt = f"""
        {system_header}
        TASK: Draft Pre-Bid Queries.
        
        REQUIRED OUTPUT FORMAT (Markdown Table):
        | S.No | Clause Reference | Ambiguity / Issue | Drafted Query |
        |---|---|---|---|
        | 1 | [Clause No] | [Description] | [Query] |
        
        TENDER TEXT:
        {safe_text}
        """

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"API Error: {str(e)}"

def chat_with_tender(tender_text, user_question):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    safe_text = tender_text[:15000] 
    prompt = f"""
    Answer strictly based on the provided text.
    TENDER TEXT: {safe_text}
    QUESTION: {user_question}
    """
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Error: {str(e)}"

# --- 6. MAIN APP LOGIC ---

if uploaded_file is not None:
    if "tender_text" not in st.session_state:
        with st.spinner("📄 Reading PDF (First 60 Pages)..."):
            text = extract_text_from_pdf(uploaded_file)
            st.session_state.tender_text = text
            st.success(f"✅ Document Processed!")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📅 Synopsis", "⚖️ Eligibility", "🛠️ BoM (Export)", "🚩 Risks", "✉️ Queries", "💬 Chat"
    ])

    with tab1:
        if st.button("Generate Synopsis"):
            with st.spinner("Analyzing..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Synopsis")
                st.markdown(result)

    with tab2:
        st.info("ℹ️ Focuses on Pre-Qualification (Turnover, Exp, ISO). Ignores Tech Specs.")
        if st.button("Check Compliance"):
            with st.spinner("Scanning for PQC..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Eligibility")
                st.markdown(result)

    with tab3:
        if st.button("Extract Bill of Materials"):
            with st.spinner("Scanning for Line Items..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "BoM")
                st.markdown(result)
                df = markdown_table_to_df(result)
                if df is not None:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download CSV", data=csv, file_name="tender_bom.csv", mime="text/csv")

    with tab4:
        if st.button("Scan for Risks"):
            with st.spinner("Auditing..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Risks")
                st.markdown(result)

    with tab5:
        if st.button("Draft Pre-Bid Queries"):
            with st.spinner("Thinking..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Queries")
                st.markdown(result)

    with tab6:
        st.subheader("Ask the Document")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Ex: What are the payment terms?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                response = chat_with_tender(st.session_state.tender_text, prompt)
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👈 Upload Tender PDF to begin.")
