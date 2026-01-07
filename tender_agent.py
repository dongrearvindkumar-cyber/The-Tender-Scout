import sys
import os
import streamlit as st
import PyPDF2
from langchain_groq import ChatGroq
import pandas as pd
import io

# --- 1. SYSTEM FIXES (CRITICAL) ---
# This fixes the "UnicodeEncodeError" / Emoji crash on Windows & Logs
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# --- 2. APP CONFIGURATION ---
st.set_page_config(
    page_title="TenderScout AI - Brihaspathi",
    page_icon="🏢",
    layout="wide"
)

# --- 3. UI HEADER ---
# Columns to place Logo next to Title
col1, col2 = st.columns([1, 5])
with col1:
    # Ensure 'Brihaspathi_logo.png' is inside your GitHub repository
    # If the file is missing, it will just show a broken image icon (won't crash)
    try:
        st.image("Brihaspathi_logo.png", width=140)
    except:
        st.warning("Logo file not found.")
with col2:
    st.title("TenderScout AI")
    st.markdown("#### Intelligent Bid/No-Bid Decision System")

st.divider()

# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Secure API Key Entry
    api_key_input = st.text_input("Enter Groq API Key:", type="password")
    
    # Stop the app if no key is provided
    if not api_key_input:
        st.warning("⚠️ Please enter your API Key to proceed.")
        st.stop()
        
    GROQ_API_KEY = api_key_input

    st.divider()

    # Company Profile (Collapsible for cleaner look)
    with st.expander("🏢 Your Company Profile", expanded=False):
        st.info("Edit this to match your exact credentials.")
        default_profile = """
        Company Name: Brihaspathi Technologies Pvt Ltd.
        Annual Turnover: 200 Crores INR.
        Years of Experience: 18 Years in IT/Surveillance/Networking.
        Certifications: ISO 9001, ISO 27001, CMMI Level 3.
        Key Projects: Smart City Surveillance, ZP School Connectivity, Safe City Project, Border Security Force, Maharashtra State Road Transport Corporation.
        Solvency Certificate: Available for 15 Cr.
        Blacklisted: No.
        Manpower: 150+ Engineers on payroll.
        Locations: Head Office in Hyderabad, Branches Pan-India.
        """
        company_profile = st.text_area("Profile Data:", value=default_profile, height=300)
    
    st.divider()
    
    # File Uploader
    st.header("📂 Tender Document")
    uploaded_file = st.file_uploader("Upload PDF here", type="pdf")
    
    # Chat History Clear Button
    if st.button("Clear Chat Memory"):
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.success("Memory Cleared!")

# --- 5. CORE FUNCTIONS ---

def extract_text_from_pdf(pdf_file):
    """
    Reads PDF text safely. Limits to first 150 pages to prevent memory issues.
    """
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    try:
        for i, page in enumerate(reader.pages):
            if i > 150: # Limit to 50 pages
                break
            content = page.extract_text()
            if content:
                text += content
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def markdown_table_to_df(markdown_text):
    """
    Converts AI's Markdown table into a Pandas DataFrame for Excel export.
    """
    try:
        lines = markdown_text.split('\n')
        # Filter lines that look like table rows (contain pipes |)
        table_lines = [line for line in lines if '|' in line]
        if len(table_lines) < 3:
            return None
        
        # Join and read into Pandas
        table_str = "\n".join(table_lines)
        df = pd.read_csv(io.StringIO(table_str), sep="|", engine='python', skipinitialspace=True)
        
        # Clean up empty columns from side pipes
        df = df.dropna(axis=1, how='all')
        df.columns = [c.strip() for c in df.columns]
        return df
    except:
        return None

def analyze_tender(tender_text, my_profile, task_type):
    """
    The Brain: Sends strict prompts to Llama 3.1 via Groq.
    """
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    
    # --- CONTEXT SLICING (Fixes Rate Limit) ---
    # We take the Start (Dates/Terms) and End (BoM/Specs)
    # Total limit ~15,000 chars (approx 3.5k tokens) to stay safe within 6k limit
    if len(tender_text) > 20000:
        safe_text = tender_text[:10000] + "\n\n...[MIDDLE CONTENT SKIPPED]...\n\n" + tender_text[-5000:]
    else:
        safe_text = tender_text

    # --- STRICT SYSTEM PROMPTS (Fixes Formatting & Hallucinations) ---
    system_header = """
    You are a specialized Bid Analysis Engine. 
    SYSTEM RULES:
    1. NO HALLUCINATIONS: If a value (date, amount, clause) is not in the text, write "NOT FOUND". Do not guess.
    2. STRICT FORMATTING: Output ONLY the Markdown table requested. Do not write introductory text like "Here is the analysis".
    3. DATA SOURCE: Use only the provided 'TENDER TEXT' below.
    """

    if task_type == "Synopsis":
        prompt = f"""
        {system_header}
        TASK: Extract Key Data & Check Eligibility.
        
        MY PROFILE:
        {my_profile}
        
        REQUIRED OUTPUT FORMAT (Markdown Table Only):
        | Category | Parameter | Tender Requirement (Exact Text) | My Status (Pass/Fail/Info) | Remark/Gap |
        |---|---|---|---|---|
        | **Project Info** | Tender Reference No | [Extract] | Info | - |
        | **Project Info** | Authority / Dept Name | [Extract] | Info | - |
        | **Dates** | Bid Submission End Date | [Extract] | Info | - |
        | **Dates** | Bid Opening Date | [Extract] | Info | - |
        | **Financial** | Estimated Project Cost | [Extract] | Info | - |
        | **Financial** | EMD Amount | [Extract] | Info | - |
        | **Eligibility** | Annual Turnover Required | [Extract Value] | [Pass/Fail] | [Compare with Profile] |
        | **Eligibility** | Solvency Required | [Extract Value] | [Pass/Fail] | [Compare with Profile] |
        | **Eligibility** | Past Experience | [Extract Criteria] | [Pass/Fail] | [Compare with Profile] |
        | **Eligibility** | Certifications (ISO/CMMI) | [Extract List] | [Pass/Fail] | [Compare with Profile] |
        
        TENDER TEXT:
        {safe_text}
        """

    elif task_type == "Eligibility":
        prompt = f"""
        {system_header}
        TASK: Detailed Clause-by-Clause Compliance Matrix.
        INSTRUCTION: Extract every technical, financial, and legal mandatory requirement.
        
        MY PROFILE:
        {my_profile}
        
        REQUIRED OUTPUT FORMAT (Markdown Table Only):
        | S.No | Category | Tender Requirement (Exact Clause) | My Profile / Value | Status (PASS/FAIL) |
        |---|---|---|---|---|
        | 1 | Financial | [e.g. Min Turnover 50Cr] | [e.g. 45Cr] | FAIL |
        | 2 | Technical | [e.g. CMMI Lvl 5] | [e.g. CMMI Lvl 3] | FAIL |
        | 3 | Legal | [e.g. Not Blacklisted] | [Not Blacklisted] | PASS |
        
        TENDER TEXT:
        {safe_text}
        """

    elif task_type == "BoM":
        prompt = f"""
        {system_header}
        TASK: Extract Bill of Materials (BoM).
        INSTRUCTION: List all hardware/software items found in Scope or Technical Specs.
        
        REQUIRED OUTPUT FORMAT (Markdown Table Only):
        | S.No | Item Name | Detailed Specifications (Make/Model/Specs) | Quantity |
        |---|---|---|---|
        | 1 | [Item Name] | [Extract Key Specs] | [Qty] |
        | 2 | [Item Name] | [Extract Key Specs] | [Qty] |
        
        TENDER TEXT:
        {safe_text}
        """

    elif task_type == "Risks":
        prompt = f"""
        {system_header}
        TASK: Identify Commercial Risks.
        INSTRUCTION: Look for Payment Terms (>90 days), High Penalty (>10%), Liquidated Damages, Unlimited Liability.
        
        REQUIRED OUTPUT FORMAT (Bulleted List):
        ### 🚨 Risk Assessment Report
        * **Payment Terms:** [Extract Clause] - [Risk: High/Medium/Low]
        * **Penalty / LD:** [Extract Clause]
        * **Warranty:** [Extract Period]
        * **Termination:** [Extract Clause]
        
        TENDER TEXT:
        {safe_text}
        """

    elif task_type == "Queries":
        prompt = f"""
        {system_header}
        TASK: Draft Pre-Bid Queries.
        INSTRUCTION: Find ambiguous terms (e.g. "Reputed make", "Standard warranty") and ask for clarification.
        
        REQUIRED OUTPUT FORMAT (Markdown Table Only):
        | S.No | Clause Reference | Ambiguity / Issue | Drafted Query for Authority |
        |---|---|---|---|
        | 1 | [e.g. Clause 4.1] | [Explain issue] | [Draft polite query] |
        
        TENDER TEXT:
        {safe_text}
        """

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"API Error (Likely Rate Limit): {str(e)}"

def chat_with_tender(tender_text, user_question):
    """
    Free-form chat. Uses a smaller context window (15k) to stay fast.
    """
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    safe_text = tender_text[:15000] 
    
    prompt = f"""
    You are a helpful Tender Assistant.
    Answer the user question strictly based on the text below.
    
    TENDER TEXT:
    {safe_text}
    
    QUESTION: {user_question}
    """
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"Error: {str(e)}"

# --- 6. MAIN APP LOGIC ---

if uploaded_file is not None:
    # 1. Read File (Only once per upload)
    if "tender_text" not in st.session_state:
        with st.spinner("📄 Reading and Indexing Document..."):
            text = extract_text_from_pdf(uploaded_file)
            st.session_state.tender_text = text
            st.success(f"✅ Document Processed! ({len(text)} characters extracted)")

    # 2. Define Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📅 Synopsis", 
        "⚖️ Eligibility", 
        "🛠️ BoM (Export)", 
        "🚩 Risks", 
        "✉️ Queries", 
        "💬 Chat"
    ])

    # 3. Tab Functionality
    
    # --- SYNOPSIS TAB ---
    with tab1:
        st.caption("Generates a 'Bid/No-Bid' Summary Table.")
        if st.button("Generate Synopsis"):
            with st.spinner("Analyzing Tender Data..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Synopsis")
                st.markdown(result)

    # --- ELIGIBILITY TAB ---
    with tab2:
        st.caption("Checks every clause against your Company Profile.")
        if st.button("Check Compliance"):
            with st.spinner("Verifying Credentials..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Eligibility")
                st.markdown(result)

    # --- BOM TAB (With CSV Export) ---
    with tab3:
        st.caption("Extracts Line Items for Cost Estimation.")
        if st.button("Extract Bill of Materials"):
            with st.spinner("Scanning for Hardware/Software..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "BoM")
                st.markdown(result)
                
                # Convert to CSV button
                df = markdown_table_to_df(result)
                if df is not None:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download BoM as CSV",
                        data=csv,
                        file_name="tender_bom.csv",
                        mime="text/csv"
                    )

    # --- RISK TAB ---
    with tab4:
        st.caption("Highlights Hidden Penalties & Payment Issues.")
        if st.button("Scan for Risks"):
            with st.spinner("Auditing Legal Clauses..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Risks")
                st.markdown(result)

    # --- QUERIES TAB ---
    with tab5:
        st.caption("Drafts official questions for the Pre-Bid Meeting.")
        if st.button("Draft Pre-Bid Queries"):
            with st.spinner("Finding Ambiguities..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Queries")
                st.markdown(result)

    # --- CHAT TAB ---
    with tab6:
        st.subheader("Ask the Document")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ex: What are the payment terms?"):
            # User Message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # AI Response
            with st.spinner("Thinking..."):
                response = chat_with_tender(st.session_state.tender_text, prompt)
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # Landing Page State
    st.info("👈 Please upload a Tender PDF to begin analysis.")
    st.markdown("""
    ### How to use:
    1. Enter your **Groq API Key** in the sidebar.
    2. Upload a **Tender PDF**.
    3. Click the tabs above to extract data.
    """)
