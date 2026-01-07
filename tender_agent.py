import streamlit as st
import PyPDF2
from langchain_groq import ChatGroq
import pandas as pd
import io

# --- CONFIGURATION ---
st.set_page_config(
    page_title="TenderScout AI - Brihaspathi",
    page_icon="🏢",
    layout="wide"
)

# --- HEADER WITH LOGO ---
# Use columns to place the logo and title side-by-side
col1, col2 = st.columns([1, 4])
with col1:
    # Display the logo. Make sure 'Brihaspathi_logo.png' is in your folder.
    st.image("Brihaspathi_logo.png", width=150) 
with col2:
    st.title("TenderScout AI")
    st.markdown("#### Intelligent Bid/No-Bid Analysis & Extraction")
st.divider() # Add a clean line below the header

# --- SIDEBAR: INPUTS & SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings & Inputs")
    
    # --- API KEY ---
    api_key_input = st.text_input("Enter Groq API Key:", type="password")
    if not api_key_input:
        st.warning("⚠️ Please enter your API Key to proceed.")
        st.stop()
    GROQ_API_KEY = api_key_input
    st.divider()

    # --- COMPANY PROFILE (Cleaner layout with Expander) ---
    # Using an expander keeps the sidebar neat. Users can open it to see/edit details.
    with st.expander("1. Your Company Profile", expanded=False):
        st.info("Edit this to match your company details.")
        default_profile = """
        Company Name: Brihaspathi Technologies Pvt Ltd.
        Annual Turnover: 45 Crores INR.
        Years of Experience: 12 Years in IT/Surveillance.
        Certifications: ISO 9001, ISO 27001.
        Key Projects: Smart City Surveillance, ZP School Connectivity.
        Solvency Certificate: Available for 10 Cr.
        Blacklisted: No.
        Manpower: 150 Engineers on payroll.
        Locations: Head Office in Mumbai, Branch in Pune.
        """
        company_profile = st.text_area("Your Credentials:", value=default_profile, height=250)
    
    st.divider()
    
    # --- FILE UPLOADER ---
    st.header("2. Tender Document")
    uploaded_file = st.file_uploader("Upload PDF here", type="pdf")
    
    st.divider()
    if st.button("Clear Chat History"):
        if "messages" in st.session_state:
            st.session_state.messages = []

# --- HELPER FUNCTIONS (Same as before) ---
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
    except Exception as e:
        return None

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    try:
        # Read first 40 pages to stay within limits
        for i, page in enumerate(reader.pages):
            if i > 40: 
                break
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def analyze_tender(tender_text, my_profile, task_type):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    
    # Reduce text to 15,000 chars to avoid rate limits
    if len(tender_text) > 20000:
        safe_text = tender_text[:10000] + "\n...[MIDDLE SKIPPED]...\n" + tender_text[-5000:]
    else:
        safe_text = tender_text

    system_rules = """
    CRITICAL INSTRUCTIONS:
    1. Answer ONLY based on the "Tender Text" provided below.
    2. Do NOT use outside knowledge.
    3. If a specific detail (like a date or amount) is NOT in the text, write "NOT FOUND". Do not guess.
    4. Be exact. Do not round off numbers.
    """

    if task_type == "Synopsis":
        prompt = f"""
        {system_rules}
        Role: Precision Bid Manager.
        Task: Create a 'Tender At-a-Glance' Synopsis Table.
        REQUIRED FIELDS: Tender Reference No., Name of Work / Project Name, Name of Department / Authority, Ministry (if applicable), Tender Fee & EMD Amount, Project Estimated Cost, Bid Submission Start Date, Bid Submission End Date, Bid Opening Date.
        ELIGIBILITY CHECK (Pass/Fail): Turnover vs My Profile, Solvency vs My Profile, Experience vs My Profile, Technical Certifications vs My Profile.
        MY PROFILE: {my_profile}
        Tender Text: {safe_text}
        """
    elif task_type == "Eligibility":
        prompt = f"""
        {system_rules}
        Role: Strict Compliance Officer.
        Task: Create a Clause-by-Clause Compliance Matrix.
        Instructions: Extract eligibility criteria (Financial, Technical, Legal). Compare against 'MY PROFILE'.
        Output Markdown Table Columns: | Category | Exact Tender Requirement | My Profile Value | Status (PASS/FAIL/UNKNOWN) |
        MY PROFILE: {my_profile}
        Tender Text: {safe_text}
        """
    elif task_type == "BoM":
        prompt = f"""
        {system_rules}
        Role: Senior Estimation Engineer.
        Task: Extract the Bill of Materials (BoM).
        Instructions: Extract Item Name, Specs, and Quantity. If Quantity is not clear, write "1 Set".
        Output Markdown Table Columns: | S.No | Item Name | Detailed Specifications | Quantity |
        Tender Text: {safe_text}
        """
    elif task_type == "Risks":
        prompt = f"""
        {system_rules}
        Role: Legal Risk Analyst.
        Task: Identify High Risk Clauses.
        Instructions: Look for: Payment Terms > 90 days, Unlimited Liability, High Penalty (>10%).
        Tender Text: {safe_text}
        """
    elif task_type == "Queries":
        prompt = f"""
        {system_rules}
        Role: Bid Consultant.
        Task: Draft Pre-Bid Queries.
        Instructions: Identify ambiguous clauses. Create a formal query table.
        Tender Text: {safe_text}
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
    Role: Helpful Tender Assistant.
    Context: You are reading a specific Tender Document.
    Task: Answer the user's question strictly based on the Tender Text below.
    Tender Text: {safe_text}
    User Question: {user_question}
    """
    response = llm.invoke(prompt)
    return response.content

# --- MAIN INTERFACE ---
if uploaded_file is not None:
    if "tender_text" not in st.session_state:
        with st.spinner("📄 Reading PDF & Indexing..."):
            st.session_state.tender_text = extract_text_from_pdf(uploaded_file)
            st.success(f"✅ Document Loaded! ({len(st.session_state.tender_text)} characters extracted)")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📅 Synopsis", "⚖️ Eligibility", "🛠️ BoM (Export)", "🚩 Risk Analysis", "✉️ Pre-Bid Queries", "💬 Chat"
    ])

    # (The tab content remains the same as in your previous working code)
    with tab1:
        if st.button("Generate Synopsis"):
            with st.spinner("Analyzing..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Synopsis")
                st.markdown(result)
    with tab2:
        if st.button("Check Eligibility"):
            with st.spinner("Checking compliance..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Eligibility")
                st.markdown(result)
    with tab3:
        st.info("💡 Tip: Generates a table you can download as CSV.")
        if st.button("Extract BoM"):
            with st.spinner("Extracting Line Items..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "BoM")
                st.markdown(result)
                df = markdown_table_to_df(result)
                if df is not None:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download BoM as CSV", data=csv, file_name="tender_bom.csv", mime="text/csv")
                else:
                    st.warning("Could not auto-convert to CSV. Please copy the table manually.")
    with tab4:
        if st.button("Scan for Risks"):
            with st.spinner("Hunting for Red Flags..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Risks")
                st.markdown(result)
    with tab5:
        if st.button("Draft Queries"):
            with st.spinner("Drafting..."):
                result = analyze_tender(st.session_state.tender_text, company_profile, "Queries")
                st.markdown(result)
    with tab6:
        st.subheader("Ask specific questions")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Ask about penalties, payment terms, etc..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                try:
                    response = chat_with_tender(st.session_state.tender_text, prompt)
                    st.chat_message("assistant").markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("👈 Waiting for file upload...")