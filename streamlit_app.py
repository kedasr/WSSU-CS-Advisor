"""
WSSU CS Graduate Advisor - Streamlit Web App
Deploy this to make your advisor accessible via web browser
"""

import streamlit as st
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import pandas as pd
from pypdf import PdfReader
import tempfile

# Load environment variables


# Constants
MAX_CREDITS = 9
CHROMA_PERSIST_DIR = "./chroma_db"

# Degree Requirements
TRACK_REQUIREMENTS = {
    "thesis": {
        "core": {"CST 5320", "CST 5322", "CST 6306"},
        "required": {"CST 6301", "CST 6302"},
        "other": {"CST 6601"},
    },
    "project": {
        "core": set(),
        "required": {"CST 5325", "CST 5328", "CST 6305"},
        "other": {"CST 6312"},
    },
    "exam": {
        "core": set(),
        "required": {"CST 5320", "CST 6301", "CST 6302", "CST 5325", "CST 5328", "CST 6305"},
        "other": {"CST 6000"},
    }
}

SHARED_ELECTIVES = {
    "CST 5101", "CST 5130", "CST 5301", "CST 5302", "CST 5303", "CST 5304", "CST 5305",
    "CST 5306", "CST 5307", "CST 5308", "CST 5309", "CST 5316", "CST 5323", "CST 5324",
    "CST 5326", "CST 5329", "CST 5330", "CST 5331", "CST 5332", "CST 5333", "CST 5334",
    "CST 5335", "CST 5340", "CST 5350", "CST 6000", "CST 6130", "CST 6303", "CST 6304",
    "CST 6307", "CST 6308", "CST 6309", "CST 6310", "CST 6311", "CST 6314", "CST 6320",
    "CST 7130"
}

# Page config
st.set_page_config(
    page_title="WSSU CS Graduate Advisor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #C41E3A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #1a1a1a 0%, #4a4a4a 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #C41E3A;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #9a1529;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #C41E3A;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_advisor' not in st.session_state:
    st.session_state.rag_advisor = None

class LangChainRAGAdvisor:
    """RAG-based advisor using LangChain."""
    
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        """Initialize the RAG system with LangChain components."""
        api_key = st.secrets["OPENAI_API_KEY"]
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)
        self.persist_directory = persist_directory
        
        if Path(persist_directory).exists():
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            self.vectorstore = None
    
    def answer_general_question(self, query: str) -> str:
        if not self.vectorstore:
            return "⚠️ Knowledge base not initialized."
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        template = """You are a friendly WSSU Computer Science graduate advisor.

Use the following context to answer the student's question about the MS in Computer Science 
and Information Technology program. Be brief (2-4 sentences), warm, and helpful.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(query)
    
    def answer_tuition_question(self, query: str) -> str:
        if not self.vectorstore:
            return "⚠️ Knowledge base not initialized."
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 1, "filter": {"source": "tuition_fees"}}
        )
        
        template = """You are a WSSU advisor helping with graduate tuition for Fall 2025 – Spring 2026.

Context:
{context}

Question: {question}

Answer (2-3 sentences):"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(query)

def extract_taken_courses_from_pdf(pdf_file) -> set[str]:
    """Extract completed courses from uploaded PDF."""
    reader = PdfReader(pdf_file)
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    return set(re.findall(r"\bCST\s+\d{4}\b", text))

def load_spring_courses(excel_path: str = "data/Spring2026_Courses.xlsx") -> pd.DataFrame:
    """Load Spring 2026 courses."""
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    df["Title"] = df["Title"].astype(str).str.strip()
    df["Subject"] = df["Subject"].astype(str).str.strip()
    df["Course Number"] = df["Course Number"].astype(str).str.strip()
    df["code"] = df["Subject"] + " " + df["Course Number"]
    df["Credits"] = pd.to_numeric(df["Credits"], errors="coerce").fillna(3).astype(int)
    df["Days"] = df["Meeting Days"].astype(str).str.strip()
    df["Time"] = df["Meeting Times"].astype(str).str.strip()
    
    if "Instructional Menthods." in df.columns:
        df["Instruction Mode"] = df["Instructional Menthods."].astype(str).str.strip()
    else:
        df["Instruction Mode"] = ""
    
    df["Status"] = df["Status"].astype(str).str.strip() if "Status" in df.columns else ""
    return df

def is_online_or_async(mode: str) -> bool:
    m = (mode or "").lower()
    return "online" in m or "async" in m or "asynchronous" in m

def time_conflicts(course_row: dict, selected_rows: list[dict]) -> bool:
    if is_online_or_async(course_row.get("Instruction Mode", "")):
        return False
    
    c_days = str(course_row.get("Days", "")).strip()
    c_time = str(course_row.get("Time", "")).strip()
    
    if not c_days or not c_time or c_days.lower() == "nan" or c_time.lower() == "nan":
        return False
    
    for s in selected_rows:
        if is_online_or_async(s.get("Instruction Mode", "")):
            continue
        if c_days == s.get("Days") and c_time == s.get("Time"):
            return True
    
    return False

def pick_courses_for_track(track: str, taken: set[str], spring_df: pd.DataFrame) -> Tuple[List[dict], int]:
    req = TRACK_REQUIREMENTS[track]
    needed = (req["core"] | req["required"] | req["other"]) - taken
    
    spring_df = spring_df[spring_df["Subject"].str.upper() == "CST"].copy()
    
    def status_rank(x: str) -> int:
        x = (x or "").lower()
        if "open" in x:
            return 0
        if "wait" in x:
            return 1
        return 2
    
    spring_df["status_rank"] = spring_df["Status"].apply(status_rank)
    spring_df = spring_df.sort_values(["status_rank", "code"])
    
    selected = []
    total_credits = 0
    
    def try_add(row: dict) -> bool:
        nonlocal total_credits
        cr = row.get("Credits", 3)
        if total_credits + cr > MAX_CREDITS:
            return False
        if time_conflicts(row, selected):
            return False
        selected.append(row)
        total_credits += cr
        return True
    
    for _, row in spring_df[spring_df["code"].isin(needed)].iterrows():
        if total_credits >= MAX_CREDITS:
            break
        try_add(row.to_dict())
    
    if total_credits < MAX_CREDITS:
        electives = spring_df[
            spring_df["code"].isin(SHARED_ELECTIVES)
            & (~spring_df["code"].isin(taken))
            & (~spring_df["code"].isin([s["code"] for s in selected]))
        ]
        for _, row in electives.iterrows():
            if total_credits >= MAX_CREDITS:
                break
            try_add(row.to_dict())
    
    return selected, total_credits

def generate_course_advice(track: str, selected: List[dict], credits: int) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    
    courses_text = "\n".join(
        f"- {c['code']} ({c['Credits']} credits): {c['Title']}"
        for c in selected
    )
    
    prompt = f"""You are a supportive WSSU CS graduate advisor.

Track: {track.title()}
Credits: {credits}/9

Recommended Courses:
{courses_text}

Provide brief, encouraging advice (3-4 sentences) about why these fit their {track} track 
and next steps.

Be warm and helpful!"""
    
    response = llm.invoke(prompt)
    return response.content

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 WSSU CS Graduate Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by LangChain RAG</p>', unsafe_allow_html=True)
    
    # Initialize RAG
    if st.session_state.rag_advisor is None:
        with st.spinner("Loading knowledge base..."):
            st.session_state.rag_advisor = LangChainRAGAdvisor()
    
    rag = st.session_state.rag_advisor
    
    if not rag.vectorstore:
        st.error("⚠️ Knowledge base not found. Please contact the administrator.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("📚 About")
    st.sidebar.info("""
    This AI advisor uses:
    - **LangChain** for RAG framework
    - **ChromaDB** for vector storage
    - **OpenAI** for embeddings & LLM
    
    Ask questions about:
    - Program requirements
    - Tuition & fees
    - Course recommendations
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Contact**")
    st.sidebar.markdown("📧 betheaj@wssu.edu")
    st.sidebar.markdown("📞 (336) 750-2478")
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["📚 General Questions", "💰 Tuition & Fees", "🎓 Course Advising"])
    
    # Tab 1: General Questions
    with tab1:
        st.header("Ask About the MCST Program")
        
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What are the thesis track requirements?",
            key="general_question"
        )
        
        if st.button("Get Answer", key="general_btn"):
            if question:
                with st.spinner("Searching knowledge base..."):
                    answer = rag.answer_general_question(question)
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**💡 Answer:**\n\n{answer}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a question.")
        
        # Example questions
        with st.expander("📝 Example Questions"):
            st.markdown("""
            - What are the three concentration options?
            - How many credits does the thesis track require?
            - What is the time limit to complete the program?
            - Can I transfer credits from another university?
            - What are the admission requirements?
            """)
    
    # Tab 2: Tuition
    with tab2:
        st.header("Tuition & Fees Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            residency = st.selectbox("Residency Status:", ["In-State", "Out-of-State"])
        
        with col2:
            credits = st.selectbox("Credit Hours:", list(range(1, 10)))
        
        if st.button("Calculate Tuition", key="tuition_btn"):
            query = f"How much for {residency.lower()} student taking {credits} credit hours?"
            with st.spinner("Calculating..."):
                answer = rag.answer_tuition_question(query)
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"**💰 Cost:**\n\n{answer}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Course Advising
    with tab3:
        st.header("Spring 2026 Course Recommendations")
        
        track = st.selectbox(
            "Your Track:",
            ["Thesis", "Project", "Exam"],
            key="track_select"
        )
        
        uploaded_file = st.file_uploader(
            "Upload Your Transcript (PDF):",
            type=['pdf'],
            help="Upload your unofficial transcript to get personalized recommendations"
        )
        
        if st.button("Get Course Recommendations", key="advising_btn"):
            if uploaded_file is None:
                st.warning("Please upload your transcript PDF.")
            else:
                with st.spinner("Analyzing your transcript..."):
                    # Extract courses
                    taken = extract_taken_courses_from_pdf(uploaded_file)
                    st.success(f"✓ Found {len(taken)} completed courses")
                    
                    # Load Spring courses
                    spring_df = load_spring_courses()
                    
                    # Get recommendations
                    selected, total_credits = pick_courses_for_track(
                        track.lower(),
                        taken,
                        spring_df
                    )
                    
                    if not selected:
                        st.warning("No suitable courses found for Spring 2026.")
                    else:
                        # Display recommendations
                        st.subheader(f"📋 Your Spring 2026 Plan ({total_credits}/{MAX_CREDITS} credits)")
                        
                        for i, course in enumerate(selected, 1):
                            mode_icon = "💻" if is_online_or_async(course.get("Instruction Mode", "")) else "🏫"
                            status_icon = "✅" if "open" in course.get("Status", "").lower() else "⏳"
                            
                            with st.expander(f"{i}. {course['code']} - {course['Title']}", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Credits:** {course['Credits']}")
                                    st.markdown(f"**Schedule:** {course['Days']} {course['Time']}")
                                with col2:
                                    st.markdown(f"**Mode:** {mode_icon} {course['Instruction Mode']}")
                                    st.markdown(f"**Status:** {status_icon} {course['Status']}")
                        
                        # Get AI advice
                        with st.spinner("Generating advisor recommendation..."):
                            advice = generate_course_advice(track.lower(), selected, total_credits)
                        
                        st.markdown("---")
                        st.subheader("💬 Advisor's Recommendation")
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown(advice)
                        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">🐏 Go Rams! | Winston-Salem State University</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
