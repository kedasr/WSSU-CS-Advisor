"""
WSSU CS Graduate Advisor - Streamlit Web App
Deploy this to make your advisor accessible via web browser
"""

import streamlit as st
from pathlib import Path
from typing import List

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Shared constants and utilities
from advisor_core import (
    MAX_CREDITS,
    CHROMA_PERSIST_DIR,
    TRACK_REQUIREMENTS,
    extract_taken_courses_from_pdf,
    load_spring_courses,
    is_online_or_async,
    pick_courses_for_track,
)

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
        font-size: 2.5rem !important;
        color: #FFFFFF !important;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #C41E3A 0%, #1a1a1a 100%);
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

def generate_course_advice(track: str, selected: List[dict], credits: int) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, openai_api_key=st.secrets["OPENAI_API_KEY"])
    
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
