"""
WSSU CS Graduate Advisor using LangChain RAG
Simplified version compatible with latest LangChain
"""

import os
from pathlib import Path
from typing import List, Tuple

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import pandas as pd

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required when using Streamlit secrets

# Shared constants and utilities
from advisor_core import (
    MAX_CREDITS,
    CHROMA_PERSIST_DIR,
    TRACK_REQUIREMENTS,
    SHARED_ELECTIVES,
    extract_taken_courses_from_pdf,
    load_spring_courses,
    is_online_or_async,
    pick_courses_for_track,
)

class LangChainRAGAdvisor:
    """
    RAG-based advisor using LangChain for document retrieval and Q&A.
    """
    
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        """Initialize the RAG system with LangChain components."""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.persist_directory = persist_directory
        
        # Load or create vector store
        if Path(persist_directory).exists():
            print("✓ Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("⚠️  Vector store not found. Please run prepare_documents() first.")
            self.vectorstore = None
    
    def prepare_documents(self) -> None:
        """
        Prepare and index all documents into the vector store.
        This is the RAG setup - run once to build knowledge base.
        """
        print("\n" + "=" * 60)
        print("Building RAG Knowledge Base with LangChain")
        print("=" * 60)
        
        documents = []
        
        # 1. General Department Info
        print("\n📚 Processing general department information...")
        with open("data/general_dept.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        # Use LangChain's RecursiveCharacterTextSplitter for RAG
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        general_chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(general_chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={"source": "general_department_info", "chunk": i}
            ))
        print(f"  ✓ Created {len(general_chunks)} chunks")
        
        # 2. Tuition Information
        print("\n💰 Processing tuition information...")
        tuition_text = """
GRADUATE TUITION & FEES FALL 2025 - SPRING 2026

IN-STATE TUITION AND FEES (per semester):
1 credit hour: $373.80 total tuition & fees
2 credit hours: $737.62 total tuition & fees
3 credit hours: $1,101.46 total tuition & fees
4 credit hours: $1,465.12 total tuition & fees
5 credit hours: $1,909.10 total tuition & fees
6 credit hours: $2,192.92 total tuition & fees
7 credit hours: $2,556.57 total tuition & fees
8 credit hours: $2,920.48 total tuition & fees
9 credit hours: $3,284.50 total tuition & fees

OUT-OF-STATE TUITION AND FEES (per semester):
1 credit hour: $959.06 total tuition & fees
2 credit hours: $1,908.12 total tuition & fees
3 credit hours: $2,857.18 total tuition & fees
4 credit hours: $3,806.12 total tuition & fees
5 credit hours: $4,755.30 total tuition & fees
6 credit hours: $5,704.36 total tuition & fees
7 credit hours: $6,653.32 total tuition & fees
8 credit hours: $7,469.48 total tuition & fees
9 credit hours: $8,552.00 total tuition & fees

OPTIONAL FEES:
- Book Rental fee: $33.33 per credit hour (option to opt out)
- Health Insurance fee: $1,396.54 for 6+ credit hours (option to waive)
  Deadlines: September 11th (Fall), January 31st (Spring)

Students can choose between Premium Plan and Value Plan for health insurance.
All students are defaulted to the VALUE Plan.
"""
        documents.append(Document(
            page_content=tuition_text,
            metadata={"source": "tuition_fees", "chunk": 0}
        ))
        print("  ✓ Added tuition information")
        
        # 3. Degree Requirements
        print("\n🎓 Processing degree requirements...")
        degree_req_text = """
DEGREE REQUIREMENTS FOR MCST

The MS in Computer Science and Information Technology requires 30-33 credit hours.
Students must maintain GPA of 3.0. No more than two grades of "C" allowed.

THESIS CONCENTRATION (30 credit hours):
Core Courses (9 credits):
- CST 5320: Design and Analysis of Algorithms Methods
- CST 5322: Advanced Software Engineering
- CST 6306: Advanced Database Management Systems

Required Courses (9 credits):
- CST 6301: Advanced Computer Architecture
- CST 6302: Programming Languages and Compilers

Electives (6 credits): From shared elective list

Thesis Research (6 credits):
- CST 6601: Master's Thesis Research (written report and presentation required)

PROJECT CONCENTRATION (33 credit hours):
Core Courses: None

Required Courses (9 credits):
- CST 5325: Electronic Commerce Technology
- CST 5328: Computer Networks
- CST 6305: Internet Technology Systems

Electives (12 credits): From shared elective list

Master's Project (3 credits):
- CST 6312: Master's Project (written report and presentation required)

EXAM CONCENTRATION (33 credit hours):
Core Courses: None

Required Courses (9 credits): Choose from either:
  Thesis track: CST 5320, CST 6301, CST 6302
  OR Project track: CST 5325, CST 5328, CST 6305

Electives (15 credits): From shared elective list

Comprehensive Exam (0 credits):
- CST 6000: Comprehensive exam on core and required courses

SHARED ELECTIVES (all concentrations):
CST 5101, CST 5130, CST 5301, CST 5302, CST 5303, CST 5304, CST 5305, CST 5306,
CST 5307, CST 5308, CST 5309, CST 5316, CST 5323, CST 5324, CST 5326, CST 5329,
CST 5330, CST 5331, CST 5332, CST 5333, CST 5334, CST 5335, CST 5340, CST 5350,
CST 6130, CST 6303, CST 6304, CST 6307, CST 6308, CST 6309, CST 6310, CST 6311,
CST 6314, CST 6320, CST 7130

TIME LIMITS:
- Full-time completion: 2 years
- Maximum allowed: 6 years

TRANSFER CREDITS:
- Up to 6 semester hours allowed
- Grade "B" or better required
- Request at admission
"""
        
        degree_chunks = text_splitter.split_text(degree_req_text)
        for i, chunk in enumerate(degree_chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={"source": "degree_requirements", "chunk": i}
            ))
        print(f"  ✓ Created {len(degree_chunks)} chunks")
        
        # 4. Spring 2026 Courses
        print("\n📅 Processing Spring 2026 courses...")
        df = pd.read_excel("data/Spring2026_Courses.xlsx")
        
        # Clean column names (remove trailing spaces)
        df.columns = df.columns.str.strip()
        
        for _, row in df.iterrows():
            course_text = f"""
Spring 2026 Course: {row['Subject']} {row['Course Number']} - {row['Title']}
Credits: {row['Credits']}
CRN: {row['CRN']}
Status: {row['Status']}
Meeting Days: {row['Meeting Days']}
Meeting Times: {row['Meeting Times']}
Instructional Method: {row.get('Instructional Menthods.', row.get('Instructor', 'N/A'))}
Instructor: {row.get('Instructor', 'TBA')}
"""
            documents.append(Document(
                page_content=course_text.strip(),
                metadata={
                    "source": "spring_2026_courses",
                    "course_code": f"{row['Subject']} {row['Course Number']}",
                    "crn": str(row['CRN'])
                }
            ))
        print(f"  ✓ Indexed {len(df)} courses")
        
        # Create vector store using LangChain
        print("\n🔍 Creating vector store with embeddings...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"\n✅ RAG Knowledge Base Ready!")
        print(f"   Total documents indexed: {len(documents)}")
        print(f"   Stored in: {self.persist_directory}")
    
    def answer_general_question(self, query: str) -> str:
        """Answer general departmental questions using RAG."""
        if not self.vectorstore:
            return "⚠️  Knowledge base not initialized. Please run setup first."
        
        # RAG: Retrieve relevant documents
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Create RAG chain using LangChain LCEL (LangChain Expression Language)
        template = """You are a friendly WSSU Computer Science graduate advisor.

Use the following context to answer the student's question about the MS in Computer Science 
and Information Technology program. Be brief (2-4 sentences), warm, and helpful.

If the answer isn't in the context, say you don't have that information and suggest 
contacting the department at betheaj@wssu.edu or (336)750-2478.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Build RAG chain
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
        """Answer tuition and fees questions using RAG."""
        if not self.vectorstore:
            return "⚠️  Knowledge base not initialized. Please run setup first."
        
        # RAG: Retrieve from tuition documents only
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 1,
                "filter": {"source": "tuition_fees"}
            }
        )
        
        template = """You are a WSSU advisor helping with graduate tuition for Fall 2025 – Spring 2026.

Context shows tuition tables for In-State and Out-of-State students (1-9 credit hours).

When asked about costs:
1. Identify residency (in-state or out-of-state)
2. Find credit hours
3. State the total tuition & fees
4. Mention optional fees if relevant

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

# =========================
# Course Advice Generation
# =========================

def generate_course_advice(track: str, selected: List[dict], credits: int) -> str:
    """Generate personalized course advice using LLM."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    
    courses_text = "\n".join(
        f"- {c['code']} ({c['Credits']} credits): {c['Title']} | {c['Days']} {c['Time']} | {c['Instruction Mode']} | {c['Status']}"
        for c in selected
    )
    
    prompt = f"""You are a supportive WSSU CS graduate advisor.

Track: {track.title()}
Credits: {credits}/9

Recommended Courses:
{courses_text}

Provide brief, encouraging advice (3-4 sentences) about:
- Why these fit their {track} track
- Important scheduling notes
- Next steps (register early, check prerequisites)

Be warm and helpful!"""
    
    response = llm.invoke(prompt)
    return response.content

# =========================
# Main Interface
# =========================

def main():
    """Main interactive advisor interface."""
    print("\n" + "=" * 60)
    print("        WSSU CS Graduate Advisor")
    print("        Powered by LangChain RAG")
    print("=" * 60)
    
    # Initialize RAG system
    rag = LangChainRAGAdvisor()
    
    if not rag.vectorstore:
        print("\n⚠️  Setting up knowledge base for first time...")
        response = input("Would you like to set it up now? (y/n): ").strip().lower()
        if response == 'y':
            rag.prepare_documents()
            rag = LangChainRAGAdvisor()  # Reload
        else:
            print("Please run prepare_documents() first.")
            return
    
    print("\nI can help you with:")
    print("  1. General questions about the MCST program")
    print("  2. Tuition and fees information")
    print("  3. Course advising for Spring 2026")
    print("  4. Exit")
    
    while True:
        print("\n" + "-" * 60)
        choice = input("\nWhat would you like help with? (1/2/3/4): ").strip()
        
        if choice == "1":
            print("\n📚 General Program Questions")
            print("-" * 40)
            question = input("\nAsk about the MCST program: ").strip()
            
            if question:
                print("\n🤔 Searching knowledge base...\n")
                answer = rag.answer_general_question(question)
                print(f"💡 {answer}")
        
        elif choice == "2":
            print("\n💰 Tuition & Fees")
            print("-" * 40)
            question = input("\nAsk about tuition (e.g., 'in-state 6 credits'): ").strip()
            
            if question:
                print("\n🤔 Calculating...\n")
                answer = rag.answer_tuition_question(question)
                print(f"💡 {answer}")
        
        elif choice == "3":
            print("\n🎓 Spring 2026 Course Advising")
            print("-" * 40)
            
            track = input("\nTrack (Thesis/Project/Exam): ").strip().lower()
            if track not in TRACK_REQUIREMENTS:
                print("❌ Invalid track.")
                continue
            
            transcript_path = input("Transcript PDF path: ").strip().replace("\\ ", " ")
            
            if not Path(transcript_path).exists():
                print("❌ File not found.")
                continue
            
            print("\n📄 Processing transcript...")
            taken = extract_taken_courses_from_pdf(transcript_path)
            print(f"✓ Found {len(taken)} completed courses")
            
            spring_df = load_spring_courses("data/Spring2026_Courses.xlsx")
            
            print("\n🔍 Finding recommendations...")
            selected, credits = pick_courses_for_track(track, taken, spring_df)
            
            if not selected:
                print("\n⚠️  No suitable courses found.")
                continue
            
            print("\n" + "=" * 60)
            print("📋 SPRING 2026 COURSE PLAN")
            print("=" * 60)
            
            for i, c in enumerate(selected, 1):
                mode_icon = "💻" if is_online_or_async(c.get("Instruction Mode", "")) else "🏫"
                status_icon = "✅" if "open" in c.get("Status", "").lower() else "⏳"
                
                print(f"\n{i}. {c['code']} - {c['Title']}")
                print(f"   Credits: {c['Credits']} | {mode_icon} {c['Instruction Mode']}")
                print(f"   {c['Days']} {c['Time']}")
                print(f"   {status_icon} {c['Status']}")
            
            print(f"\n📊 Total: {credits}/{MAX_CREDITS} credits")
            
            print("\n💬 Advisor's Recommendation:")
            print("-" * 60)
            advice = generate_course_advice(track, selected, credits)
            print(advice)
        
        elif choice == "4":
            print("\n👋 Good luck! Go Rams! 🐏")
            break
        
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    main()
