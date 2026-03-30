# WSSU CS Graduate Advisor - RAG Implementation

An AI-powered graduate advisor for Winston-Salem State University's Master of Science in Computer Science and Information Technology (MCST) program. Uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware advice.

## Features

🎓 **General Program Questions**
- Answer questions about MCST program structure
- Explain concentration options (Thesis, Project, Exam)
- Provide information on core courses, requirements, and electives
- Share admission requirements and program policies

💰 **Tuition & Fees Calculator**
- Calculate tuition for in-state and out-of-state students
- Provide information for 1-9 credit hours
- Explain optional fees (Book Rental, Health Insurance)

📚 **Personalized Course Advising**
- Analyze student transcripts
- Recommend appropriate courses for Spring 2026
- Check for time conflicts
- Support online/asynchronous and in-person schedules
- Limit recommendations to 9 credit hours (as per advising rules)
- Track-specific recommendations (Thesis/Project/Exam)

## Architecture

### RAG (Retrieval-Augmented Generation)
The system uses RAG to provide accurate answers based on official university documents:

1. **Document Processing**: University catalogs, course schedules, and degree requirements are chunked and embedded
2. **Vector Store**: Embeddings stored in ChromaDB for fast semantic search
3. **Retrieval**: User questions are embedded and matched against the knowledge base
4. **Generation**: LLM generates accurate answers using retrieved context

### Technology Stack

- **LangChain**: RAG framework and orchestration
- **OpenAI**: Embeddings (text-embedding-3-small) and LLM (GPT-4o-mini)
- **ChromaDB**: Vector database for document storage
- **Pandas**: Data processing for course schedules
- **PyPDF**: Transcript parsing

## Project Structure

```
wssu-cs-advisor/
├── data/                           # Source documents
│   ├── general_dept.txt           # Program info, mission, requirements
│   ├── Spring2026_Courses.xlsx    # Course schedule
│   ├── Degree_Requirements_for_MCST.docx
│   └── 25-26-graduate.pdf         # Tuition info
├── processed/                      # Generated embeddings (CSV-based)
│   ├── all_embeddings.csv
│   ├── general_embeddings.csv
│   └── tuition_embeddings.csv
├── chroma_db/                      # Vector store (LangChain-based)
├── langchain_advisor.py            # Main LangChain RAG implementation ⭐
├── integrated_advisor.py           # Alternative: custom RAG with OpenAI
├── prepare_rag_data.py            # CSV embedding generation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Prepare RAG Knowledge Base

**Option A: LangChain with ChromaDB (Recommended)**

```bash
python langchain_advisor.py
```

On first run, the system will offer to build the knowledge base automatically.

**Option B: Custom CSV-based embeddings**

```bash
python prepare_rag_data.py
```

This creates embeddings in `processed/` directory.

## Usage

### Interactive Mode

Run the main advisor:

```bash
python langchain_advisor.py
```

You'll see:
```
============================================================
        WSSU CS Graduate Advisor
        Powered by LangChain RAG
============================================================

I can help you with:
  1. General questions about the MCST program
  2. Tuition and fees information
  3. Course advising for Spring 2026
  4. Exit

What would you like help with? (1/2/3/4):
```

### Example Interactions

#### 1. General Questions

```
What would you like help with? 1

📚 General Program Questions
----------------------------------------

Ask about the MCST program: What are the core courses for thesis students?

🤔 Searching knowledge base...

💡 Thesis students have three core courses: CST 5320 (Design and Analysis of 
Algorithms Methods), CST 5322 (Advanced Software Engineering), and CST 6306 
(Advanced Database Management Systems). These 9 credit hours form the foundation 
of the thesis concentration.
```

#### 2. Tuition Calculation

```
What would you like help with? 2

💰 Tuition & Fees
----------------------------------------

Ask about tuition (e.g., 'in-state 6 credits'): How much for an out-of-state 
student taking 9 credit hours?

🤔 Calculating...

💡 For an out-of-state graduate student taking 9 credit hours, the total tuition 
and fees are $8,552.00 per semester. Optional fees include Book Rental ($33.33 
per credit hour) and Health Insurance ($1,396.54 for 6+ credit hours, can be waived).
```

#### 3. Course Advising

```
What would you like help with? 3

🎓 Spring 2026 Course Advising
----------------------------------------

Track (Thesis/Project/Exam): thesis

Transcript PDF path: data/ThesisStudentTranscript1.pdf

📄 Processing transcript...
✓ Found 3 completed courses

🔍 Finding recommendations...

============================================================
📋 SPRING 2026 COURSE PLAN
============================================================

1. CST 5320 - Design and Analysis of Algorithms Methods
   Credits: 3 | 🏫 In-Person
   TR 6:00PM-8:45PM
   ✅ Open

2. CST 6306 - Advanced Database Management Systems
   Credits: 3 | 💻 Online Asynchronous
   TBA TBA
   ✅ Open

3. CST 6302 - Programming Languages and Compilers
   Credits: 3 | 🏫 In-Person
   MW 6:00PM-8:45PM
   ✅ Open

📊 Total: 9/9 credits

💬 Advisor's Recommendation:
------------------------------------------------------------
This schedule perfectly fits your thesis track requirements with all core and 
required courses. You'll have a nice mix of in-person and online learning, 
with in-person classes on different days to avoid conflicts. Register early 
as these core courses are essential for your degree progression!
```

## Implementation Details

### How RAG Works in This System

1. **Document Chunking**:
   - General info: 800 chars with 150 char overlap
   - Degree requirements: 600 chars with 100 char overlap
   - Tuition: Single chunk for accuracy
   - Courses: One chunk per course

2. **Embedding**:
   - Model: `text-embedding-3-small` (1536 dimensions)
   - Fast and cost-effective
   - Semantic similarity search

3. **Retrieval**:
   - Source filtering (general, tuition, courses)
   - Top-k retrieval (k=1 to 3 depending on query type)
   - Similarity thresholding

4. **Generation**:
   - Model: `gpt-4o-mini`
   - Temperature: 0.2-0.4 (factual but friendly)
   - Custom prompts for each query type

### Course Selection Algorithm

The advisor uses rule-based logic for course selection:

1. **Parse transcript**: Extract completed CST courses
2. **Identify needs**: Compare against track requirements
3. **Filter availability**: Check Spring 2026 offerings
4. **Avoid conflicts**: Check time overlaps (online courses never conflict)
5. **Prioritize**: Required courses → Electives
6. **Limit credits**: Maximum 9 credit hours
7. **Generate advice**: LLM explains the recommendations

### Key Features

✅ **Accurate**: Uses official university documents  
✅ **Context-aware**: Knows student's track and completed courses  
✅ **Conflict-free**: Checks scheduling conflicts  
✅ **Flexible**: Handles online, async, and in-person courses  
✅ **Conversational**: Natural language interface  
✅ **Up-to-date**: Easy to update with new course schedules  

## Customization

### Adding New Documents

1. Place document in `data/` directory
2. Update `prepare_documents()` in `langchain_advisor.py`:

```python
# Add new document processing
print("\n📄 Processing new document...")
with open("data/new_doc.txt", "r") as f:
    text = f.read()

chunks = text_splitter.split_text(text)
for i, chunk in enumerate(chunks):
    documents.append(Document(
        page_content=chunk,
        metadata={"source": "new_document", "chunk": i}
    ))
```

3. Rebuild knowledge base

### Updating Course Schedule

Simply replace `data/Spring2026_Courses.xlsx` with new semester data and rebuild.

### Adjusting Credit Limits

Change `MAX_CREDITS` constant in the code:

```python
MAX_CREDITS = 12  # For part-time advisors allowing more credits
```

## Comparison: Two Implementations

| Feature | langchain_advisor.py | integrated_advisor.py |
|---------|---------------------|----------------------|
| Framework | LangChain | Custom |
| Vector Store | ChromaDB | CSV + NumPy |
| Setup Complexity | Medium | Low |
| Performance | Faster retrieval | Slower retrieval |
| Extensibility | Easier to extend | More manual |
| Dependencies | More libraries | Fewer libraries |

**Recommendation**: Use `langchain_advisor.py` for production. Use `integrated_advisor.py` for learning or if you want minimal dependencies.

## Troubleshooting

### "Knowledge base not found"
Run the advisor once and choose to set up the knowledge base, or run `prepare_rag_data.py` manually.

### "Transcript file not found"
Ensure the PDF path is correct. Try drag-and-drop if using a terminal.

### "No suitable courses found"
Student may have completed all requirements, or Spring course offerings may be limited. Check the Excel file.

### OpenAI API Errors
- Check `.env` file has correct API key
- Verify API key has credits
- Check network connectivity

## Future Enhancements

- 🔄 Add support for Fall semester advising
- 📊 Track student progress across semesters
- 🔔 Notifications for course registration deadlines
- 💬 Slack/Teams integration
- 🌐 Web interface
- 📧 Email advisor summary
- 🤖 Multi-turn conversations with memory
- 📈 Analytics on common questions

## Contributing

To add features:
1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Test with sample transcripts
5. Submit a pull request

## License

Educational use for WSSU Computer Science Department.

## Contact

For questions about the MCST program:
- **Jacqueline Bethea**
- Email: betheaj@wssu.edu
- Phone: (336) 750-2478
- Office: Elva Jones Computer Science Building, Room 4211

---

**Built with ❤️ for WSSU CS students | Go Rams! 🐏**
