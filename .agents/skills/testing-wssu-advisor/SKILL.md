# Testing WSSU-CS-Advisor

## Overview
Streamlit-based RAG advisor app for WSSU CS graduate students. Uses OpenAI API for LLM queries and ChromaDB for vector storage.

## Running Locally
```bash
cd /home/ubuntu/repos/WSSU-CS-Advisor
pip install -r requirements.txt
streamlit run streamlit_app.py --server.headless true --server.port 8501
```
App will be available at http://localhost:8501

## Devin Secrets Needed
- `OPENAI_API_KEY` — Required for LLM-powered features (General Questions, Tuition Q&A, Course Advising AI recommendations). Without it, the app shows a friendly error message.

## What to Test

### Without API Key (Error Handling)
1. Ensure no `OPENAI_API_KEY` is set in environment or `.streamlit/secrets.toml`
2. Run the app and verify:
   - App loads without crashing (no `StreamlitSecretNotFoundError`)
   - WSSU header renders at the top
   - Red error box shows "OpenAI API key not configured"
   - Setup instructions for Streamlit Cloud and local dev are displayed
   - No tabs are rendered below the error (`st.stop()` works)

### With API Key (Full Functionality)
1. Set `OPENAI_API_KEY` via environment variable or `.streamlit/secrets.toml`
2. Verify three tabs load: General Questions, Tuition & Fees, Course Advising
3. Test General Questions tab: type a question and click "Get Answer"
4. Test Tuition tab: select residency/credits and click "Calculate Tuition"
5. Test Course Advising tab: requires a PDF transcript upload

### UI Checks
- Header should have white text on red-to-dark gradient (WSSU branding)
- Buttons should be WSSU red (#C41E3A)
- Sidebar should show "About" info and contact details

## Common Issues
- Port 8501 may be in use from a previous session. Kill with: `pkill -f 'streamlit run'`
- `chroma_db/` must be present for the knowledge base to work. It's committed to the repo.
- The app catches both `KeyError` and `FileNotFoundError` for missing secrets — these are the two failure modes in Streamlit's secrets API.
