# 🌐 Web Deployment Guide

## Three Options to Deploy Your WSSU CS Advisor

---

## ⭐ Option 1: Streamlit Cloud (RECOMMENDED - FREE & EASY)

**Perfect for: Academic projects, demos, sharing with professors**

### Why Streamlit?
- ✅ **100% Free** hosting
- ✅ **5 minutes** to deploy
- ✅ **No coding required** for deployment
- ✅ **Shareable link** - just send URL to professors
- ✅ **Auto-updates** when you push to GitHub

### Step-by-Step Deployment

#### 1. Test Locally First

```bash
# Install streamlit
pip install streamlit

# Run locally to test
streamlit run streamlit_app.py
```

Your browser will open to `http://localhost:8501`

#### 2. Push to GitHub

```bash
# Create a new repo on GitHub
# Then in your project folder:

git init
git add .
git commit -m "Initial commit - WSSU CS Advisor"
git remote add origin https://github.com/YOUR_USERNAME/wssu-cs-advisor.git
git push -u origin main
```

**Important Files to Include:**
```
your-repo/
├── streamlit_app.py          ← Main app
├── requirements.txt          ← Dependencies
├── data/
│   ├── general_dept.txt
│   └── Spring2026_Courses.xlsx
├── chroma_db/                 ← Vector database (important!)
└── README.md
```

#### 3. Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository
5. Set:
   - **Main file**: `streamlit_app.py`
   - **Requirements file**: `requirements.txt`
6. Click **"Advanced settings"**
7. Add secrets:
   ```toml
   OPENAI_API_KEY = "sk-proj-your-key-here"
   ```
8. Click **"Deploy"**!

#### 4. Share the Link

You'll get a URL like:
```
https://wssu-cs-advisor.streamlit.app
```

Send this to your boss and professors! 🎉

---

## 🚀 Option 2: Hugging Face Spaces (FREE - Alternative)

**Perfect for: AI/ML projects, more control**

### Deploy to Hugging Face

1. Go to **https://huggingface.co/spaces**
2. Click **"Create new Space"**
3. Choose **"Gradio"** or **"Streamlit"**
4. Upload your files
5. Add API key in Settings → Secrets

**URL**: `https://huggingface.co/spaces/YOUR_USERNAME/wssu-advisor`

---

## 💻 Option 3: Render.com (FREE Tier Available)

**Perfect for: More professional deployment**

### Deploy to Render

1. Go to **https://render.com**
2. Sign up (free)
3. Click **"New +"** → **"Web Service"**
4. Connect your GitHub repo
5. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run streamlit_app.py --server.port=$PORT`
6. Add environment variable:
   - Key: `OPENAI_API_KEY`
   - Value: Your API key

**URL**: `https://wssu-cs-advisor.onrender.com`

---

## 📱 Quick Comparison

| Feature | Streamlit Cloud | Hugging Face | Render.com |
|---------|----------------|--------------|------------|
| **Ease of Setup** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Free Tier** | Yes (forever) | Yes (forever) | Yes (limited) |
| **Custom Domain** | No | No | Yes (paid) |
| **Auto-deploy** | Yes | Yes | Yes |
| **Best For** | Demos/Academic | ML Projects | Production |

---

## 🎨 What Your Web App Looks Like

The Streamlit app I created has:

### **Three Tabs:**

#### 📚 **Tab 1: General Questions**
- Text input for questions
- Example questions dropdown
- AI-powered answers using RAG

#### 💰 **Tab 2: Tuition Calculator**
- Dropdown for residency (In-State/Out-of-State)
- Dropdown for credit hours (1-9)
- Instant cost calculation

#### 🎓 **Tab 3: Course Advising**
- Track selection (Thesis/Project/Exam)
- PDF upload for transcript
- Smart course recommendations
- AI advisor tips

---

## 🔧 Customization

### Change Colors (WSSU Red & Black)

Edit in `streamlit_app.py`:

```python
st.markdown("""
<style>
    .main-header {
        color: #C41E3A;  /* WSSU Red */
        background: linear-gradient(135deg, #1a1a1a 0%, #4a4a4a 100%);
    }
    .stButton>button {
        background-color: #C41E3A;
    }
</style>
""", unsafe_allow_html=True)
```

### Add Your Logo

```python
st.image("wssu_logo.png", width=200)
```

---

## 🐛 Troubleshooting

### "Module not found"
→ Make sure `requirements.txt` is in your repo

### "Knowledge base not found"
→ Make sure you uploaded the `chroma_db/` folder

### "API key error"
→ Check Streamlit Cloud secrets have your OpenAI key

### "Large files warning"
→ ChromaDB folder might be big. Add `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 500
```

---

## 📊 Usage Analytics (Streamlit Cloud)

Streamlit Cloud shows:
- Number of visitors
- Active users
- App performance

Perfect for showing your professor how many people use it!

---

## 🎯 Best Practices

1. **Test locally first** - Run `streamlit run streamlit_app.py`
2. **Use .gitignore** - Don't commit `.env` files
3. **Add README** - Explain what the app does
4. **Monitor costs** - OpenAI charges per API call
5. **Set usage limits** - In OpenAI dashboard

---

## 💡 Demo Script for Your Boss

*"I deployed our AI advisor as a web application using Streamlit. Anyone can access it by visiting this link [your-url]. The app uses LangChain for RAG, has three main features: general Q&A, tuition calculator, and personalized course recommendations. It's hosted for free on Streamlit Cloud and updates automatically when I push changes to GitHub."*

---

## 🚦 Next Steps

**Immediate (5 minutes):**
1. Run `streamlit run streamlit_app.py` locally
2. Test all three tabs
3. Fix any issues

**Today (30 minutes):**
1. Create GitHub repo
2. Push code
3. Deploy to Streamlit Cloud
4. Share link with one person to test

**This Week:**
1. Share with boss and professors
2. Gather feedback
3. Add analytics tracking
4. Consider custom domain

---

## 📞 Need Help?

- **Streamlit Docs**: https://docs.streamlit.io
- **Streamlit Community**: https://discuss.streamlit.io
- **Example Apps**: https://streamlit.io/gallery

---

**Your app is ready to deploy! Pick Option 1 (Streamlit Cloud) for the quickest results.** 🚀
