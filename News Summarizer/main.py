import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()

# Setup Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Prompt template
summarize_prompt = PromptTemplate(
    template="Summarize the following news article:\n\n{article}\n\nSummary:",
    input_variables=["article"]
)

# LLM Chain
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# News extraction function
def extract_news(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ''.join([p.get_text() for p in paragraphs])
        return text
    except Exception as ex:
        return f"❌ Failed to fetch news from {url}: {ex}"

# Page Config
st.set_page_config(page_title="📰 Gemini News Summarizer", layout="centered")

# Custom CSS for popup-like effect
st.markdown("""
    <style>
        .summary-box {
            background-color: #f1f3f6;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.1);
            margin-top: 20px;
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# UI Elements
st.title("🧠 Gemini News Summarizer")
st.markdown("Just paste a news article URL and let AI do the reading for you!")

url = st.text_input("🔗 Paste News Article URL")

if st.button("✨ Summarize Now"):
    if not url.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("🔍 Extracting article and summarizing with Gemini..."):
            article = extract_news(url)

            if article.startswith("❌"):
                st.error(article)
            else:
                summary = summarize_chain.run(article=article)

                with st.expander("📄 Click to view full article content", expanded=False):
                    st.write(article[:3000])  # Limit characters to keep UI clean

                st.markdown(f"""
                <div class="summary-box">
                    <h4>✅ Summary:</h4>
                    <p>{summary}</p>
                </div>
                """, unsafe_allow_html=True)

