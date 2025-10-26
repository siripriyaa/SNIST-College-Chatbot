import streamlit as st
import pandas as pd
import nltk
from nltk.chat.util import Chat, reflections
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import re
from difflib import get_close_matches
import uuid

# ------------------------------
# Setup
# ------------------------------
nltk.download('punkt')
nltk.download('wordnet')
st.set_page_config(page_title="🎓 SNIST College Chatbot", layout="wide")

st.title("🎓 SNIST AI Chatbot")
st.markdown("Ask about courses, timings, events, or follow-up questions!")

# ------------------------------
# Assign unique session ID
# ------------------------------
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
user_id = st.session_state.user_id

# ------------------------------
# Initialize session state
# ------------------------------
if 'chat_history_dict' not in st.session_state:
    st.session_state.chat_history_dict = {}
if user_id not in st.session_state.chat_history_dict:
    st.session_state.chat_history_dict[user_id] = []

if 'all_texts_dict' not in st.session_state:
    st.session_state.all_texts_dict = {}
if 'embeddings_dict' not in st.session_state:
    st.session_state.embeddings_dict = {}

# ------------------------------
# Embedding model
# ------------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------
# Predefined FAQs fallback
# ------------------------------
pairs = [
    [r"(.*)your name(.*)", ["I am SNIST AI Chatbot, your virtual assistant!"]],
    [r"(.*)college timings(.*)", ["Our college timings are from 9:10 AM to 4:00 PM."]],
    [r"(.*)break time(.*)", ["Lunch break is from 12:20 PM to 1:10 PM."]],
    [r"(.*)principal(.*)name(.*)", ["Our principal is Dr. S. P. V. Subba Rao."]],
    [r"(.*)courses(.*)offered(.*)", ["SNIST offers B.Tech, M.Tech, MBA, and Ph.D programs."]],
    [r"(.*)location(.*)", ["Our college is located at Ghatkesar, Hyderabad."]],
    [r"(.*)departments(.*)", ["CSE, AIML, ECE, EEE, Civil, Mechanical, and IT are the major departments."]],
    [r"(.*)", ["Sorry, I didn’t get that. Please try asking about timings, principal, or courses."]]
]
chatbot_fallback = Chat(pairs, reflections)

# ------------------------------
# File upload
# ------------------------------
st.sidebar.header("📂 Upload your files")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV, Excel, TXT, or PDF files",
    type=['csv', 'xlsx', 'txt', 'pdf'],
    accept_multiple_files=True
)

# ------------------------------
# Process uploaded files
# ------------------------------
all_texts = []
for file in uploaded_files:
    if file.type == "text/csv":
        df = pd.read_csv(file)
        if 'Question' in df.columns and 'Answer' in df.columns:
            all_texts.extend(df['Question'].fillna('').tolist())
            all_texts.extend(df['Answer'].fillna('').tolist())
    elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
        df = pd.read_excel(file)
        if 'Question' in df.columns and 'Answer' in df.columns:
            all_texts.extend(df['Question'].fillna('').tolist())
            all_texts.extend(df['Answer'].fillna('').tolist())
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
        chunks = re.findall(r'(.{50,300}?)\s(?=[A-Z])', text.replace("\n"," "))
        all_texts.extend(chunks if chunks else [text])
    elif file.type == "application/pdf":
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunks = re.findall(r'(.{50,300}?)\s(?=[A-Z])', text.replace("\n"," "))
                all_texts.extend(chunks if chunks else [text])

# Store texts & embeddings per user
if all_texts:
    st.session_state.all_texts_dict[user_id] = all_texts
    st.session_state.embeddings_dict[user_id] = embedding_model.encode(all_texts, convert_to_tensor=True)

# ------------------------------
# Functions
# ------------------------------
def get_dataset_answer(question):
    texts = st.session_state.all_texts_dict.get(user_id, [])
    if not texts:
        return None
    close = get_close_matches(question.lower(), texts, n=1, cutoff=0.5)
    if close:
        return close[0]
    return None

def get_semantic_answer(question):
    texts = st.session_state.all_texts_dict.get(user_id, [])
    embeddings = st.session_state.embeddings_dict.get(user_id)
    if not texts or embeddings is None:
        return None
    query_emb = embedding_model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, embeddings, top_k=1)
    best_hit_idx = hits[0][0]['corpus_id']
    score = hits[0][0]['score']
    if score > 0.5:
        return texts[best_hit_idx]
    return None

def build_contextual_query(user_input, history, max_turns=3):
    """Combine previous N turns for context"""
    context = ""
    for speaker, message in history[-(max_turns*2):]:  # 2 entries per turn (You + Bot)
        context += f"{speaker}: {message}\n"
    context += f"You: {user_input}"
    return context

# ------------------------------
# User Input
# ------------------------------
user_input = st.text_input("💬 Type your question here:")

if user_input:
    contextual_query = build_contextual_query(user_input, st.session_state.chat_history_dict[user_id])
    response = get_dataset_answer(contextual_query)
    if not response:
        response = get_semantic_answer(contextual_query)
    if not response:
        response = chatbot_fallback.respond(user_input)

    st.session_state.chat_history_dict[user_id].append(("You", user_input))
    st.session_state.chat_history_dict[user_id].append(("Bot", response))

# ------------------------------
# Display Chat History
# ------------------------------
for speaker, message in st.session_state.chat_history_dict[user_id]:
    if speaker == "You":
        st.markdown(f"**{speaker}:** {message} 🗨️")
    else:
        st.markdown(f"**{speaker}:** {message} 🤖")

# ------------------------------
# Clear Chat
# ------------------------------
if st.button("Clear Chat"):
    st.session_state.chat_history_dict[user_id] = []
