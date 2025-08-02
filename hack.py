import streamlit as st
import fitz  # PyMuPDF
import requests
import json
import os
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "granite3.3:2b" # Or any other model you have, e.g., "gemma:2b"
USER_DB_FILE = "users.json"
HISTORY_DIR = "history"

# --- PAGE CONFIG ---
st.set_page_config(page_title="📘COGNIFY", layout="centered")

# --- SESSION STATE INITIALIZATION ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = ""

# --- CORE FUNCTIONS ---
def extract_text(file):
    """Extracts text from an uploaded PDF file."""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return ""

def ask_ollama_stream(question, context):
    """Sends a question to Ollama and yields the response tokens as a stream."""
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided. If the answer is not in the context, say 'The answer is not available in the provided document.'

Context:
{context}

Question:
{question}

Answer:"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True  # Enable streaming
    }

    try:
        # Use stream=True to handle the streaming response
        # --- FIX: Increased the timeout from 60 to 300 seconds ---
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        # Each line is a JSON object; parse it and extract the response part
                        chunk = json.loads(line.decode('utf-8'))
                        yield chunk.get("response", "")
            else:
                yield f"❌ Ollama Error {response.status_code}: {response.text}"
    except requests.exceptions.ReadTimeout:
        yield f"❌ Ollama connection timed out. The model may be taking too long to load. Please try again."
    except requests.exceptions.ConnectionError as e:
        yield f"❌ Failed to connect to Ollama. Is it running? Error: {e}"
    except Exception as e:
        yield f"❌ An unexpected error occurred: {e}"


def download_pdf_report(email, messages):
    """Generates a PDF report of the Q&A session."""
    path = f"{HISTORY_DIR}/{email}/QnA_Report.pdf"
    c = canvas.Canvas(path)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, 10.5 * inch, "Cognify Q&A Report")
    c.setFont("Helvetica", 11)

    text = c.beginText(inch, 10 * inch)
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        # Simple line wrapping
        full_line = f"{role}: {msg['content']}"
        content_lines = full_line.splitlines()
        for line in content_lines:
            # Wrap long lines
            while len(line) > 90:
                # Find the last space before the limit
                split_pos = line.rfind(' ', 0, 90)
                if split_pos == -1: # No space found, hard break
                    split_pos = 90
                text.textLine(line[:split_pos])
                line = line[split_pos:].lstrip()
            text.textLine(line)
        text.textLine("") # Add a blank line for spacing

        if text.getY() < 1.5 * inch:
            c.drawText(text)
            c.showPage()
            c.setFont("Helvetica", 11)
            text = c.beginText(inch, 10.5 * inch)

    c.drawText(text)
    c.save()
    return path

# --- USER & HISTORY MANAGEMENT ---
def load_users():
    if not os.path.exists(USER_DB_FILE): return {}
    with open(USER_DB_FILE, "r") as f: return json.load(f)

def save_users(users):
    with open(USER_DB_FILE, "w") as f: json.dump(users, f, indent=2)

def login(email, password):
    users = load_users()
    return email in users and users[email] == password

def signup(email, password):
    if not email or not password:
        st.error("Email and password cannot be empty.")
        return False
    users = load_users()
    if email in users: return False
    users[email] = password
    save_users(users)
    os.makedirs(f"{HISTORY_DIR}/{email}", exist_ok=True)
    return True

def load_history(email):
    path = f"{HISTORY_DIR}/{email}/chat.json"
    if os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except json.JSONDecodeError:
            return [] # Return empty list if history is corrupted
    return []

def save_history(email, messages):
    os.makedirs(f"{HISTORY_DIR}/{email}", exist_ok=True)
    with open(f"{HISTORY_DIR}/{email}/chat.json", "w") as f:
        json.dump(messages, f, indent=2)

# --- AUTHENTICATION UI ---
if not st.session_state.authenticated:
    st.title("📘 COGNIFY")
    st.subheader("Login or Sign Up to Get Started")
    tab1, tab2 = st.tabs(["🔐 Login", "🆕 Sign Up"])
    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if login(email, password):
                    st.session_state.authenticated = True
                    st.session_state.current_user = email
                    st.session_state.messages = load_history(email)
                    st.session_state.context = ""
                    st.rerun()
                else:
                    st.error("❌ Invalid email or password.")
    with tab2:
        with st.form("signup_form"):
            email = st.text_input("New Email")
            password = st.text_input("Create Password", type="password")
            if st.form_submit_button("Sign Up"):
                if signup(email, password):
                    st.success("✅ Account created. Please proceed to the Login tab.")
                else:
                    st.warning("⚠️ Email already exists or is invalid.")

# --- MAIN APPLICATION UI ---
if st.session_state.authenticated:
    # --- SIDEBAR ---
    st.sidebar.title("⚙️ Options")
    st.sidebar.write(f"👤 Logged in as: **{st.session_state.current_user}**")
    if st.sidebar.button("🔓 Logout"):
        save_history(st.session_state.current_user, st.session_state.messages)
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        save_history(st.session_state.current_user, st.session_state.messages)
        st.rerun()

    # --- MAIN CHAT INTERFACE ---
    st.title("📘 COGNIFY")
    st.caption("Upload a PDF and ask questions. Your entire chat history is saved to your account.")
    uploaded_file = st.file_uploader("📄 Upload your study PDF", type="pdf")
    if uploaded_file:
        with st.spinner("🔍 Reading document..."):
            st.session_state.context = extract_text(uploaded_file)
        st.success("✅ Document loaded. The chat will now use this context.")

    # Display past messages from the continuous history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle new question input
    if question := st.chat_input("Ask something from the PDF..."):
        if not st.session_state.context:
            st.warning("⚠️ Please upload a PDF document before asking questions.")
        else:
            # Add user question to UI and history
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Use st.write_stream to display the streaming response
            with st.chat_message("assistant"):
                response_generator = ask_ollama_stream(question, st.session_state.context)
                full_response = st.write_stream(response_generator)

            # Add the complete assistant response to history and save
            if "❌" not in full_response: # Don't save error messages to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_history(st.session_state.current_user, st.session_state.messages)

    # Download button in the sidebar
    if st.session_state.messages:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Download Report")
        if st.sidebar.button("Generate Q&A PDF Report"):
            with st.spinner("Generating PDF..."):
                pdf_file = download_pdf_report(st.session_state.current_user, st.session_state.messages)
                with open(pdf_file, "rb") as f:
                    st.sidebar.download_button(
                        label="📄 Click to Download",
                        data=f,
                        file_name="Cognify_QA_Report.pdf",
                        mime="application/pdf"
                    )