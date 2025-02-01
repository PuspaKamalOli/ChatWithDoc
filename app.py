import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime, timedelta
import re

# Load environment variables
load_dotenv()

# Configure Google API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()
genai.configure(api_key=api_key)

# Validate email
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email)

# Validate phone number
def is_valid_phone(phone):
    phone_regex = r'^\+?\d{7,15}$'
    return re.match(phone_regex, phone)

# Extract date from natural language input
def extract_date(user_input):
    today = datetime.today()
    if "next monday" in user_input.lower():
        days_ahead = (7 - today.weekday() + 0) % 7 or 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    elif "tomorrow" in user_input.lower():
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    # Add more natural language date handling as needed
    return None

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text()
        if not text.strip():
            raise ValueError("Uploaded PDF files are empty or contain unsupported content.")
        return text
    except Exception as e:
        st.error(f"Error reading PDF files: {str(e)}")
        raise

# Function to split text into chunks
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text into chunks: {str(e)}")
        raise

# Function to create and save vector store
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise

# Function to load a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the
    provided context, respond with "The answer is not available in the context" and nothing else.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error setting up conversational chain: {str(e)}")
        raise

# Function to handle user input and generate responses
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)

        if not docs:
            st.write("No relevant context found in the uploaded PDFs.")
            return

        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        raise

# Main Streamlit app
def main():
    st.set_page_config("Chat with PDF", layout="wide")
    st.title(" Chat with PDFs")  # Corrected Unicode characters

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.header("📂 Upload and Process PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDF files (multiple allowed)", accept_multiple_files=True, type=["pdf"]
        )

        if st.button("Process PDFs"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file before processing.")
            else:
                with st.spinner("Processing PDFs..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs processed successfully!")
                    except Exception as e:
                        st.error(f"Error during PDF processing: {str(e)}")

    # Chat interface
    st.header("
    Ask Questions from the PDFs")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "user_details" not in st.session_state:
        st.session_state.user_details = {"name": "", "email": "", "phone": ""}

    if "appointment_details" not in st.session_state:
        st.session_state.appointment_details = {"date": None}

    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False

    user_question = st.text_input("Type your question here")
    if st.button("Submit Question"):
        if user_question.strip():
            with st.spinner("Generating response..."):
                try:
                    response = user_input(user_question.strip())
                    st.session_state.chat_history.append(("You", user_question))
                    st.session_state.chat_history.append(("Bot", response))
                except Exception as e:
                    st.error(f"Error during chat: {str(e)}")
        else:
            st.warning("Please type a question before submitting.")

    # Conversational form for callback or appointments
    if "call me" in user_question.lower() or "appointment" in user_question.lower():
        st.subheader(" Provide Your Details")

        name = st.text_input("Name", value=st.session_state.user_details["name"])
        email = st.text_input("Email", value=st.session_state.user_details["email"])
        phone = st.text_input("Phone", value=st.session_state.user_details["phone"])

        # Update session state
        st.session_state.user_details.update({"name": name, "email": email, "phone": phone})

        # Extract date if applicable
        if not st.session_state.appointment_details["date"]:
            extracted_date = extract_date(user_question)
            st.session_state.appointment_details["date"] = extracted_date

        # Disable submit button if fields are incomplete
        if not name or not is_valid_email(email) or not is_valid_phone(phone):
            st.warning("Please fill all fields with valid data.")
            submit_disabled = True
        else:
            submit_disabled = False

        if st.button("Submit Details", disabled=submit_disabled):
            st.success(
                f"Details submitted successfully! Appointment scheduled on {st.session_state.appointment_details['date'] or 'a suitable date'}."
            )
            st.session_state.form_submitted = True

    if st.session_state.form_submitted:
        st.write(" Thank you! We'll be in touch soon.")

    # Display chat history
    st.subheader("Chat History")
    for role, text in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"** {role}:** {text}")
        else:
            st.markdown(f"** {role}:** {text}")

    # Clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

if __name__ == "__main__":
    main()

