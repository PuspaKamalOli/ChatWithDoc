# Chat with PDFs using Gemini AI

This Streamlit application allows us to upload PDF files and chat with them using Google's Gemini AI. The app processes the PDFs, creates embeddings for text chunks, and enables multi-turn conversations with context awareness.

## Features

1. **Upload Multiple PDFs:** 
   - Upload multiple PDF files at once for processing.

2. **PDF Text Extraction:**
   - Extract text content from PDFs efficiently.

3. **Text Chunking:**
   - Split large text into manageable chunks with overlap for better context.

4. **Vector Store Integration:**
   - Use FAISS to create and load vector stores for text embeddings.

5. **Generative AI Responses:**
   - Powered by Google Gemini AI for accurate and detailed answers.

6. **Natural Language Date Extraction:**
   - Parse natural language inputs like "next Monday" or "tomorrow" to extract dates.

7. **Form Handling for Appointments:**
   - Collect user details (name, email, phone) and schedule appointments.

8. **Multi-Turn Conversations:**
   - Engage in context-aware conversations with memory of previous questions and answers.

## Setup

### Prerequisites

- Python 3.9 or higher
- Streamlit
- Required Python packages (see `requirements.txt`)
- Google Generative AI API key

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root folder and add  Google API key:
   ```
   GOOGLE_API_KEY=your-google-api-key
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload
2.  PDF files using the sidebar.
3. Process the PDFs to create embeddings and store them in a FAISS vector store.
4. Ask questions in the chat interface, and get context-aware answers powered by Gemini AI.
5. Optionally, provide your details to schedule an appointment.

## File Structure

- `app.py`: Main Streamlit application file.
- `requirements.txt`: List of Python dependencies.
- `.env`: Environment variables for API keys (not included in the repository).



## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- Built using [LangChain](https://github.com/hwchase17/langchain).
- Powered by [Google Generative AI](https://developers.generativeai.google).
