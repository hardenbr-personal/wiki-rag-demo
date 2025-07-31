Wikipedia RAG Assistant

An AI-powered assistant that answers your natural language questions by dynamically retrieving and summarizing relevant Wikipedia articles using LangChain, ChromaDB, and OpenAI's GPT models.

---

FEATURES

- Automatically selects the best Wikipedia article based on your question
- Loads, chunks, and embeds article content using LangChain tools
- Answers questions with a lightweight Retrieval-Augmented Generation (RAG) pipeline
- Efficient design to reduce token usage and costs by leveraging GPT-3.5

---

INSTALLATION

1. Clone the repository:
   git clone https://github.com/hardenbr-personal/wikipedia-rag-assistant.git
   cd wikipedia-rag-assistant

2. Create and activate a virtual environment:
   Windows:
     python -m venv rag-env
     .\rag-env\Scripts\activate
   macOS/Linux:
     python3 -m venv rag-env
     source rag-env/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Add your OpenAI API key to a .env file:
   OPENAI_API_KEY=sk-...

---

USAGE

Run the assistant with:
   python main.py

Then enter your question in the terminal:
   Ask a question (or type 'exit' to quit): What is the capital of the Roman Empire?

---

HOW IT WORKS

1. Accepts a natural language question from the user
2. Uses GPT-3.5 to suggest the most relevant Wikipedia article
3. Loads that article, splits it into chunks, and embeds them into Chroma
4. Retrieves the most relevant chunks and uses GPT-3.5 to answer the question

---

EXAMPLE

Input:
   What is the capital of the Roman Empire?

Output:
   The capital of the Roman Empire was Rome. Initially a monarchy, then a republic, Rome later became the seat of imperial power for centuries.

---

DEPENDENCIES

- LangChain
- ChromaDB
- OpenAI GPT-3.5
- python-dotenv

---

LICENSE

MIT License (or your preferred license)

---

ACKNOWLEDGEMENTS

Built as a learning project with help from OpenAI and LangChain documentation.
