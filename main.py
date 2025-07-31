import os
import shutil
import hashlib

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Setup models
gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings()

# Set up logs
log_path = "session_log.txt"
def log_interaction(question, article, answer):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now()}]\n")
        f.write(f"Q: {question}\n")
        f.write(f"Article: {article}\n")
        f.write(f"A: {answer}\n")

# Main loop
while True:
    question = input("\nAsk a question (or type 'exit' to quit): ").strip()
    if question.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    # Use GPT to pick a good Wikipedia article title
    search_term_prompt = (
        f"Suggest the exact Wikipedia article title that would best answer this question:\n'{question}'\n"
        f"Return only the article title."
    )
    search_term = gpt35.invoke(search_term_prompt).content.strip().strip('"')
    print(f"Using Wikipedia article: {search_term}")

    # Generate a unique DB folder name based on the article title
    article_hash = hashlib.md5(search_term.encode()).hexdigest()
    db_path = f"./chroma_db/{article_hash}"

    # Clean out old DB directory if it exists
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    # Load the Wikipedia article
    loader = WikipediaLoader(query=search_term, load_max_docs=1)
    docs = loader.load()

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=db_path
    )

    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = qa.run(question)

    print("\nAnswer:")
    print(response)

    # Log it
    log_interaction(question, search_term, response)
