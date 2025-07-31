import os

from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()

# Key will be loaded from environment, not hardcoded
api_key = os.getenv("OPENAI_API_KEY")

# 1. Load a Wikipedia article
loader = WikipediaLoader(query="Python (programming language)", load_max_docs=1)
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 3. Embed and store locally
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. Setup retriever + LLM
retriever = db.as_retriever()
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 5. Ask a question
question = "What are key features of Python as a programming language?"
response = qa.run(question)
print(response)
