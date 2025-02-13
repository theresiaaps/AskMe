from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import logging
from langchain_community.document_loaders import SeleniumURLLoader

app = FastAPI()

template = """
You are an assistant for questioning-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.
Question: {question}
Answer: 
"""

embeddings = OllamaEmbeddings(model="mistral")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="mistral")


class QueryRequest(BaseModel):
    url: str
    question: str


def load_page(url):
    try:
        logging.info(f"Loading page: {url}")
        loader = SeleniumURLLoader(urls=[url])
        documents = loader.load()
        logging.info(f"Successfully loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        logging.error(f"Error loading page: {e}")
        return []


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def index_docs(documents):
    vector_store.add_documents(documents)


def retrieve_docs(query):
    return vector_store.similarity_search(query)


def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})


@app.post("/ask")
def ask_question(request: QueryRequest):
    documents = load_page(request.url)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)
    retrieved_docs = retrieve_docs(request.question)

    context = ".\n.\n".join([doc.page_content for doc in retrieved_docs])
    answer = answer_question(request.question, context)

    print(f"Context: {context}")
    print(f"Answer: {answer}")

    return {"answer": answer}
