import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PyPDF2 import PdfReader

#Load environment variables and configure Google's Generative AI with an API key.
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#This function takes a URL as input, loads the text from the website, splits the text into chunks, creates embeddings for these chunks, 
# and stores these embeddings in a vector store. It returns the vector store.
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    documents = [Document(text) for text in loader.load()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    document_chunks = text_splitter.split_documents([doc.page_content for doc in documents])
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return document_chunks, embeddings

#Define a Document class to represent a document with text content and metadata.
class Document:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}

#This function is similar to get_vectorstore_from_url, but it takes a string of text as input instead of a URL. 
# It's used for processing text from PDF documents.
def get_vectorstore_from_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    document = Document(text)
    document_chunks = text_splitter.split_documents([document])
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    vector_store.save_local("faiss_index")
    return document_chunks, embeddings

#This function sets up a conversational model and a question-answering chain. 
# The model is configured with a prompt template that instructs it to answer questions based on the provided context.
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # Add a general knowledge AI model
    general_knowledge_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    general_knowledge_prompt = PromptTemplate(template = "{context}\n{question}\n\nAnswer:", input_variables = ["context", "question"])
    general_knowledge_chain = load_qa_chain(general_knowledge_model, chain_type="stuff", prompt=general_knowledge_prompt)

    return chain, general_knowledge_chain

#This function takes a user question, finds the documents most similar to the question, and generates a response. 
# The response is then displayed in the Streamlit app.
def user_input(user_question):
    document_chunks, embeddings = st.session_state.vector_store
    new_db = FAISS.from_documents(document_chunks, embeddings)
    docs = new_db.similarity_search(user_question)
    chain, general_knowledge_chain = get_conversational_chain()
    response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)

    greetings = ["hi", "hello", "hey", "greetings", "good day", "Namaste", "Hola", "Bonjour", "Ciao", "Salut", "Hallo", "Konnichiwa", "Ni Hao", "Salaam", "Shalom", "Sawubona", "Zdravstvuyte", "Merhaba", "Privet", "Aloha", "Guten Tag", "Olá", "Hej", "Hei", "Hej", "Hoi"]

    if user_question.lower() in greetings:
        st.write("Reply: Hello! How can I assist you today?")
        return

    not_in_context_phrases = ["this question cannot be answered from the given context", "answer is not available in the context", "unable to provide an answer based on the context", "the context does not contain information to answer this question"]

    if any(phrase in response["output_text"].lower() for phrase in not_in_context_phrases):
        response = general_knowledge_chain({"input_documents": [], "question": user_question}, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])
 #This function sets up the Streamlit app, takes user input for a website URL or a PDF file, 
 # processes the text and creates a vector store, takes user queries, generates responses, and displays the responses in the app.
def main():
    st.set_page_config(page_title="Chatbot", page_icon="🤖")
    st.title("Chatbot")

    with st.sidebar:
        st.header("Settings")
        website_url = st.text_input("Website URL")
        uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)

    if (website_url is None or website_url == "") and not uploaded_files:
        st.info("Please enter a website URL or upload a PDF")
    else:
        if "vector_store" not in st.session_state:
            if website_url is not None and website_url != "":
                st.session_state.vector_store = get_vectorstore_from_url(website_url)
            elif uploaded_files:
                text = ""
                for uploaded_file in uploaded_files:
                    pdf_file = PdfReader(uploaded_file)
                    for page in pdf_file.pages:
                        text += page.extract_text()
                st.session_state.vector_store = get_vectorstore_from_text(text)

        user_query = st.text_input("Type your message here...")
        if user_query is not None and user_query != "":
            user_input(user_query)

if __name__ == "__main__":
    main()