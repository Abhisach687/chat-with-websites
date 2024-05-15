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

#The script first loads environment variables using dotenv, and configures the Google Generative AI with the API key.
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#The get_vectorstore_from_url function takes a URL as input, loads the webpage content, splits it into chunks, 
#and creates a FAISS vector store from these chunks. This vector store is then saved locally.
def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    vector_store.save_local("faiss_index")

    return vector_store

#The get_conversational_chain function sets up a conversational model using Google's Generative AI. 
# It uses a prompt template for the model to generate responses.
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

#The user_input function takes a user's question as input, loads the previously saved FAISS vector store, 
# and performs a similarity search on the user's question. 
# It then uses the conversational chain to generate a response based on the most similar documents found in the vector store.
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

#The main function sets up the Streamlit interface. It provides a text input for the user to enter a website URL and their question. 
# If a vector store for the given URL doesn't exist in the session state, it calls get_vectorstore_from_url to create one. 
# It then calls user_input to generate a response to the user's question.
def main():
    st.set_page_config(page_title="Chat with websites", page_icon="🤖")
    st.title("Chat with websites")

    with st.sidebar:
        st.header("Settings")
        website_url = st.text_input("Website URL")

    if website_url is None or website_url == "":
        st.info("Please enter a website URL")

    else:
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_url(website_url)    

        # user input
        user_query = st.text_input("Type your message here...")
        if user_query is not None and user_query != "":
            user_input(user_query)

if __name__ == "__main__":
    main()