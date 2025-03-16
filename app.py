#import all the packages 
import os 
import streamlit as st
import certifi  # Import certifi
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

#---------------------------------------------------
#for steamlit deployment only 
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] #Comment both key when runing on vs code 
groq_api_key = st.secrets["GROQ_API_KEY"] #Comment both key when runing on vs code 
#---------------------------------------------------

#RAG libraries 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
#intialize env variables 
load_dotenv()

# Set SSL certificate file path using certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

#get all the keys needed
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")



#instantiate the llm model
llm=ChatGroq(model_name="Llama3-8b-8192",groq_api_key=groq_api_key)


#define the chat prompt for llm to understand
prompt=ChatPromptTemplate.from_template(
    '''
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}  
    </context>
    Question:{input}
'''
)
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        ##data ingestion
        #initialize loader
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")
        st.session_state.docs=st.session_state.loader.load() 

        ##data tranformation to chunks
        #initialize text splitter
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

        ##Vector Embedding & push vectors to vectorstore db
        #initialize embeddings
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


####--------------## Streamlit application ##--------------###

st.title("RAG Document Q&A With Groq And LLama3")

##displayed on app: this prompt is written by user
user_prompt=st.text_input("Enter your query from the Research Paper")

##displayed on app
if st.button("Document Embeddings"):
    create_vector_embeddings()
    st.write("Vector Database is ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    #instantiate the retriever
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({"input":user_prompt})
    print(f"Response time :{time.process_time()-start} ")

    ##displayed on app
    st.write(response['answer'])


    #With a Streamlit expander
    with st.expander("Document Similarity Search"):
        for i ,doc in enumerate(response['context']):
            ##diplayed on app
            st.write(doc.page_content)
            st.write("----------------------------------")

