import streamlit as st
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import faiss
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = "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"
pinecone.init(api_key="5dbdf6f8-f0e4-4e8a-8cf6-3dcc6ace494a", environment="gcp-starter")

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
# db = FAISS.load_local("db", embeddings)
index_name = "ind"
# docsearch = db.similarity_search(query)
docsearch = Pinecone.from_existing_index(index_name, embeddings)
llm = OpenAI()
qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever())
chat_history = []
st.title('ICICI Helpbot')
user_input = st.text_input("Please enter your question:")
submit_button = st.button('Submit')
if submit_button:
    if user_input.lower() == 'exit':
        st.write("Thank you for using ICICI Helpbot!")
    else:
        result = qa({"question": user_input, "chat_history": chat_history})
        chat_history.append((user_input, result['answer']))

        st.write(result['answer'])