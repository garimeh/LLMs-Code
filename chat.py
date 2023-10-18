import os
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from IPython.display import display
import ipywidgets as widgets
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


os.environ["OPENAI_API_KEY"] = "<insert key>"

# initialize pinecone
pinecone.init(
    api_key="<insert key",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
)

filename = "scraped data.txt"
chunk_size = 1024 # Adjust the chunk size as needed
index_name = "ind"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    length_function=len,
)

embeddings = OpenAIEmbeddings()

# Connect to an existing Pinecone index
index = pinecone.Index(index_name=index_name)
docsearch = Pinecone(index, embedding_function=embeddings, text_key="text")

with open(filename, "r") as f:
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break

        # Split the text into chunks
        docs_chunks = text_splitter.split_text(chunk)

        # Add the data to the Pinecone index
        docsearch.upsert(items=docs_chunks)

query = "How to log in to my account"
docs = docsearch.similarity_search(query)
print(docs[0])
