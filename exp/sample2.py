
from dotenv import load_dotenv
load_dotenv()

from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma.vectorstores import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain import hub
import os 

PATH = r"C:\Users\anant\PycharmProjects\chatbot-agent\exp\file.pdf"
STORAGE = "./storedb"

llm = ChatOpenAI(model="gpt-4o",temperature=0.7)

documents = []
pdf_content = PyPDFLoader(PATH)
loader = pdf_content.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n"],
)

# for i in range(len(loader)):
#     documents.append(Document(metadata=loader[i].metadata, page_content=loader[i].page_content))

documents = text_splitter.split_documents(loader)

if not os.path.exists(STORAGE):
    vectorstore = Chroma.from_documents(documents=documents,embedding=OpenAIEmbeddings(),persist_directory=STORAGE)
else:
    vectorstore = Chroma(persist_directory=STORAGE,embedding_function=OpenAIEmbeddings())
    
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
        
chain = (
     {'context': retriever, "question": RunnablePassthrough()}
     | prompt
     | llm
     | StrOutputParser()
)

query = "What is transformer?"
print(chain.invoke(query))