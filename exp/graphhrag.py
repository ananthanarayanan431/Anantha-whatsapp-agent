
import os 
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_neo4j.vectorstores import neo4j_vector
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_neo4j.graphs import neo4j_graph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from neo4j import GraphDatabase

embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

graph = neo4j_graph(
    uri=NEO4J_URI,
    user=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

