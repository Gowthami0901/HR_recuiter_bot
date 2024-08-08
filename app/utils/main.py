from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate 
from dotenv import load_dotenv
import os

class ChatBot():
  load_dotenv()
  file_path = './app/utils/it_sector.txt'
  with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
  docs = text_splitter.split_text(text)
  embeddings = HuggingFaceEmbeddings()
  vectorstore = FAISS.from_texts(texts=docs, embedding=embeddings)

  repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
  llm = HuggingFaceHub(
      repo_id=repo_id, model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50}, huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
  )

  template = """
  You are an IT support chatbot. If the user greets you, respond politely and let them know you are here to assist with IT support questions.
  If they ask a question about the IT sector, use the following context to answer the question.
  If the question is out of context, respond politely that you are an IT support chatbot and can only assist with IT-related questions.
  If you don't know the answer, just say you don't know.
  You should respond with short and concise answers, no longer than 2 sentences.

  Context: {context}
  Question: {question}
  Answer:
  """

  prompt = PromptTemplate(template=template, input_variables=["context", "question"])

  rag_chain = (
    {"context": vectorstore.as_retriever(),  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
  )