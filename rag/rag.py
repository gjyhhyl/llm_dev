from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import LLMChain
import os 

# 加载embedding模型，用于将query向量化
embeddings = HuggingFaceEmbeddings(model_name='/Users/ymnl/Desktop/agent/huggingface_models/bge-base-zh-v1.5')

# 加载faiss向量库，用于知识召回
vector_db=FAISS.load_local('LLM.faiss',embeddings, allow_dangerous_deserialization=True)
retriever=vector_db.as_retriever(search_kwargs={"k":5})

chat=ChatOpenAI(openai_api_key="sk-p2yK6ZkxTSiAObHEE11b5aE49f50479a855368016e83C0Ad", openai_api_base="https://openkey.cloud/v1")

# Prompt模板
prompt_template = """
You are a helpful assistant,().
Answer the question based only on the following context:

{context}

History: {history}
Question: {question}
Please answer the question in Chinese.
"""



# LLM chain
PROMPT = PromptTemplate(template=prompt_template, input_variables=["history", "question", "context"])
llm_chain = LLMChain(prompt=PROMPT, llm=chat)


import gradio as gr


chat_history = []

def chat(query):
    global chat_history
    response = llm_chain.run(history=chat_history, question=query, context=retriever)
    chat_history.extend([HumanMessage(content=query), response])
    chat_history = chat_history[-20:]  
    return response

# 创建 Gradio 接口
iface = gr.Interface(
    fn=chat,  
    inputs=gr.Textbox(label="Enter your query"),  
    outputs=gr.Textbox(label="Response"),  
    title="业务知识检索",  
    description=" ",  
)

# 启动界面
iface.launch()
