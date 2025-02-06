from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import tool
import os
import streamlit as st
import streamlit.components.v1 as components
from plotly.graph_objs import Figure
from pydantic import BaseModel
from streamlit_chat import message
import requests

os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL

"""定义prompt"""
prompt_template = """
You are a powerful assistant and your responses should always be text. But you do not know how to Predict the sales of automotive parts with product ID.
{input}
"""
prompt_template = PromptTemplate.from_template(prompt_template)

"""定义模型"""
llm = ChatOpenAI(temperature=0.0)
"""定义工具"""
@tool
def get_word_length(word):
    """get the length of a word"""
    return len(word)

@tool
def call_prediction_model():
    """This function is used to invoke a prediction model to forecast the sales of automotive parts with the given product ID for the next five weeks.
"""
    # url =
    # headers =
    # data =

    # response = requests.post(url, headers=headers, json=data)
    return "this is api"
def agent_openai_construct():

    tools = [get_word_length, call_prediction_model]
    """定义agent"""
    agent = initialize_agent(
        tools,
        llm,
        agent="structured-chat-zero-shot-react-description",
        verbose=True
    )
    return agent

if __name__ == '__main__':
    """调用agent"""
    agent = agent_openai_construct()

    print(agent.run("Calculate the length of the word 'strawberry'."))


