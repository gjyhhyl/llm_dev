import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import TextLoader
import os
import warnings
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


warnings.filterwarnings("ignore")
# 加载 .env 文件中的环境变量
load_dotenv()
bge_path = os.getenv('BGE_MODEL_PATH')
milvus_user = os.getenv('MILVUS_USER')
milvus_password = os.getenv('MILVUS_PASSWORD')
milvus_host = os.getenv('MILVUS_HOST')
milvus_port = os.getenv('MILVUS_PORT')

# 生成嵌入
embeddings = HuggingFaceEmbeddings(model_name=bge_path)

def main():
    pdf_loader = PyPDFLoader('E10 6.0 培训教材-采购管理.pdf', extract_images=True)  # 使用OCR解析pdf中图片里面的文字
    docs = pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10))

    Milvus.from_documents(
        docs,
        embedding=embeddings,  # 嵌入向量，需要根据具体 embedding 计算
        collection_name="pm_pdf",
        connection_args={
            "user": milvus_user,
            "password": milvus_password,
            "host": milvus_host,
            "port": milvus_port
        }
    )
    print(0)# 文档上传成功


if __name__ == "__main__":
    main()
    # file_path = '/Users/ymnl/Desktop/API.txt'