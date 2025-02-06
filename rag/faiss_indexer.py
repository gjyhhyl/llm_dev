from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# 解析PDF，切成chunk片段
pdf_loader=PyPDFLoader('E10 6.0 培训教材-采购管理.pdf',extract_images=True)   # 使用OCR解析pdf中图片里面的文字
chunks=pdf_loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10))

# 加载embedding模型，用于将chunk向量化
embeddings = HuggingFaceEmbeddings(model_name='/Users/ymnl/Desktop/agent/huggingface_models/bge-base-zh-v1.5')

# 将chunk插入到faiss本地向量数据库
vector_db=FAISS.from_documents(chunks,embeddings)
vector_db.save_local('LLM.faiss')

print('faiss saved!')