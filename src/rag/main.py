from pydantic import BaseModel, Field

from src.rag.file_loader import Loader
from src.rag.vectorstores import  VectorDB
from src.rag.offline_rag import Offline_Rag

class InputQA(BaseModel):
    question: str = Field(..., title='Question to ask model')

class OutputQA(BaseModel):
    answer: str = Field(..., title='Answer to ask model')

def build_rag_chain(llm, data_dir, data_type):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    retriever = VectorDB(documents=doc_loaded).get_retriever()
    rag_chain = Offline_Rag(llm).get_chain(retriever)
    return rag_chain