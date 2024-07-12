import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from src.base_model.llm_model import get_huggingface_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA
from logs.log import get_logger

# Logger
logger = get_logger(__name__)

llm = get_huggingface_llm(model_name="src/base_model/models", temperature=0.9)
logger.info("Loaded HuggingFace LLM models")
genai_docs = "data_sources/gen_ai"

# -----------------------Chains-------------

genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

# ---------------------- App - FastAPI----------------
app = FastAPI(
    title="Rag Langchain API",
    version="1.0",
    description="A Simple Rag Langchain API Runnable Interface",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---------------------- Routes - FastAPI----------------
@app.get("/check")
async def check():
    logger.info("Checking Rag Langchain API OK")
    return {"status": "ok"}

@app.post("/gen_ai", response_model=OutputQA)
async def gen_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}

# ---------------------- LangServe Routes - PlayGround----------------
add_routes(app,
           genai_chain,
           playground_type="default",
           path="/gen_ai")

