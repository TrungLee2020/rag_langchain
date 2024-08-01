from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

from typing import List, Dict, Any
import uvicorn
from find_qs import ComplexQuestionHandler
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import requests
import logging
import json
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
VALID_HASH = "827ccb0eea8a706c4c34a16891f84e7b"  # md5 hash of "12345"

# Initialize ComplexQuestionHandler
handler = ComplexQuestionHandler('data_sources/db.csv')

# ThreadPoolExecutor for concurrent question processing
executor = ThreadPoolExecutor(max_workers=10)  # Adjust the number of workers as needed

# Initialize global variables for model and tokenizer
MODEL_NAME = "vilm/vinallama-2.7b"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer once at startup
logger.info("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=nf4_config,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, device_map="auto")
llm = HuggingFacePipeline(pipeline=model_pipeline)
prompt = hub.pull("rlm/rag-prompt")
logger.info("Model and tokenizer loaded successfully.")

# Load JSON data once at startup
# logger.info("Loading JSON data...")
# with open('data.json', 'r', encoding='utf-8') as file:
#     json_data = json.load(file)
# logger.info("JSON data loaded successfully.")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

def verify_hash(provided_hash: str) -> bool:
    return provided_hash == VALID_HASH

# Function to load JSON from file
def load_json_from_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        if isinstance(data, list):
            return data
        else:
            return [data]
class QuestionModel(BaseModel):
    question: str
    authen_pass: str

class QuestionResponse(BaseModel):
    question_ids: List[str]
    std_questions: List[str]

class QueryModel(BaseModel):
    link: str
    authen_pass: str
    use_local: bool = False
    local_file_path: str = None


async def process_question(question: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, handler.process_complex_question, question)

    return {
        "question_ids": [str(qid) for qid in result['question_ids']],
        "std_questions": result['std_questions'],
    }

@app.post("/recieve_question", response_model=QuestionResponse)
async def recieve_question(data: QuestionModel):
    if not verify_hash(data.authen_pass):
        logger.error("Invalid hash")
        logger.info(f"-"*50)
        raise HTTPException(status_code=403, detail="Invalid hash")

    try:
        result = await process_question(data.question)
        return QuestionResponse(**result)
    except Exception as e:
        logger.error(e)
        logger.info(f"-" * 50)
        raise HTTPException(status_code=500, detail=str(e))

# Function to process the loaded JSON and run the RAG chain
def process_json_and_run_rag(item: Dict[str, Any]) -> str:
    context = item["context"]
    std_answer = item.get("std_answer", "")
    std_question = item.get("std_question", "")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len
    )

    docs = text_splitter.split_text(context)

    vector_db = Chroma.from_texts(docs, embedding=embeddings)
    retriever = vector_db.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # Combine std_answer and context to create a more comprehensive answer
    combined_input = f"Question: {std_question}\nStandard Answer: {std_answer}\nAdditional Context: {context}"
    output = rag_chain.invoke(combined_input)

    # Extract the answer part and make it more conversational
    answer = output.split('Answer: ')[1].strip()
    conversational_answer = f"Đối với câu hỏi '{std_question}', tôi xin được chia sẻ như sau: {answer} .Tôi hy vọng thông tin này hữu ích cho bạn. Nếu bạn có bất kỳ câu hỏi nào khác, đừng ngần ngại hỏi nhé!"

    return conversational_answer

@app.post("/ask")
async def ask_question(query: QueryModel):
    if not verify_hash(query.authen_pass):
        logger.error("Invalid hash")
        logger.info(f"-" * 50)
        raise HTTPException(status_code=403, detail="Invalid hash")

    try:
        if query.use_local:
            if not query.local_file_path or not os.path.exists(query.local_file_path):
                raise HTTPException(status_code=400, detail="Invalid local file path")
            json_data = load_json_from_file(query.local_file_path)
            logger.info(f"Data loaded from local file: {query.local_file_path}")
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(query.link)
                response.raise_for_status()

                content_type = response.headers.get("content-type")
                if "application/json" not in content_type:
                    raise HTTPException(status_code=400, detail="URL không trả về JSON")

                json_data = response.json()
            logger.info(f"Data fetched successfully from backend API")

            # Send acknowledgment of receipt
            ack_response = requests.post(query.link, json={"status": "received", "message": "Data received successfully"})
            ack_response.raise_for_status()
            logger.info("Acknowledgment of data receipt sent to backend")

        if not json_data:
            raise ValueError("JSON data is empty")

        results = []
        for item in json_data:
            if all(key in item for key in ['std_question', 'question_id', 'context', 'std_answer']):
                try:
                    answer = process_json_and_run_rag(item)
                    results.append({
                        "context_id": item.get('context_id'),
                        "std_question": item['std_question'],
                        "answer": answer,
                        "question_id": item['question_id']
                    })
                except Exception as e:
                    logger.error(f"Error processing item: {e}")

        if not results:
            return {"message": "No valid results found"}

        if not query.use_local:
            # Send results back to the backend API
            logger.info(f"Sending results back to backend API")
            post_response = requests.post(query.link, json={"results": results})
            post_response.raise_for_status()

            # Wait for acknowledgment from backend
            ack_response = requests.post(query.link, json={"status": "sent", "message": "Results sent successfully"})
            ack_response.raise_for_status()
            logger.info(f"Results successfully sent to backend API")
            return {"message": "Results processed and sent to backend API successfully"}
        else:
            return {"results": results, "message": "Results processed successfully from local file"}

    except requests.RequestException as e:
        logger.error(f"Error communicating with backend API: {e}")
        raise HTTPException(status_code=500, detail=f"Error communicating with backend API: {str(e)}")
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def read_root():
    return {"status": "active", "message": "Server is ready to analyze questions and generate answers"}

if __name__ == "__main__":
    logger.info(f"Starting Server with {uvicorn.__version__}")
    uvicorn.run(app, host="0.0.0.0", port=8686)