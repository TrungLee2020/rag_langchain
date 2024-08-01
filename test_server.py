from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx
from typing import List, Dict, Any, Optional, Union
import uvicorn
from find_qs import ComplexQuestionHandler
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import torch
import logging
import json
from datetime import datetime
import pytz
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
VALID_HASH = "827ccb0eea8a706c4c34a16891f84e7b"  # md5 hash of "12345"

# Initialize ComplexQuestionHandler
handler = ComplexQuestionHandler('data_sources/db.csv')

# ThreadPoolExecutor for concurrent question processing
executor = ThreadPoolExecutor(max_workers=10)

# Initialize the Sentence Transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize a question-answering pipeline
# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Initialize mT5 model for text generation
# Load more advanced models
model_checkpoint = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
# qa_pipeline = pipeline("question-answering", model=model_checkpoint, tokenizer=model_checkpoint)



# llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer, max_length=1024, truncation=True)


def verify_hash(provided_hash: str) -> bool:
    return provided_hash == VALID_HASH


def load_json_from_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data if isinstance(data, list) else [data]


def get_current_time() -> str:
    hcm_tz = pytz.timezone('Asia/Ho_Chi_Minh')
    now = datetime.now(hcm_tz)
    return now.strftime("%H:%M:%S ngày %d-%m-%y")


def get_weather() -> str:
    api_key = "your_api_key"
    city = "Hanoi"
    response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric")
    data = response.json()
    temp = data['main']['temp']
    weather_description = data['weather'][0]['description']
    return f"Thời tiết hiện tại ở {city} là {temp}°C với {weather_description}."


def handle_specific_questions(question: str) -> str:
    question_lower = question.lower()
    if any(phrase in question_lower for phrase in ["tên", "bạn là ai", "hi", "hello", "xin chào"]):
        return "Xin chào! Tôi là một trợ lý ảo, bạn hãy gọi tôi là Lý. Chúc bạn một ngày tốt lành. Tôi có thể giúp gì được cho bạn ?"
    elif any(phrase in question_lower for phrase in ["bạn làm gì", "nghề nghiệp của bạn"]):
        return "Tôi là một trợ lý ảo, được sử dụng để trả lời các câu hỏi của bạn."
    elif any(phrase in question_lower for phrase in ["ai tạo ra bạn", "bạn đến từ đâu"]):
        return "Tôi là một trợ lý ảo, được team AI CNTT phát triển"
    elif any(phrase in question_lower for phrase in ["mấy giờ rồi", "bây giờ là mấy giờ", "hiện tại là mấy giờ"]):
        return f"Hiện tại là {get_current_time()}."
    # elif any(phrase in question_lower for phrase in ["thời tiết hôm nay thế nào?", "hôm nay thời tiết thế nào"]):
    #     return get_weather()
    else:
        return None


class QuestionModel(BaseModel):
    question: str
    authen_pass: str


class QuestionResponse(BaseModel):
    question_ids: List[str]
    std_questions: List[str]
    success: bool


class QueryModel(BaseModel):
    link: Optional[str] = Field(None, description="URL to fetch JSON data")
    authen_pass: str = Field(..., description="Authentication password")
    use_local: bool = Field(False, description="Flag to use local file")
    local_file_path: Optional[str] = Field(None, description="Path to local JSON file")


class ResultItem(BaseModel):
    context_id: Optional[str] = Field(None, description="ID of the context")
    std_question: str = Field(..., description="Standard question")
    answer: str = Field(..., description="Generated answer")
    question_id: str = Field(..., description="ID of the question")


class ResponseModel(BaseModel):
    answer: str = Field(..., description="Generated answer")
    success: bool


class AnswerDataModel(BaseModel):
    results: List[Dict[str, Any]]
    authen_pass: str
    use_local: bool
    local_file_path: Optional[str] = None
    link: Optional[str] = None


async def process_question(question: str):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, handler.process_complex_question, question)
    return {
        "question_ids": [str(qid) for qid in result['question_ids']],
        "std_questions": result['std_questions'],
    }


@app.post("/recieve_question", response_model=Union[QuestionResponse, ResponseModel])
async def recieve_question(data: QuestionModel):
    if not verify_hash(data.authen_pass):
        logger.error("Invalid hash")
        raise HTTPException(status_code=403, detail="Invalid hash")

    try:
        # Xử lý các câu hỏi cụ thể trước
        specific_answer = handle_specific_questions(data.question)
        if specific_answer:
            # Trả về câu trả lời cho câu hỏi cụ thể nếu có
            return ResponseModel(answer=specific_answer, success=True)

        # Nếu không phải câu hỏi cụ thể, xử lý theo quy trình chính
        result = await process_question(data.question)
        return QuestionResponse(**result, success=True)
    except Exception as e:
        logger.error(f"Error in recieve_question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def retrieve_relevant_context(question: str, contexts: List[str], top_k: int = 3) -> List[str]:
    question_embedding = sentence_model.encode([question])
    context_embeddings = sentence_model.encode(contexts)

    similarities = cosine_similarity(question_embedding, context_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [contexts[i] for i in top_indices]


def generate_answer(question: str, contexts: List[str]) -> str:
    relevant_contexts = retrieve_relevant_context(question, contexts)
    combined_context = " ".join(relevant_contexts)

    prompt = f"""Dựa trên thông tin sau đây:

{combined_context}

Hãy trả lời câu hỏi sau một cách chi tiết và chính xác:
{question}

Trả lời:"""

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=1024,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    # answer = qa_pipeline(prompt)
    return post_process_answer(answer)


def post_process_answer(answer: str) -> str:
    answer = answer.split("Trả lời:")[-1].strip()
    sentences = answer.split('. ')
    paragraphs = ['. '.join(sentences[i:i + 3]) for i in range(0, len(sentences), 3)]
    return '\n\n'.join(paragraphs)


def improved_answer_generation(json_data: List[Dict[str, Any]]) -> str:
    all_answers = []
    all_contexts = [item.get("context", "") for item in json_data]

    for item in json_data:
        std_question = item.get("std_question", "")
        if std_question:
            answer = generate_answer(std_question, all_contexts)
            all_answers.append(f"Câu hỏi: {std_question}\n\nTrả lời: {answer}\n")

    final_answer = "\n\n".join(all_answers)
    final_answer += "\n\nTôi hy vọng những thông tin này hữu ích cho bạn. Nếu bạn có bất kỳ câu hỏi nào khác, đừng ngần ngại hỏi nhé!"

    return final_answer


# Update the process_json_and_run_rag function
def process_json_and_run_rag(json_data: List[Dict[str, Any]]) -> str:
    try:
        answer = improved_answer_generation(json_data)
        return answer
    except Exception as e:
        logger.error(f"Error in improved_answer_generation: {str(e)}", exc_info=True)
        return f"Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại."


@app.post("/ask", response_model=ResponseModel)
async def ask_question(query: QueryModel):
    if not verify_hash(query.authen_pass):
        logger.error("Invalid hash")
        raise HTTPException(status_code=403, detail="Invalid hash")

    try:
        if query.use_local:
            if not query.local_file_path:
                raise ValueError("Local file path is required when use_local is True")
            if not os.path.exists(query.local_file_path):
                raise FileNotFoundError(f"Local file not found: {query.local_file_path}")
            json_data = load_json_from_file(query.local_file_path)
            logger.info(f"Data loaded from local file: {query.local_file_path}")
        else:
            if not query.link:
                raise ValueError("URL link is required when use_local is False")

            logger.info(f"Attempting to fetch data from: {query.link}")
            async with httpx.AsyncClient() as client:
                response = await client.get(query.link)
                response.raise_for_status()
                json_data = response.json()
            logger.info("Data fetched successfully from backend API")

        if not json_data:
            raise ValueError("JSON data is empty")

        logger.info("Start processing answer")
        answer = process_json_and_run_rag(json_data)
        # Check if the answer is empty
        if not answer or answer.strip() == "":
            logger.warning("Generated answer is empty")
            return ResponseModel(answer="Không có câu trả lời phù hợp. Bạn hãy cho tôi biết thêm thông tin!", success=False)

        return ResponseModel(answer=answer, success=True)

    except httpx.RequestError as e:
        logger.error(f"HTTP Request error in ask_question: {e}")
        raise HTTPException(status_code=500, detail=f"Error making HTTP request: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode error in ask_question: {e}")
        raise HTTPException(status_code=500, detail=f"Error decoding JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/send_answers", response_model=ResponseModel)
async def send_answers(data: AnswerDataModel):
    if not verify_hash(data.authen_pass):
        logger.error("Invalid hash")
        raise HTTPException(status_code=403, detail="Invalid hash")

    try:
        if not data.results:
            raise ValueError("No results to send")

        if data.use_local:
            with open(data.local_file_path, 'w', encoding='utf-8') as f:
                json.dump(data.results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to local file: {data.local_file_path}")
        else:
            if not data.link:
                raise ValueError("URL link is required when use_local is False")

            async with httpx.AsyncClient() as client:
                response = await client.post(data.link, json={"results": data.results})
                response.raise_for_status()

            logger.info("Results sent to backend API successfully")

        return ResponseModel(results=[ResultItem(**result) for result in data.results], success=True)

    except Exception as e:
        logger.error(f"Unexpected error in send_answers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def read_root():
    return {"status": "active", "message": "Server is ready to analyze questions and generate answers"}


if __name__ == "__main__":
    logger.info(f"Starting Server with {uvicorn.__version__}")
    uvicorn.run(app, host="0.0.0.0", port=8686)