import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
# from langchain.llms import CTransformers
from config import config

from huggingface_hub import login
login(token=config.hf_key)

from logs.log import get_logger

# logger
logger = get_logger(__name__)

# Quantization models 4bit
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="n4f",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# change to model vietnamese
def get_huggingface_llm(model_name: str = "src/base_model/models/vinallama-7b-chat_q5_0.gguf", max_new_token=1024, **kwargs):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        low_cpu_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_token=max_new_token,
        pad_on_left=tokenizer.eos_token_id,
        # device="auto"
        device=torch.device('gpu')
    )
    logger.info("Number of GPU usage {}".format(torch.cuda.device_count()))

    # embedding model
    llm_model = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs=kwargs
    )

    return llm_model

