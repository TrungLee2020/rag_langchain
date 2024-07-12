import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Reranker
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

from logs.log import get_logger

# logger
logger = get_logger(__name__)
class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(self, text_response: str, pattern: str = r"Answer:\s*{.*}") -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response
class Offline_Rag:
    def __init__(self, llm) -> None:
        self.llm = llm
            # prompt:
                # You are an assistant for question-answering tasks.
                # Use the following pieces of retrieved context to answer the question.
                # If you don't know the answer, just say that you don't know.
                # Use three sentences maximum and keep the answer concise.
                # Question: {question}
                # Context: {context}
                # Answer:
        self.prompt = hub.pull("rlm/rag-prompt")
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever):
        compressor = CohereRerank(top_k=3) # top_k: Number of documents to return
        logger.info(f"CohereRerank with top_k = 3")
        # Rerank retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

        # Setup prompt
        input_data = {
            "context": compression_retriever | self.format_docs,
            "question": RunnablePassthrough()
        }

        rag_chain = (
            input_data | self.prompt | self.llm | self.str_parser
        )
        return rag_chain

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
