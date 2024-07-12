from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
from sklearn.feature_extraction.text import TfidfVectorizer

# Extract file docx with endswith = ?
def extract_qa_pairs_from_docx(docx_path):
    doc = Document(docx_path)
    qa_pairs = []
    current_question = ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            if text.endswith('?'):
                current_question = text
            elif current_question:
                qa_pairs.append((current_question, text))
                current_question = ""
    return qa_pairs

# Đọc dữ liệu từ file DOCX
docx_path = 'data_sources/FQA dịch vụ chi hộ.docx'
qa_pairs = extract_qa_pairs_from_docx(docx_path)
questions = [pair[0] for pair in qa_pairs]

# Sử dụng TF-IDF để vector hóa DB
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.85, min_df=2)
tfidf_matrix = vectorizer.fit_transform(questions)

# Tải mô hình vinallama cho Q&A
llm = Llama(model_path="models/vinallama-7b-chat_q5_0.gguf", n_ctx=2048, n_threads=4)

def get_most_relevant_qa(query, top_k=1):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [qa_pairs[i] for i in top_indices]

def generate_response(relevant_qa, question):
    context = "\n".join([f"Q: {qa[0]}\nA: {qa[1]}" for qa in relevant_qa])
    prompt = f"""Dưới đây là thông tin liên quan từ tài liệu:
            {context}
            Dựa vào thông tin trên, hãy trả lời câu hỏi sau một cách chính xác và ngắn gọn:
            Người dùng: {question}
            AI: """

    response = llm(prompt, max_tokens=50, stop=["Người dùng:", "\n"], echo=False)
    return response['choices'][0]['text'].strip()


# Main loop
if __name__ == '__main__':
    print("Chào bạn! Hãy đặt câu hỏi (gõ 'thoát' để kết thúc).")
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == 'thoát':
            print("AI: Tạm biệt!")
            break

        relevant_qa = get_most_relevant_qa(user_input)
        if relevant_qa and cosine_similarity(vectorizer.transform([user_input]), vectorizer.transform([relevant_qa[0][0]]))[0][0] > 0.8:
            print("AI:", relevant_qa[0][1])
        else:
            response = generate_response(relevant_qa, user_input)
            print("AI:", response)