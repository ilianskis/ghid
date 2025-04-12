import openai
from PyPDF2 import PdfReader
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'

class RAGSystem:
    def __init__(self, pdf_folder_path, openai_api_key):
        self.pdf_folder_path = pdf_folder_path
        self.openai_api_key = openai_api_key
        self.documents = self.load_documents()
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorize_documents()

    def load_documents(self):
        documents = []
        for filename in os.listdir(self.pdf_folder_path):
            if filename.endswith('.pdf'):
                with open(os.path.join(self.pdf_folder_path, filename), 'rb') as f:
                    reader = PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text() or ''
                    documents.append(text)
        return documents

    def vectorize_documents(self):
        return self.vectorizer.fit_transform(self.documents)

    def get_most_similar_document(self, query):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors)
        most_similar_index = np.argmax(similarities)
        return self.documents[most_similar_index]

    def generate_response(self, query):
        most_similar_document = self.get_most_similar_document(query)
        openai.api_key = self.openai_api_key
        client = openai.OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please shorten very little your answerto save tokens. If the question is not related to the topic you should not to answer."},
                {"role": "user", "content": f"Document: {most_similar_document}\n\nQuestion: {query}"}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

rag_system = RAGSystem('pdfs', 'your-key')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Initialize user context if it doesn't exist
    if 'context' not in session:
        session['context'] = []

    # Append the current question to the user's context
    session['context'].append(question)

    # Generate response based on the current question
    response = rag_system.generate_response(question)
    return jsonify({'response': response, 'context': session['context']})

if __name__ == '__main__':
    app.run(debug=True)
