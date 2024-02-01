from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import pymongo
import numpy as np
import heapq
import openai


app = Flask(__name__)
openai.api_key = "open_ai_key"
# Initialize the BERT model
model = SentenceTransformer('bert-base-uncased')

# MongoDB connection
client = pymongo.MongoClient("mongodb_uri")
db = client.rag

# Function to maintain top N similar items
def add_to_top_n(top_n, item, n=1):
    if len(top_n) < n:
        heapq.heappush(top_n, item)
    else:
        heapq.heappushpop(top_n, item)


def vectorize_prompt(prompt):
    return model.encode(prompt, convert_to_tensor=False)


# Find similar vectors of data based on the user prompt
def find_similar_documents(prompt_vector, collection_vectorized):
    top_n_similar_documents = []

    for document in collection_vectorized.find({}):
        if 'Name_vectors' in document:
            document_vector = np.array(document['Name_vectors'])
            similarity = np.dot(prompt_vector, document_vector) / (np.linalg.norm(prompt_vector) * np.linalg.norm(document_vector))
            add_to_top_n(top_n_similar_documents, (similarity, document), n=1)

    top_n_similar_documents.sort(key=lambda x: x[0], reverse=True)

    return top_n_similar_documents

# Re-vectorize the similar documents
def re_vectorize_documents(top_n_similar_documents):
    return [np.array(doc['Name_vectors']) for _, doc in top_n_similar_documents]

# Fetch and display relevant data
def fetch_relevant_data(collection, ids):
    relevant_data_list = []

    for id in ids:
        document = collection.find_one({'ID': id})
        if document:
            relevant_data = {key: value for key, value in document.items() if value == 1}
            relevant_data_list.append({"document_id": id, "relevant_data": relevant_data})

    return relevant_data_list

# Route for the homepage
@app.route('/')
def index():
    return render_template('bot.html')

# Route to handle prompt submission and data processing
@app.route('/process-prompt', methods=['POST'])
def process_prompt():
    user_prompt = request.form.get('prompt')
    prompt_vector = vectorize_prompt(user_prompt)

    collection_vectorized = db.vectorized_data
    top_n_similar_documents = find_similar_documents(prompt_vector, collection_vectorized)
    re_vectorized_data = re_vectorize_documents(top_n_similar_documents)

    collection_original = db.original_data
    top_n_similar_document_ids = [doc.get('ID', 'N/A') for _, doc in top_n_similar_documents]
    relevant_data = fetch_relevant_data(collection_original, top_n_similar_document_ids)

    return user_prompt, relevant_data

 
@app.route('/generate-response', methods=['POST'])
def generate_response():
    user_prompt, relevant_data = process_prompt()  # Call the previous function to get data

    # Properly concatenate data to form a single input string
    input_text = "here is the prompt: " + user_prompt + "\n" + "here is the data: " + "\n".join(map(str, relevant_data))

    # Make a request to the GPT API for text generation
    try:
        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",  
            messages=[
                {"role": "system", "content": "here is the user prompt and the data related to the prompt. Use this data and create a good response for the user"},
                {"role": "user", "content": input_text}
            ],
            max_tokens=300  # Adjust based on your requirements
        )
        generated_text = gpt_response.choices[0].message['content'].strip()
    except Exception as e:
        generated_text = f"Error generating response: {str(e)}"
    
    return jsonify({"generated_response": generated_text})



if __name__ == '__main__':
    app.run(debug=True)
