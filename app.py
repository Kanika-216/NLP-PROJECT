from flask import Flask, render_template, request, jsonify
from nlp_engine import process_text

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    results = process_text(text)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)