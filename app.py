from nlp_engine import process_text, summarize_text 
from flask import Flask, render_template, request, jsonify
# IMPORT CHANGE: We now import 'summarize_text' alongside 'process_text'

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Get data from the Frontend
    data = request.get_json()
    
    text = data.get('text', '')
    mode = data.get('mode', 'general')    # Mode 1, 2, or 3
    action = data.get('action', 'check')  # 'summarize', 'readability', or 'check'
    tone = data.get('tone', 'neutral')    # 'formal', 'academic', etc.

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # 2. Run the Base Analysis (Grammar, Spelling, Stats)
    # We do this first so we always have grammar corrections available
    results = process_text(text)
    
    # 3. Handle "Text Summarization" Action
    if action == 'summarize':
        # Call the new Sumy function we added to nlp_engine.py
        summary = summarize_text(text, num_sentences=2)
        
        # Overwrite the output text with the summary
        results['corrected_text'] = f"📝 **Summary:**\n{summary}"


    # 5. Handle Tone Changes (Only if we are NOT summarizing)
    elif tone in ['formal', 'academic']:
        # Get the grammatically correct text first
        improved_text = results['corrected_text']
        
        # Apply simple Formal/Academic replacements
        replacements = {
            "can't": "cannot",
            "don't": "do not",
            "won't": "will not",
            "I'm": "I am",
            "it's": "it is",
            "gonna": "going to",
            "wanna": "want to",
            "kids": "children",
            "guys": "individuals"
        }
        
        for old, new in replacements.items():
            # simple case-insensitive replacement could be added here, 
            # but this works for standard typing
            improved_text = improved_text.replace(f" {old} ", f" {new} ")
            
        results['corrected_text'] = improved_text

    # 6. Return the final JSON to the UI
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)