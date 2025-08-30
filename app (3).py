import os
import io
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from docx import Document
import requests

# --- CONFIGURATION ---
# IMPORTANT: Get your Hugging Face API token from https://huggingface.co/settings/tokens
# Replace the placeholder below with your actual token.
HF_API_TOKEN="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"

# Models to use (safe defaults). You can change to other HF model ids if desired.
MODEL_SIMPLIFY = "google/flan-t5-small"
MODEL_NER = "dslim/bert-base-NER"

API_URL_SIMPLIFY = f"https://api-inference.huggingface.co/models/{MODEL_SIMPLIFY}"
API_URL_NER = f"https://api-inference.huggingface.co/models/{MODEL_NER}"

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- HELPER FUNCTIONS FOR HUGGING FACE INFERENCE ---
    
def query_simplify(payload):
    """Sends a payload to the text simplification model."""
    response = requests.post(API_URL_SIMPLIFY, headers=headers, json=payload)
    return response.json()

def query_ner(payload):
    """Sends a payload to the NER model."""
    response = requests.post(API_URL_NER, headers=headers, json=payload)
    return response.json()

# --- TEXT EXTRACTION ---

def extract_text_from_pdf(file_stream):
    """Extracts text from a PDF file."""
    reader = PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_stream):
    """Extracts text from a DOCX file."""
    doc = Document(file_stream)
    text = " ".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def extract_text_from_txt(file_stream):
    """Extracts text from a TXT file."""
    return file_stream.read().decode('utf-8')

def extract_text(file, filename):
    """Determines file type and extracts text accordingly."""
    extension = filename.rsplit('.', 1)[-1].lower()
    file_stream = io.BytesIO(file.read())
    
    if extension == 'pdf':
        return extract_text_from_pdf(file_stream)
    elif extension == 'docx':
        return extract_text_from_docx(file_stream)
    elif extension == 'txt':
        return extract_text_from_txt(file_stream)
    else:
        # Handle unsupported formats
        raise ValueError("Unsupported file type. Supported formats are: PDF, DOCX, TXT.")

# --- FLASK APP ---

app = Flask(__name__)
# Allow requests from all origins. In a real app, you'd restrict this.
CORS(app) 

# ---------- API ENDPOINT -----------
@app.route("/analyze-document", methods=["POST"])
def analyze_document():
    """Endpoint to receive a document, analyze it, and return results."""
    
    if 'document' not in request.files:
        return jsonify({"error": "No document part in the request"}), 400
    
    file = request.files['document']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    analysis = {}
    try:
        document_text = extract_text(file, file.filename)
        
        # Check if the document is empty or too short for meaningful analysis
        if not document_text or len(document_text) < 10:
             return jsonify({"error": "The document is empty or too short for analysis."}), 400

        analysis["document_text"] = document_text[:500] + "..." # Truncate for display
        
        # 1. CLAUSE SIMPLIFICATION
        input_text_simplify = f"Simplify the following legal text: {document_text}"
        response = query_simplify({"inputs": input_text_simplify})
        if response and isinstance(response, list) and 'generated_text' in response[0]:
            analysis["simplified_text"] = response[0]["generated_text"]
        else:
            analysis["simplified_text"] = "Simplification failed. Check the model and API."

        # 2. NAMED ENTITY RECOGNITION (NER)
        ner_response = query_ner({"inputs": document_text})
        if ner_response and isinstance(ner_response, list):
            analysis["named_entities"] = {}
            for entity in ner_response:
                entity_type = entity.get("entity_group")
                entity_word = entity.get("word")
                if entity_type not in analysis["named_entities"]:
                    analysis["named_entities"][entity_type] = []
                # Ensure each word is added only once
                if entity_word not in analysis["named_entities"][entity_type]:
                    analysis["named_entities"][entity_type].append(entity_word)
        else:
            analysis["named_entities"] = {"error": "NER failed. Check the model and API."}

        # 3. CUSTOM PATTERN RECOGNITION (e.g., Dates, Money, Parties)
        date_matches = re.findall(r'\b\d{1,2}(?:st|nd|rd|th)? \w+ \d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', document_text)
        money_matches = re.findall(r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b|EUR\s*\d+|\£\s*\d+', document_text)
        parties_re = re.compile(r'(?<=\s)(LLC|Corp|Inc|Company|Association|Trust|Fund)\b', re.IGNORECASE)
        parties = list(set([match.group(0) for match in parties_re.finditer(document_text)]))

        analysis["highlights"] = {
            "money": list(dict.fromkeys(money_matches))[:20],
            "dates": list(dict.fromkeys(date_matches))[:20],
            "parties": parties[:20]
        }
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during analysis: {e}"}), 500

    return jsonify({"analysis": analysis}), 200

# ---------- START THE SERVER ----------
# This will run the Flask app on localhost (127.0.0.1) on port 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)