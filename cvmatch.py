#!/usr/bin/env python

import json
import argparse
import os
from http import client
from typing import Optional, Dict, List, Any, Union, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import PyPDF2

def calculate_cosine_sim(resume: str, job_desc: str) -> float:
    # Cosine Similarity Scoring
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume, job_desc])
    cos_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    return cos_sim

def read_prompt_file(file_path: str) -> str:
    """Read prompt template from file and return it."""
    try:
        with open(file_path, 'r') as f:
            prompt_template = f.read()
        return prompt_template
    except FileNotFoundError:
        print(f"Error: Prompt file not found: {file_path}")
        exit(1)
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        exit(1)

def call_ollama_api(url: str, model: str, prompt: str, temperature: float = 0.0) -> Optional[str]:
    try:
        conn = client.HTTPConnection(url)
        headers: Dict[str, str] = {
            "Content-Type": "application/json"
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,  # Disable streaming to get full response
            "temperature": temperature
        }
        conn.request("POST", "/api/chat", json.dumps(payload), headers)
        response = conn.getresponse()
        data: str = response.read().decode('utf-8')

        # Parse the JSON response
        response_json: Dict[str, Any] = json.loads(data)

        # Extract the actual message content
        if 'message' in response_json and 'content' in response_json['message']:
            return response_json['message']['content']

        return data

    except (client.HTTPException, json.JSONDecodeError) as e:
        print(f"Error connecting to Ollama API: {e}")
        return None

def pass_fail_str(is_passed: bool) -> str:
    return '✅ PASS' if is_passed else '❌ FAIL'

url: str = "127.0.0.1:11434"
model: str = "deepseek-r1:latest"

# Set up argument parser
parser = argparse.ArgumentParser(description='ATS CV Checker - Compare your resume against a job description')
parser.add_argument('resume_file', help='Path to your resume file (PDF or DOCX format)')
parser.add_argument('job_description_file', help='Path to the job description text file')
parser.add_argument('--eval-prompt', default=os.path.join("prompts", "evaluation_prompt.txt"), 
                    help='Path to custom evaluation prompt file')
parser.add_argument('--improve-prompt', default=os.path.join("prompts", "improve_prompt.txt"), 
                    help='Path to custom improvement prompt file')
args = parser.parse_args()

uploaded_file: str = args.resume_file
job_desc_file: str = args.job_description_file

resume: str = ""
# Resume text extraction
if uploaded_file:
    if uploaded_file.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume += page.extract_text() or ""
    elif uploaded_file.endswith(".docx"):
        resume = docx2txt.process(uploaded_file)
# Main logic on button press
with open(job_desc_file, "r") as f:
    job_desc: str = f.read()

print("*** 📄 ATS CV Checker ***")

# Read evaluation prompt from file
prompt_template: str = read_prompt_file(args.eval_prompt)
prompt: str = prompt_template.format(resume=resume, job_desc=job_desc)

# Ollama API Call
print("GPT-4 Evaluation in progress:")
evaluation: Optional[str] = call_ollama_api(url, model, prompt)


# Output Section
print("**GPT-4 Evaluation:**")
print(evaluation)

# Improved Resume Generator
improve_prompt_template: str = read_prompt_file(args.improve_prompt)
improve_prompt: str = improve_prompt_template.format(resume=resume, job_desc=job_desc)
improved_resume: Optional[str] = call_ollama_api(url, model, improve_prompt, temperature=0.4)

print("*** Improved Resume (ATS-Optimized) ***")
print(improved_resume)

print("**Similarity check:**")
cos_sim: float = calculate_cosine_sim(resume, job_desc)
passed: bool = cos_sim > 70
cos_sim_improved: float = calculate_cosine_sim(improved_resume or "", job_desc)
passed_improved: bool = cos_sim_improved > 70

print(f"Cosine Similarity Score (original): {cos_sim:.2f},   {pass_fail_str(passed)}")
print(f"Cosine Similarity Score (improved): {cos_sim_improved:.2f},   {pass_fail_str(passed_improved)}")
print("***\n")
