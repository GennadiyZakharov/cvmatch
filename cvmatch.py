#!/usr/bin/env python

import json
import argparse
from http import client

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import PyPDF2

def calculate_cosine_sim(resume, job_desc):
    # Cosine Similarity Scoring
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume, job_desc])
    cos_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    return cos_sim

def call_ollama_api(url, model, prompt, temperature=0.0):
    try:
        conn = client.HTTPConnection(url)
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,  # Disable streaming to get full response
            "temperature": temperature
        }
        conn.request("POST", "/api/chat", json.dumps(payload), headers)
        response = conn.getresponse()
        data = response.read().decode('utf-8')

        # Parse the JSON response
        response_json = json.loads(data)

        # Extract the actual message content
        if 'message' in response_json and 'content' in response_json['message']:
            return response_json['message']['content']

        return data

    except (client.HTTPException, json.JSONDecodeError) as e:
        print(f"Error connecting to Ollama API: {e}")
        return None

url="127.0.0.1:11434"
model="qwen2.5-coder:7b"

# Set up argument parser
parser = argparse.ArgumentParser(description='ATS CV Checker - Compare your resume against a job description')
parser.add_argument('resume_file', help='Path to your resume file (PDF or DOCX format)')
parser.add_argument('job_description_file', help='Path to the job description text file')
args = parser.parse_args()

uploaded_file = args.resume_file
job_desc_file = args.job_description_file
resume = ""
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
    job_desc = f.read()

print("*** 📄 ATS CV Checker ***")
# GPT-4 Resume Evaluation Prompt
prompt = (
    "Evaluate the following resume against the job description. "
    "Give a score out of 100, a short rationale, and improvement suggestions if any.\n\n"
    f"Resume:\n{resume}\n\nJob Description:\n{job_desc}"
)
# Ollama API Call
print("GPT-4 Evaluation in progress:")
evaluation = call_ollama_api(url, model, prompt)


# Output Section
print("**GPT-4 Evaluation:**")
print(evaluation)

# Improved Resume Generator
improve_prompt = f"""
You are an expert resume coach and editor. Rewrite the following resume to be optimized for the job description below.
- Incorporate missing skills, tools, or responsibilities based on the job.
- Keep original experiences factual but improve alignment.
- Use clear, ATS-friendly formatting and job-aligned language.
- Do not invent roles or exaggerate.
Resume:
{resume}
Job Description:
{job_desc}
"""
improved_resume = evaluation = call_ollama_api(url, model, improve_prompt, temperature=0.4)

print("*** Improved Resume (ATS-Optimized) ***")
print(improved_resume)

print("**Similarity check:**")
cos_sim = calculate_cosine_sim(resume, job_desc)
passed = cos_sim > 70
cos_sim_improved = calculate_cosine_sim(improved_resume, job_desc)
passed_improved = cos_sim_improved > 70

print(f"Cosine Similarity Score (original): {cos_sim:.2f},   {'✅ PASS' if passed else '❌ FAIL'}")
print(f"Cosine Similarity Score (improved): {cos_sim_improved:.2f},   {'✅ PASS' if passed_improved else '❌ FAIL'}")
print("***\n")
