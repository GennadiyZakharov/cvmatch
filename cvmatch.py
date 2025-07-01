#!/usr/bin/env python

import sys
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import PyPDF2
import os
from dotenv import load_dotenv

url="127.0.0.1:11434"
model="qwen2.5-coder:7b"

# Access the API key securely
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
# Streamlit app UI
print("=== 📄 ATS Resume Checker ===")

uploaded_file = sys.argv[1]
job_desc_file = sys.argv[2]
resume = ""
# Resume text extraction
if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume += page.extract_text() or ""
    elif uploaded_file.name.endswith(".docx"):
        resume = docx2txt.process(uploaded_file)
# Main logic on button press
with open(job_desc_file, "r") as f:
    job_desc = f.read()

# Cosine Similarity Scoring
vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform([resume, job_desc])
cos_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
# GPT-4 Resume Evaluation Prompt
prompt = (
    "Evaluate the following resume against the job description. "
    "Give a score out of 100, a short rationale, and improvement suggestions if any.\n\n"
    f"Resume:\n{resume}\n\nJob Description:\n{job_desc}"
)
# GPT-4 API Call
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)
evaluation = response.choices[0].message.content
passed = cos_sim >= 60
# Output Section
print("Results")
print(f"Cosine Similarity Score: {cos_sim:.2f} %")
print("**GPT-4 Evaluation:**")
print(evaluation)
print(f"### {'✅ PASS' if passed else '❌ FAIL'}")
print("---")

# Improved Resume Generator
print("📝 Generate ATS-Optimized Resume")
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
improved_response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": improve_prompt}],
    temperature=0.4
)
improved_resume = improved_response.choices[0].message.content
print("Improved Resume (ATS-Optimized)")
print(improved_resume)
