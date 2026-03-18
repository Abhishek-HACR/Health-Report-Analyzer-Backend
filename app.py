import os
from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv
from utils.pdf_reader import extract_text_from_pdf
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_medical_report(report_text):

    prompt = f"""
You are an AI medical report analyzer.

Analyze the following medical report and produce:

1. Positive Health Indicators (5 points)
2. Negative Health Indicators (5 points)
3. Abnormal Values (congratulate/ALERT and highlight any values that are outside the normal range)
4. Health Suggestions
5. Health Score (0-10)(user ko uski medical reports ke basis per main reason bhi batao)
6. Health Risk Summary
7. Disclaimer

Medical Report:
{report_text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

@app.route("/")
def home():
    return "Medical Report Analyzer API is running"


@app.route("/analyze", methods=["POST"])
def analyze():
    
    os.makedirs("uploads", exist_ok=True)

    print("Request received")

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    print("File name:", file.filename)

    path = os.path.join("uploads", file.filename)
    file.save(path)

    print("File saved")

    text = extract_text_from_pdf(path)

    print("PDF text extracted")

    result = analyze_medical_report(text)

    print("AI result generated")

    return jsonify({
    "analysis": result,
    "report_text": text
})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data["question"]
    report_text = data["report"]

    prompt = f"""
You are the AI assistant of the project "AI Medical Report Analyzer".

Creator: Rahul

Your role:
- Explain medical reports
- Answer user questions about the report
- Be simple and clear

If someone asks who created you, respond:
"I am part of the AI Medical Report Analyzer project created by ABHISHEK ."

Medical report:
{report_text}

User question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    return jsonify({"answer": answer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)