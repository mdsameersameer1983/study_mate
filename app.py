import os

from flask import Flask, jsonify, render_template, request
from google import genai

from chatbot_config import SYSTEM_PROMPT

app = Flask(__name__, template_folder=".")

MODEL_NAME = "gemini-3.1-flash-lite"


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return genai.Client(api_key=api_key)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        client = get_client()

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"{SYSTEM_PROMPT}\n\nUser message:\n{message}"
                        }
                    ],
                }
            ],
        )

        answer = (response.text or "").strip()

        if not answer:
            answer = "I could not generate a response. Please try again."

        return jsonify({"answer": answer})

    except Exception:
        return jsonify({
            "error": "Unable to process your request right now."
        }), 500


if __name__ == "__main__":
    app.run()
