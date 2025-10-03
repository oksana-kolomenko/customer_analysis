from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

MODEL_NAME = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message")

    system_prompt = (
        "You are a helpful customer service assistant for ACME Corp. "
        "Answer questions about orders, shipping, returns, and products politely.\n"
    )
    prompt = system_prompt + "User: " + user_message + "\nBot:"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Bot:" in reply:
        reply = reply.split("Bot:")[-1].strip()

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
