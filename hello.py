# Para instalar dependencias: pip install llama-cpp-python flask
from llama_cpp import Llama
from flask import Flask, jsonify, request, render_template, stream_with_context

app = Flask(__name__)

def get_llm(model_path: str) -> Llama:
    return Llama(
        model_path,
        seed=0,
        n_ctx=2048,
        n_gpu_layers=0
    )

MODEL_PATH = "/Users/mb99219/Documents/llama/Llama-3.2-1B-Instruct-Q4_1.gguf"
llm = get_llm(MODEL_PATH)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask_bot', methods=['POST'])
def chat():
    output = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": request.json['question']
            }
        ],
        max_tokens=1024,
        temperature=0.8,
        stream=True
    )

    def response_coroutine():
        for c in output:
            yield c['choices'][0]['delta'].get('content', '')

    return app.response_class(stream_with_context(response_coroutine()))


if __name__ == '__main__':
    app.run(
        host='localhost',
        port=5002,
        debug=True
    )