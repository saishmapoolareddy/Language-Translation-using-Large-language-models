from flask import Flask, render_template, request
import torch
import json
from eval import Evaluator

app = Flask(__name__)

# Load your model and other necessary setup
device = torch.device("cpu")
config_path = r'C:\Users\amrut\Downloads\eval\eval\config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

from_lang = 'fra'  # Update with your source language
to_lang = 'eng'  # Update with your target language
evaluator = Evaluator(config, device=device)

@app.route('/', methods=['GET'])
def InitialandReload():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def translate():
    translation = None
    input_text = None

    if request.method == 'POST':
        input_text = request.form.get('input_text', '')
        print("Input Text from Form:", input_text)
        expected, output_sentence = evaluator.web_eval(input_phrase=input_text, randomarg=False)
        translation = output_sentence
        print("Expected:", expected)
        print("Output Sentence:", translation)
        
    return render_template('index.html', translation=translation, input_text=input_text)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
