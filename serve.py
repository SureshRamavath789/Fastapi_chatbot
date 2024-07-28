import os
import json
import torch
import random
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

MODEL_FILE_NAME = 'chatdata.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model data
data_dir = os.path.join(os.path.dirname(__file__))
FILE = os.path.join(data_dir, MODEL_FILE_NAME)

with open(FILE, 'rb') as f:
    data = torch.load(f)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Load intents
with open(os.path.join(data_dir, "intents.json")) as json_data:
    intents = json.load(json_data)

def model_fn(model_dir):
    model_path = os.path.join(model_dir, MODEL_FILE_NAME)
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return data['message']  # Assuming the input JSON has a "message" field
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    sentence = tokenize(input_data)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    with torch.no_grad():
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

    if prob.item() > 0.8:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses']), intent.get('buttons', [])
    return "I do not understand...", []

def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        response = {"response": prediction[0], "buttons": prediction[1]}
        return json.dumps(response)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
