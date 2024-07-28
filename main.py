# main.py
import os
import json
import torch
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Define FastAPI app
app = FastAPI()

# Load model data
MODEL_FILE_NAME = 'chatdata.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = os.path.dirname(__file__)
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

# Load the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def predict(message: str):
    sentence = tokenize(message)
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

@app.get("/")
async def get():
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Chat</title>
        </head>
        <body>
            <h1>Chat with Bot</h1>
            <form action="" onsubmit="sendMessage(event)">
                <input type="text" id="messageText" autocomplete="off"/>
                <button>Send</button>
            </form>
            <ul id='messages'>
            </ul>
            <script>
                var ws = new WebSocket("ws://localhost:8000/ws");
                ws.onmessage = function(event) {
                    var messages = document.getElementById('messages')
                    var message = document.createElement('li')
                    var content = document.createTextNode(event.data)
                    message.appendChild(content)
                    messages.appendChild(message)
                };
                function sendMessage(event) {
                    var input = document.getElementById("messageText")
                    var message = JSON.stringify({ "message": input.value });
                    ws.send(message)
                    input.value = ''
                    event.preventDefault()
                }
            </script>
        </body>
    </html>
    """
    return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data).get("message")
            except json.JSONDecodeError:
                message = data
            response, _ = predict(message)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
