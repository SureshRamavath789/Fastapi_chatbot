import json
import torch
from serve import model_fn, input_fn, predict_fn, output_fn

# Initialize the model
model_dir = '.'  # Adjust if your model directory is different
model = model_fn(model_dir)

# Define the input JSON
input_json = {
    "message": "how can I support you"
}

# Simulate the request
request_body = json.dumps(input_json)
request_content_type = 'application/json'

# Process the input
input_data = input_fn(request_body, request_content_type)
prediction = predict_fn(input_data, model)
response_content_type = 'application/json'

# Generate and print the output
output = output_fn(prediction, response_content_type)
print(output)
