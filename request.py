import requests

url = "http://127.0.0.1:8000/chat"
data = {"message": "how can I support you"}

response = requests.post(url, json=data)
print(response.json())
