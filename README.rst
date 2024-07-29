docker build -t classify .

#docker run --rm classify
docker run -p 80:80 classify

#CURL
curl -X POST "http://localhost/predict/" -H "Content-Type: application/json" -d '{"cycle_number": 110, "model_name": "RandomForestClassifier"}'

#PYTHON
import requests

url = "http://localhost/predict/"
payload = {
    "cycle_number": 110,
    "model_name": "RandomForestClassifier"
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
