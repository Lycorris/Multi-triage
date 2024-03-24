import requests
TOKEN = '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'
API_URL = "https://api-inference.huggingface.co/models/google-bert/bert-base-uncased"
headers = {"Authorization": f"Bearer {TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "I wanna fuck [MASK].",
})

print(output)