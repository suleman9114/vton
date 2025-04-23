import requests

API_URL = "https://m0w3ywx9kin4nrxb.us-east-1.aws.endpoints.huggingface.cloud"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def check_health():
    try:
        response = requests.get(f"{API_URL}/health", headers=headers)
        if response.status_code == 200:
            print(f"Health check successful: {response.json()}")
        else:
            print(f"Health check failed with status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error connecting to the API: {e}")

if __name__ == "__main__":
    check_health() 