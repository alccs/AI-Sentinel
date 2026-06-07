import requests
import base64
import json
import sys

# Test image (create a typical 1080p camera frame to see if size is rejected)
from PIL import Image
import io

img = Image.new('RGB', (1920, 1080), color = 'black')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='JPEG')
base64_image = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

payload = {
    "model": "qwen3.5-0.8b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ],
    "max_tokens": 50,
    "temperature": 0.1
}

print(f"Sending 1080p resolution request to LM Studio for model: {payload['model']}")
try:
    response = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload, timeout=30)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error detail: {response.text}")
    else:
        print(f"Success: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Exception: {e}")
