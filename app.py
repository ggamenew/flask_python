from flask import Flask
import os
import requests
import moondream as md
from PIL import Image
import time
import threading

app = Flask(__name__)

file_url = 'https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz'
file_name = 'moondream-0_5b-int8.mf.gz'
if not os.path.exists(file_name):
    response = requests.get(file_url)
    print('Downloading model file...', file_name)
    with open(file_name, 'wb') as f:
        f.write(response.content)
else:
    print('Model file already exists:', file_name)
    

model = None 

@app.route('/')
def home():
    return 'Hello, World!'


# Add a global lock
request_lock = threading.Lock()

@app.route('/predict')
def predict():
    startTime = time.time()
    global model
    
    # Acquire the lock - if another request is being processed, this will wait
    with request_lock:
        if model is None:
            model = md.vl(model='./moondream-0_5b-int8.mf.gz')  # Initialize model

        image = Image.open('./KkAfQv7.jpeg')
        encoded_image = model.encode_image(image)
        result = {
            "caption": model.caption(encoded_image)["caption"],
            "time": time.time() - startTime
        }
        
    # Lock is automatically released when the with block exits
    return result



# show 404 eror list of available endpoints
@app.errorhandler(404)
def page_not_found(e):
    # Automatically collect all registered endpoints
    endpoints = set()
    for rule in app.url_map.iter_rules():
        endpoints.add(rule.rule)
    endpoints_list = '<br/>- '.join(sorted(endpoints))
    return f'404 Not Found: The requested URL was not found.<br/>Available endpoints:<br/>- {endpoints_list}'


if __name__ == '__main__':
    app.run()