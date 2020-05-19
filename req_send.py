import numpy as np
import requests

sample_input = np.random.rand(22)
to_send = sample_input.reshape(1, -1).tolist()
prediction = requests.post('http://localhost:8888', json={'data': to_send})
print(prediction.text)
