# import tensorflow as tf
import datetime
from tflite_runtime.interpreter import Interpreter 
from tflite_runtime.interpreter import load_delegate 
from PIL import Image
import numpy as np
#def preprocess(img):
#    img = cv2.resize(img, (1344,448))
#    return (img).reshape(1,1344,448,3).astype(np.uint8)

img_path = 'IMG_20210201_171916.jpg'
#img = cv2.imread(img_path)
#img = preprocess(img)
image = Image.open(img_path).convert('RGB').resize((1344,448), Image.ANTIALIAS)
path = 'model.tflite'
interpreter = Interpreter(model_path=path)
# interpreter = Interpreter(model_path=path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# input_index = interpreter.get_input_details()[0]["index"]
# output_index = interpreter.get_output_details()[0]["index"]

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

img = np.array(image).reshape(1,1344,448,3).astype(np.uint8)
print("allocating tensor")
interpreter.allocate_tensors()
print("setting tensor")
interpreter.set_tensor(input_index, img)
print("invoking")
interpreter.invoke()

print(datetime.datetime.now())
pred = interpreter.get_tensor(output_index) > 0.5
print(pred.shape)
print(datetime.datetime.now())
