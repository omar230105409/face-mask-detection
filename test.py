import keras
import cv2

import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = keras.models.load_model("models/mask_detector_model.keras")
test_img = cv2.imread("man.png")
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(test_img_gray, 1.1, 4)

for(x,y,w,h) in faces:
    test_img = test_img[y:y+h, x:x+w]

test_img = cv2.resize(test_img,(224,224))
test_img_array = np.array(test_img).astype("float32") / 255.0
test_img_array = np.expand_dims(test_img_array, axis=0)
prediction = model.predict(test_img_array)
result = np.argmax(prediction)

print("Mask Detected!" if result == 1 else "No Mask Detected")
print(f"Raw Prediction Probabilities: {prediction}")

