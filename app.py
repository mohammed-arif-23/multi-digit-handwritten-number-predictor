from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from cnnmnist import ConvolutionNeuralNetwork
import cv2

app = Flask(__name__)

model = ConvolutionNeuralNetwork()
model.load_state_dict(torch.load('my_model.pth'))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = np.array(data['image'], dtype=np.float32)
    if image_data.size == 700*350:
        image = (image_data.reshape(350, 700) * 255).astype(np.uint8)
    elif image_data.size == 560*280:
        image = (image_data.reshape(280, 560) * 255).astype(np.uint8)
    else:
        image = (image_data.reshape(-1, int(image_data.size**0.5)) * 255).astype(np.uint8)
        image = cv2.resize(image, (700, 350), interpolation=cv2.INTER_NEAREST)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_imgs, bboxes = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 1000:
            continue
        digit = thresh[y:y+h, x:x+w]
        size = max(w, h)
        padded = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = digit
        digit_resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
        digit_imgs.append(digit_resized)
        bboxes.append((x, y))
    digit_imgs = [img for _, img in sorted(zip(bboxes, digit_imgs), key=lambda pair: pair[0][0])]
    predictions, accuracies = [], []
    for digit_img in digit_imgs:
        digit_img = digit_img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(digit_img).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output, pred = model(tensor)
            probabilities = torch.exp(output)
            total = probabilities.sum(dim=1, keepdim=True)
            accuracy = (probabilities / total).max(dim=1)[0].item()
        predictions.append(str(pred.item()))
        accuracies.append(accuracy)
        print(predictions, accuracies)
    avg_prediction = ''.join(predictions) if predictions else ''
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    return jsonify({'Prediction': avg_prediction, 'Accuracy': avg_accuracy})

if __name__ == '__main__':
    app.run()
