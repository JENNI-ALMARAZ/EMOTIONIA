import os
import re
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, render_template, redirect, Response
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def draw_cross(image, center, size=5, color=(0, 0, 255)):
    """Dibuja una cruz en la imagen en la posición especificada."""
    x, y = center
    cv2.line(image, (x - size, y - size), (x + size, y + size), color, 2)
    cv2.line(image, (x + size, y - size), (x - size, y + size), color, 2)

def process_image(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks_mapping = {
                33: 'left_eye_center', 263: 'right_eye_center',
                133: 'left_eye_inner_corner', 362: 'right_eye_inner_corner',
                130: 'left_eye_outer_corner', 359: 'right_eye_outer_corner',
                55: 'left_eyebrow_inner_end', 285: 'right_eyebrow_inner_end',
                105: 'left_eyebrow_outer_end', 334: 'right_eyebrow_outer_end',
                1: 'nose_tip', 61: 'mouth_left_corner', 291: 'mouth_right_corner',
                0: 'mouth_center_top_lip', 17: 'mouth_center_bottom_lip'
            }

            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])

                if idx in landmarks_mapping:
                    draw_cross(image, (x, y))

    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Leer la imagen en memoria
            in_memory_file = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
            
            # Procesar la imagen
            processed_image = process_image(image)

            # Convertir la imagen procesada a formato JPEG para mostrar en el navegador
            _, buffer = cv2.imencode('.jpeg', processed_image)
            processed_image_bytes = buffer.tobytes()

            # Enviar la imagen procesada como respuesta
            return Response(processed_image_bytes, mimetype='image/jpeg')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
