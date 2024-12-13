import cv2
import dlib
import math
import numpy as np
import streamlit as st

# Caminho para o preditor de landmarks
landmark_predictor_path = "data/shape_predictor_68_face_landmarks.dat"

# Funções auxiliares
def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def process_frame(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = shape_to_np(shape)

        chin = landmarks[8]
        nose_tip = landmarks[33]
        left_jaw = landmarks[0]
        right_jaw = landmarks[16]

        face_width = euclidean_distance(left_jaw, right_jaw)
        face_height = euclidean_distance(chin, nose_tip)

        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.line(frame, (nose_tip[0], nose_tip[1]), (chin[0], chin[1]), (255, 0, 0), 2)

        info_text_1 = f"Largura da face: {face_width:.2f}px"
        info_text_2 = f"Altura (nariz-queixo): {face_height:.2f}px"

        cv2.putText(frame, info_text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 255), 1)
        cv2.putText(frame, info_text_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 255), 1)
    
    return frame

# Interface Streamlit
st.title("Detecção Facial em Tempo Real")
st.write("Use sua webcam para visualizar a detecção de landmarks faciais.")

# Inicializa o detector e o preditor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_predictor_path)

# Configurações da câmera
run_camera = st.checkbox("Ativar webcam")
FRAME_WINDOW = st.image([])  # Placeholder para exibir os frames

if run_camera:
    camera = cv2.VideoCapture(0)  # Use 0 para a câmera padrão
    if not camera.isOpened():
        st.error("Não foi possível acessar a webcam.")
    else:
        while run_camera:
            ret, frame = camera.read()
            if not ret:
                st.error("Falha ao capturar o frame da câmera.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process_frame(frame, detector, predictor)
            FRAME_WINDOW.image(processed_frame)

            # Permite desligar a câmera ao desmarcar o checkbox
            if not st.session_state.get("run_camera"):
                break

        camera.release()
