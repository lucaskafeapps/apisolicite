import cv2
import streamlit as st

# Função para detectar faces
def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame, len(faces)

# Interface do Streamlit
st.title("Detecção Facial com OpenCV")
st.write("Ative sua webcam para visualizar a detecção facial em tempo real.")

run_camera = st.checkbox("Ativar Webcam")
FRAME_WINDOW = st.image([])

if run_camera:
    # Carrega o modelo de detecção facial do OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Inicia a captura da webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("Não foi possível acessar a webcam.")
    else:
        while run_camera:
            ret, frame = camera.read()
            if not ret:
                st.error("Erro ao capturar a imagem da câmera.")
                break

            # Detecta faces no frame
            frame, face_count = detect_faces(frame, face_cascade)

            # Exibe a imagem com as detecções
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            st.write(f"Faces detectadas: {face_count}")

        camera.release()
