import cv2
import mediapipe as mp
import streamlit as st

# Inicializa o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Interface do Streamlit
st.title("Detecção Facial com Mediapipe")
st.write("Ative sua webcam para visualizar a detecção facial em tempo real.")

run_camera = st.checkbox("Ativar Webcam")
FRAME_WINDOW = st.image([])

if run_camera:
    # Captura de vídeo da webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("Não foi possível acessar a webcam.")
    else:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            while run_camera:
                ret, frame = camera.read()
                if not ret:
                    st.error("Erro ao capturar a imagem da câmera.")
                    break

                # Converte BGR para RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Processa o frame com o Face Mesh
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    for landmarks in results.multi_face_landmarks:
                        for point in landmarks.landmark:
                            x = int(point.x * frame.shape[1])
                            y = int(point.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            camera.release()
