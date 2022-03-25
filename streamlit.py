import cv2
import streamlit as st
from utils import (
    build_sidebar,
    process_video,
    process_image,
)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

st.set_page_config(
    page_title="Filtros",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Filtro para vídeo")
st.markdown(
    """
    ### Este projeto realiza a aplicação de filtros em tempo real (video), aplicando o conceito de realidade aumentada.
    """
)
mode_selectbox = st.selectbox(
    "Selecione um modo",
    ("Vídeo", )
)
build_sidebar()


col_left, col_right = st.columns(2)
with col_left:
    if mode_selectbox == "Vídeo":
        process_video(col_right)
