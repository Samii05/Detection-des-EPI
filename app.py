import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import os
import pathlib
import sys

# Configuration de la page
st.set_page_config(page_title="Détection EPI", layout="centered")

# Correction pour Windows
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Titre de l'application
st.title("Système de Détection des EPI")
st.markdown("""
**Classes détectées**:  
- GLOVE (Gants) - <span style="color:green">● Vert</span>  
- HELMET (Casque) - <span style="color:red">● Rouge</span>  
- NO-GLOVE (Pas de gants) - <span style="color:blue">● Bleu</span>  
- NO-HELMET (Pas de casque) - <span style="color:yellow">● Jaune</span> 
- NO-SHOES (Pas de chaussures) - <span style="color:cyan">● Cyan</span>
- SHOE (Chaussures) - <span style="color:magenta">● Magenta</span>  
""", unsafe_allow_html=True)

# Sidebar pour les paramètres
st.sidebar.header("Paramètres")
confidence_threshold = st.sidebar.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.01)

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best2.pt', force_reload=True)
        return model
    except Exception as e:
        st.error(f"Erreur de chargement: {str(e)}")
        st.stop()

model = load_model()

# Couleurs BGR pour OpenCV
CLASS_COLORS = {
    'GLOVE': (0, 255, 0),        # Vert
    'HELMET': (0, 0, 255),       # Rouge
    'NO-GLOVE': (255, 0, 0),     # Bleu
    'NO-HELMET': (0, 255, 255),  # Cyan
    'NO-SHOES': (255, 255, 0),   # Jaune
    'SHOE': (255, 0, 255)        # Magenta
}

def draw_detections(image, detections):
    """Dessine les bounding boxes avec couleurs et labels bien visibles"""
    # Ne convertit que si l'image est en RGB (3 canaux)
    if image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image.copy()
    
    output = image_bgr.copy()
    
    for _, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']
        color = CLASS_COLORS.get(label, (255, 255, 255))
        
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        
        # Texte avec fond
        label_text = f"{label} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Fond du texte
        cv2.rectangle(output, 
                     (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), 
                     color, -1)
        
        # Texte
        cv2.putText(output, label_text, 
                   (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 0), 2)
    
    # Reconversion en RGB seulement si nécessaire
    if image.shape[2] == 3:
        return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output

def process_image(image):
    """Effectue la détection et retourne image annotée + résultats"""
    # Conversion en numpy array si c'est une image PIL
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    # S'assurer que l'image est en RGB
    if image_np.shape[2] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    elif image_np.shape[2] == 1:  # Niveau de gris
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    results = model(image_np)
    detections = results.pandas().xyxy[0]
    detections = detections[detections['confidence'] >= confidence_threshold]
    annotated_image = draw_detections(image_np, detections)
    return annotated_image, detections

# Menu principal
option = st.sidebar.radio("Mode de détection", ["Importer une image", "Caméra en direct"])

if option == "Importer une image":
    st.header("Analyse d'image")
    uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_np, caption="Image originale", use_container_width=True)
        
        if st.button("Détecter les EPI", type="primary"):
            with st.spinner("Analyse en cours..."):
                processed_img, detections = process_image(image)
                
                with col2:
                    st.image(processed_img, caption="Résultats de détection", use_container_width=True)
                    st.dataframe(detections)
                
                # Sauvegarde
                os.makedirs("results", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = f"results/detection_{timestamp}.jpg"
                cv2.imwrite(img_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                
                with open(img_path, "rb") as f:
                    st.download_button(
                        "Télécharger l'image analysée",
                        f,
                        file_name=f"detection_{timestamp}.jpg",
                        mime="image/jpeg"
                    )

else:  # Mode Caméra en direct
    st.header("Détection en temps réel")
    run = st.checkbox("Activer la caméra")
    frame_placeholder = st.empty()
    capture_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Impossible d'accéder à la caméra")
        else:
            stop_button = st.button("Arrêter la détection")
            
            while run and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Erreur de lecture de la caméra")
                    break
                
                # Conversion et détection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                
                # Filtrage par seuil de confiance
                detections = results.pandas().xyxy[0]
                detections = detections[detections['confidence'] >= confidence_threshold]
                
                # Dessin des détections
                annotated_frame = draw_detections(frame_rgb, detections)
                
                # Affichage
                frame_placeholder.image(annotated_frame, 
                                     caption="Détection en temps réel", 
                                     use_container_width=True)
            
            cap.release()
            cv2.destroyAllWindows()
            
            if stop_button:
                st.success("Détection arrêtée")
                
                # Option pour sauvegarder la dernière frame
                if 'annotated_frame' in locals():
                    if st.button("Sauvegarder la dernière capture"):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_path = f"results/capture_{timestamp}.jpg"
                        cv2.imwrite(img_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                        st.success(f"Capture sauvegardée: {img_path}")