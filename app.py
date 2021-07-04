from typing import List

import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.colors as mcolors
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from datetime import datetime
from streamlit_webrtc import ClientSettings
import socket
now = datetime.now()

CLASSES = [ 'conforme', 'non_conforme' ]


WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )


#changer le nom de la page affichée dans l'onglet du navigateur
#set_page_config doit être appelé avant toutes les fonctions streamlit
 
st.set_page_config(
    page_title="Bouteilles de GAZ",
)

st.title('Détection des Bouteilles de GAZ')

#region Functions


@st.cache(max_entries=2)
def get_yolo5(model_type='s'):
    
    return torch.hub.load('ultralytics/yolov5', 'custom', path='./gaz_best.pt')

@st.cache(max_entries=10)
def get_preds(img : np.ndarray) -> np.ndarray:
    
    return model([img]).xyxy[0].numpy()

def get_colors(indexes : List[int]) -> dict:
   
    to_255 = lambda c: int(c*255)
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) 
                                                for name_color in tab_colors]
    base_colors = list(mcolors.BASE_COLORS.values())
    base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
    rgb_colors = tab_colors + base_colors
    rgb_colors = rgb_colors*5

    color_dict = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors):
            color_dict[index] = rgb_colors[i]
        else:
            color_dict[index] = (255,0,0)

    return color_dict

def get_legend_color(class_name : int):
   

    index = CLASSES.index(class_name)
    color = rgb_colors[index]
    return 'background-color: rgb({color[0]},{color[1]},{color[2]})'.format(color=color)

# DB Management
import sqlite3
conn = sqlite3.connect('gazapp.db')
c = conn.cursor()
# DB  Functions
def create_bottle():
    c.execute('CREATE TABLE IF NOT EXISTS bottle(type VARCHAR,time VARCHAR)')

def add_bottle(type,time):
    c.execute('INSERT INTO bottle(type, time) VALUES (?,?)',(type,time))
    conn.commit()

### History of use
def create_history():
    c.execute('CREATE TABLE IF NOT EXISTS history(host VARCHAR,timeh VARCHAR)')

create_history()

def add_device(host,timeh):
    c.execute('INSERT INTO history(host, timeh) VALUES (?,?)',(host,timeh))
    conn.commit()

hostname = socket.gethostname()
host = socket.gethostbyname(hostname)
timeh= now.strftime("%d/%m/%Y %H:%M:%S")
add_device(host,timeh)


### End of History check


create_bottle()  ##create the table bottle

#################################""

class VideoTransformer(VideoTransformerBase):
   
    def __init__(self):
        self.model = model
        self.rgb_colors = rgb_colors
        self.target_class_ids = target_class_ids

    def get_preds(self, img : np.ndarray) -> np.ndarray:
        return self.model([img]).xyxy[0].numpy()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.get_preds(img)
        result = result[np.isin(result[:,-1], self.target_class_ids)]
        for bbox_data in result:
            xmin, ymin, xmax, ymax, _, label = bbox_data
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)

            img = cv2.rectangle(img, p0, p1, self.rgb_colors[label], 2) 

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#endregion


#region Load model
# ---------------------------------------------------

model_type = 's'

with st.spinner('Loading the model...'):
    model = get_yolo5(model_type)
st.success('Loading the model.. Done!')
#endregion


# UI elements
# ----------------------------------------------------

#sidebar
prediction_mode = st.sidebar.radio(
    "",
    ('Single image', 'Web camera'),
    index=0)
    
classes_selector = st.sidebar.multiselect('Select classes', 
                                        CLASSES, default='non_conforme')
all_labels_chbox = st.sidebar.checkbox('Toutes', value=False)


# Prediction section
# ---------------------------------------------------------

#target_class_ids - indices des classes sélectionnées selon la liste des classes entrainés
#rgb_colors - couleurs rgb pour les classes sélectionnées
if all_labels_chbox:
    target_class_ids = list(range(len(CLASSES)))
elif classes_selector:
    target_class_ids = [CLASSES.index(class_name) for class_name in classes_selector]
else:
    target_class_ids = [0]

rgb_colors = get_colors(target_class_ids)
detected_ids = None


if prediction_mode == 'Single image':

    # ajoute un formulaire pour télécharger une image
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'])

    # si le fichier est téléchargé
    if uploaded_file is not None:

        # conversion d'image d'octets en np.ndarray
        bytes_data = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_preds(img)

        #copier les résultats de la fonction cacheable pour que le cache ne soit pas modifié
        result_copy = result.copy()
        #sélectionner uniquement les objets des classes requises
        result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]
        
        detected_ids = []
        #Copier également l'image pour ne pas modifier l'argument du cache 
        # de la fonction get_preds
        img_draw = img.copy().astype(np.uint8)
        # dessiner des boxes pour tous les objets cibles trouvés
        for bbox_data in result_copy:
            xmin, ymin, xmax, ymax, _, label = bbox_data
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
            img_draw = cv2.rectangle(img_draw, 
                                    p0, p1, 
                                    rgb_colors[label], 2) 
            detected_ids.append(label)
            if int(label)==1:
                type ='Non_Conforme'
            elif int(label)==0:
                type='Conforme'
            time= now.strftime("%d/%m/%Y %H:%M:%S")
            add_bottle(type,time)
        
        # affichons l'image avec les cases dessinées
        # use_column_width va étirer l'image à la largeur de la colonne centrale
        st.image(img_draw, use_column_width=True)

elif prediction_mode == 'Web camera':
    
    # créer un objet pour sortir le flux de la caméra
    ctx = webrtc_streamer(
        key="example", 
        video_transformer_factory=VideoTransformer,
        client_settings=WEBRTC_CLIENT_SETTINGS,)

    # nécessaire pour que l'objet VideoTransformer récupère les nouvelles données.
    # après avoir rafraîchi la page streamlit
    if ctx.video_transformer:
        ctx.video_transformer.model = model
        ctx.video_transformer.rgb_colors = rgb_colors
        ctx.video_transformer.target_class_ids = target_class_ids

# affiche la liste des classes trouvées en travaillant avec l'image ou la liste de toutes les classes.
detected_ids = set(detected_ids if detected_ids is not None else target_class_ids)
labels = [CLASSES[index] for index in detected_ids]
legend_df = pd.DataFrame({'label': labels})
st.dataframe(legend_df.style.applymap(get_legend_color))
