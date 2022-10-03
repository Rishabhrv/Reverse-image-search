import os
import cv2
import pickle
import tensorflow
import numpy as np
from PIL import Image
import bz2file as bz2
import streamlit as st
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

file = bz2.BZ2File('distancez2.pbz2', 'rb')
distance = pickle.load(file)

file = bz2.BZ2File('filenamesz2.pbz2', 'rb')
filenames = pickle.load(file)

file = bz2.BZ2File('linkz2.pbz2', 'rb')
link = pickle.load(file)

file = bz2.BZ2File('namesz2.pbz2', 'rb')
names = pickle.load(file)

file = bz2.BZ2File('stylez2.pbz2', 'rb')
data = pickle.load(file)

# filenames = pickle.load(open('filenames.pkl', 'rb'))
# distance = pickle.load(open('distance.pkl', 'rb'))
# data = pickle.load(open('style.pkl', 'rb'))
# names = pickle.load(open('names.pkl', 'rb'))
# link = pickle.load(open('link.pkl', 'rb'))
image = Image.open('myimage.jpg')

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

st.title('Image Search')

st.sidebar.write('Built By -')
st.sidebar.title('Rishabh Vyas')
st.sidebar.image(image, caption='Machine Learning Engineer', width=160)
st.sidebar.write('E-mail - rishabhvyas472@gmail.com')


def save_uploaded_file(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(path, model):
    # img = image.load_img(path, target_size=(224, 224))
    # img_array = image.img_to_array(img)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img_array = cv2.resize(img, (224, 224))
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    distances, indices = distance.kneighbors([normalized_result])

    return indices


def trimmer(string):
    temp = string.strip('images\\')
    temp = temp.strip('.jpg')
    temp = int(temp)
    caption = data[data['id'] == temp]['productDisplayName'].values[0]

    return caption

def image_link(string):
    temp = link[link['filename'] == string]['link'].values[0]

    return temp

uploaded_image = st.file_uploader('Choose an Image')

if st.button('Search'):

    if uploaded_image is not None:
        if save_uploaded_file(uploaded_image):
            display_image = Image.open(uploaded_image)
            st.subheader('Uploded image')
            st.image(display_image, width=250, caption='Generic Product')

            features = feature_extraction(os.path.join('uploads', uploaded_image.name), model)

            st.subheader('Search Results')

            col1, col2, col3, col4 = st.columns(4, gap="small")

            with col1:
                st.image(image_link(names[features[0][0]]), caption=trimmer(filenames[features[0][0]]))
            with col2:
                st.image(image_link(names[features[0][1]]), caption=trimmer(filenames[features[0][1]]))
            with col3:
                st.image(image_link(names[features[0][2]]), caption=trimmer(filenames[features[0][2]]))
            with col4:
                st.image(image_link(names[features[0][3]]), caption=trimmer(filenames[features[0][3]]))

            col5, col6, col7, col8 = st.columns(4, gap="small")

            with col5:
                st.image(image_link(names[features[0][4]]), caption=trimmer(filenames[features[0][4]]))
            with col6:
                st.image(image_link(names[features[0][5]]), caption=trimmer(filenames[features[0][5]]))
            with col7:
                st.image(image_link(names[features[0][6]]), caption=trimmer(filenames[features[0][6]]))
            with col8:
                st.balloons()
                st.image(image_link(names[features[0][7]]), caption=trimmer(filenames[features[0][7]]))


        else:
            st.header('Some erorr')
