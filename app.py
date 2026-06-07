import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from keras.models import load_model
import numpy as np
from PIL import Image

try:
    model = load_model("brain_model (3).keras")
      
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

labels = [
    'glioma',
    'meningioma',
    'notumor',
    'pituitary'
]

def crop_image(image):
    width, height = image.size
    new_width = min(width, height)
    new_height = min(width, height)
    
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    
    return image.crop((left, top, right, bottom))

def classify_image(image):
    image = image.convert("RGB")

    cropped_image = crop_image(image)
    img = cropped_image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = labels[predicted_index]
    confidence = float(predictions[0][predicted_index])

    probs = {
        labels[i]: float(predictions[0][i])
        for i in range(len(labels))
    }
    
    return predicted_class, confidence, probs

    # predictions = model.predict(img_array)
    # probability = float(predictions[0][0])

    # if probability > 0.5:
    #     predicted_class = 'kanker'
    #     confidence = probability  
    # else:
    #     predicted_class = 'normal'
    #     confidence = 1.0 - probability
        
    # # max_index = np.argmax(predictions[0])
    # return predicted_class, confidence


st.title("Klasifikasi Histopatologi Kanker Usus Besar")
uploaded_file = st.file_uploader("Unggah gambar untuk klasifikasi", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:    
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    if st.button("Klasifikasikan"):
        with st.spinner("Memproses..."):
            label, confidence, probs = classify_image(image)

            st.write("---")
            if label == 'notumor':
                st.success(f"Hasil Prediksi: **{label.upper()}**")
            else:
                st.warning(f"Hasil Prediksi: **{label.upper()}**")

            st.info(f"Keyakinan Model: {confidence * 100:.2f}")
            for cls, prob in probs.items():
                st.write(f"{cls}: {prob*100:.2f}%")
