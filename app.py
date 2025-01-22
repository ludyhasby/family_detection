import streamlit as st 
from keras.utils import img_to_array
import numpy as np 
import json 
from PIL import Image
from tensorflow.keras.models import load_model
from gtts import gTTS

with open("label_indeks.json", "r") as json_file:
    label_indeks = json.load(json_file)
model = load_model("model.h5")

def predict(file_image, model, class_labels):
        if file_image.mode != "RGB":
            file_image = file_image.convert("RGB")
        test_image = file_image.resize((300, 300))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalisasi
        result = model.predict(test_image)
        predicted_class = np.argmax(result)
        return class_labels[str(predicted_class)], result

def result_speech(name):
    if name=="Ali Topan":
        teks = """Menurut saya, dari wajah anda terlihat perlu lebih Semangat dan juga lebih sabar ! \nKetulusan Anda dalam mengajari anak-anak anda. """
    elif name== "Dini Hanif": 
        teks = "Dari wajah anda terlihat anda perlu lebih rajin dan tidak koproh dalam melakukan apa-apa. Anda juga harus menjaga dan memprioritaskan diri anda sendiri, jangan kecapekan 😘"
    elif name=="Ludy Hasby":
        teks = """Anda boleh pinjam HP sepuasnya, karena anda terlihat Baik sekali 😊😊😊😊"""
    elif name=="Zahra Sabila":
        teks = "Hmm anda terlihat galak 🤔💭 Tapi sungguh anda cantik !"
    elif name=="Nanda Sobrina":
        teks = "Anda terlihat kayak pingin tidur ya.. kurangin tidur sorenya ya ! Tapi sungguh anda cantik !"
    elif name=="Fathan Tornado":
        teks = "Wajah anda terlihat kecanduan main game 🤔"
    elif name=="Fawwas Mubarrak":
        teks = "Sepertinya wajahnya baik sekali. Lalu saya baru ingat, apakah anda suaminya Dewi kelas 3 ? Anda ganteng 😅"
    else:
        teks = "Maaf saya belum kenal anda, yuk kenalan.."
    tts = gTTS(teks, lang='id')
    tts.save("speech.mp3")
    audio_file = open("speech.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")
    st.markdown(teks)

# page config
st.set_page_config(
    page_icon="👨‍👩‍👧‍👦",
    page_title="Alto Classification"
)
if __name__ == "__main__":
    st.title("Alto Lens 🔏")
    st.subheader("Bisa mendeteksi siapa Anda (dengan batasan !)")
    option = st.selectbox("Input File atau Camera ?", ("Camera", "File"))
    if option == "File":
        file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])
    else:
        file  = st.camera_input("Take a picture")

    if file:
        image = Image.open(file)
        top_pred, detail = predict(image, model, label_indeks)
        st.subheader(f":red[{top_pred}]")
        result_speech(top_pred)
        st.write(detail)
