import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Fungsi untuk memuat model CNN
def load_model():
    model = tf.keras.models.load_model('Deployment/model_ulos.h5')
    return model

# Fungsi untuk melakukan prediksi pada gambar
def predict_image(model, image):
    # Preprocessing gambar agar sesuai dengan input model
    image = image.resize((150, 150))  # Sesuaikan ukuran input dengan model
    image_array = np.array(image) / 255.0  # Normalisasi
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan batch dimension

    # Prediksi
    prediction = model.predict(image_array)
    return prediction

# Fungsi validasi awal untuk memastikan gambar adalah motif ulos
def validate_image(prediction, threshold=0.8):
    max_confidence = np.max(prediction)
    if max_confidence >= threshold:  # Jika confidence lebih dari atau sama dengan threshold, valid
        return True, max_confidence
    return False, max_confidence

# Fungsi untuk menampilkan visualisasi confidence level
def plot_confidence(prediction, class_names):
    fig, ax = plt.subplots()
    ax.barh(class_names, prediction[0], color='skyblue')
    ax.set_xlabel('Confidence')
    ax.set_title('Prediction Confidence Level')
    st.pyplot(fig)

# Streamlit App
st.set_page_config(layout="wide")
st.title("Ulos Image Classification")

# Sidebar untuk navigasi
page = st.sidebar.radio("Pilih Fitur", ["Klasifikasi Gambar", "Panduan Pengguna"])

if page == "Klasifikasi Gambar":
    st.markdown(
        """
        **Tentang Aplikasi:**
        Aplikasi ini menggunakan model **Convolutional Neural Network (CNN)** untuk mengklasifikasikan jenis kain **Ulos** berdasarkan gambar yang diunggah pengguna. Model ini bertujuan untuk mendukung pelestarian budaya dan meningkatkan pemahaman tentang kain tradisional ulos.
        """
    )

    # Upload file gambar
    gu_image = st.file_uploader("Upload an image of Ulos", type=["jpg", "jpeg", "png"])

    if gu_image is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(gu_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Muat model dan lakukan prediksi
        st.write("Processing image...")
        try:
            model = load_model()

            # Prediksi gambar
            prediction = predict_image(model, image)

            # Validasi gambar sebagai motif ulos
            is_valid, confidence = validate_image(prediction, threshold=0.8)  # Ambang batas 80%
            if not is_valid:
                st.error(
                    f"Gambar tidak dikenali sebagai motif ulos. Confidence: {confidence:.2f}. Silakan unggah gambar ulos yang sesuai."
                )
                st.stop()  # Menghentikan eksekusi jika gambar tidak valid

            # Validasi hasil prediksi
            class_names = ['Pinuncaan', 'Ragi Hidup', 'Ragi Hotang', 'Sadum', 'Sibolang', 'Tumtuman']  # Label kelas
            if len(prediction[0]) == len(class_names):
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                st.write(f"**Predicted Class:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}%")

                # Tampilkan visualisasi confidence level
                plot_confidence(prediction, class_names)

                # Deskripsi tambahan tentang ulos yang diprediksi
                ulos_descriptions = {
                    "Pinuncaan": {
                        "Desain": "Ulos ini memiliki struktur yang terdiri dari lima bagian yang ditenun secara terpisah dan kemudian disatukan. Motifnya biasanya menggunakan warna-warna cerah dengan pola geometris yang khas.",
                        "Kegunaan": [
                            "Acara Resmi: Sering digunakan dalam upacara adat dan acara resmi oleh pemimpin atau raja.",
                            "Pernikahan: Dipakai oleh pengantin dan keluarga dalam perayaan pernikahan.",
                            "Marpaniaran: Digunakan saat pesta besar dalam acara marpaniaran.",
                            "Simbol Kehormatan: Melambangkan status dan kehormatan bagi pemakainya."
                        ]
                    },
                    # Tambahkan deskripsi ulos lainnya di sini
                }

                ulos_info = ulos_descriptions.get(predicted_class, {})
                st.write(f"**Tentang {predicted_class}:**")
                st.write(f"**Desain:** {ulos_info.get('Desain', 'Deskripsi desain belum tersedia.')}")
                st.write("**Kegunaan:**")
                for kegunaan in ulos_info.get('Kegunaan', []):
                    st.write(f"- {kegunaan}")

            else:
                st.error("Model output dimensions do not match the number of class names. Please check the model and class labels.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

elif page == "Panduan Pengguna":
    st.header("Panduan Pengguna")
    st.markdown(
        """ 
        ### Cara Menggunakan Aplikasi
        1. **Unggah Gambar:** Klik tombol "Browse Files" untuk mengunggah gambar ulos dalam format JPG, JPEG, atau PNG.<br>
        2. **Validasi Gambar:** Pastikan gambar yang diunggah memiliki kualitas baik dan menampilkan kain ulos dengan jelas.<br>
        3. **Hasil Prediksi:** Tunggu beberapa saat untuk melihat hasil klasifikasi dan tingkat kepercayaan model.<br>
        4. **Informasi Tambahan:** Bacalah deskripsi singkat tentang jenis ulos yang terdeteksi untuk memperkaya pengetahuan.
        """,
        unsafe_allow_html=True
    )

st.write("\n\n")
st.write("Developed by Kelompok 3 - Data Mining")
