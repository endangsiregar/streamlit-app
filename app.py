import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

# Fungsi untuk memuat model EfficientNet
def load_model():
    model = tf.keras.models.load_model('Deployment/model_ulos_efficientnet.h5')
    return model

# Fungsi untuk melakukan prediksi pada gambar
def predict_image(model, image):
    # Preprocessing gambar agar sesuai dengan input EfficientNet
    image = image.resize((224, 224))  # Ukuran input untuk EfficientNet
    image_array = np.array(image)
    image_array = preprocess_input(image_array)  # Preprocessing khusus EfficientNet
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
st.title("Ulos Image Classification with EfficientNet")

# Sidebar untuk navigasi
page = st.sidebar.radio("Pilih Fitur", ["Klasifikasi Gambar", "Panduan Pengguna"])

if page == "Klasifikasi Gambar":
    st.markdown(
        """
        **Tentang Aplikasi:**
        Aplikasi ini bertujuan untuk mengenali dan mengklasifikasikan motif kain ulos menggunakan model kecerdasan buatan berbasis EfficientNet. Proyek ini mengintegrasikan teknologi modern untuk melestarikan warisan budaya Batak secara digital.
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
                        "Desain": "Ulos ini terdiri dari lima bagian yang ditenun secara terpisah, lalu digabungkan menjadi satu kain utuh. Polanya biasanya menggunakan warna cerah dengan motif geometris khas.",
                        "Kegunaan": [
                            "Upacara Adat: Dipakai dalam berbagai upacara tradisional, terutama oleh pemimpin adat atau raja.",
                            "Pernikahan: Digunakan oleh pasangan pengantin dan keluarga saat acara pernikahan tradisional.",
                            "Pesta Marpaniaran: Dikenakan dalam acara besar seperti perayaan adat dan pesta keluarga.",
                            "Simbol Status: Melambangkan kehormatan dan status sosial pemakainya."
                        ]
                    },
                    # Anda dapat menambahkan deskripsi motif ulos lainnya dengan format yang sama
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
st.write("Developed by Kelompok 12 - Data Mining")
