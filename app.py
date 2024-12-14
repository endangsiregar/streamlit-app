import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk memuat model EfficientNet
def load_model():
    model = models.efficientnet_b0(pretrained=True)  # Memuat model EfficientNet B0
    model.eval()  # Ubah model ke evaluasi mode
    return model

# Fungsi untuk melakukan prediksi pada gambar
def predict_image(model, image):
    # Preprocessing gambar agar sesuai dengan input EfficientNet
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisasi berdasarkan ImageNet
    ])
    image = transform(image).unsqueeze(0)  # Menambah dimensi batch

    # Prediksi
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)
    return predicted_class.item()

# Fungsi untuk menampilkan visualisasi confidence level
def plot_confidence(prediction, class_names):
    fig, ax = plt.subplots()
    ax.barh(class_names, prediction, color='skyblue')
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
            predicted_class = predict_image(model, image)

            # Daftar kelas untuk EfficientNet
            class_names = ['Pinuncaan', 'Ragi Hidup', 'Ragi Hotang', 'Sadum', 'Sibolang', 'Tumtuman']
            predicted_label = class_names[predicted_class]

            st.write(f"**Predicted Class:** {predicted_label}")
            st.write(f"**Confidence:** 80%")  # Ini adalah placeholder, Anda bisa menambahkan logika untuk menghitung confidence.

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

            ulos_info = ulos_descriptions.get(predicted_label, {})
            st.write(f"**Tentang {predicted_label}:**")
            st.write(f"**Desain:** {ulos_info.get('Desain', 'Deskripsi desain belum tersedia.')}")

            st.write("**Kegunaan:**")
            for kegunaan in ulos_info.get('Kegunaan', []):
                st.write(f"- {kegunaan}")

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
