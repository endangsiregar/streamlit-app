import streamlit as st #
import pandas as pd
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Classification of ulos image using EficientNet")

# Deskripsi singkat
st.write("Aplikasi ini bertujuan untuk mengenali dan mengklasifikasikan motif kain ulos menggunakan model kecerdasan buatan berbasis EfficientNet. Proyek ini mengintegrasikan teknologi modern untuk melestarikan warisan budaya Batak secara digital.")

# Input data dari pengguna
uploaded_file = st.file_uploader("Upload gambar Anda (JPG atau PNG):", type=["jpg", "png"])

if uploaded_file is not None:
    # Menampilkan nama file gambar yang diunggah
    st.write(f"Nama file gambar yang diunggah: {uploaded_file.name}")
    
    # Menampilkan pratinjau gambar
    from PIL import Image
    image = Image.open(uploaded_file)
    st.image(image, caption=f"Pratinjau {uploaded_file.name}", use_column_width=True)
    
    # Anda dapat menambahkan proses klasifikasi gambar di sini
    st.write("Proses klasifikasi akan dilakukan di sini.")

    # Tampilkan statistik deskriptif
    st.write("Statistik Deskriptif:")
    st.write(data.describe())

    # Buat grafik
    st.write("Visualisasi Data:")
    column = st.selectbox("Pilih kolom untuk divisualisasikan:", data.columns)
    plt.hist(data[column], bins=10, color='skyblue', edgecolor='black')
    st.pyplot(plt)
