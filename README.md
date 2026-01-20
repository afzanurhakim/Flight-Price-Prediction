# Flight Price Prediction

## Repository Outline
1. description.md - Penjelasan gambaran umum project
2. P1M2_muhammad_afza.ipynb - Notebook berisi EDA dan Model dari project
3. P1M2_muhammad_afza_inf.ipynb - Notebook berisi inferencing untuk model yang telah dibangun
4. P1M2_muhammad_afza_conceptual.txt - Notepad berisi jawaban untuk pertanyaan conceptual
5. Dataset.csv - File csv dataset yang digunakan dalam project ini
6. best_rf_model_tuned.pkl - File pickle yang merupakan model yang digunakan dalam project ini
7. eda.py - Program streamlit untuk EDA
8. predict.py - Program streamlit untuk prediction
9. requirements.txt - Notepad berisi version program yang digunakan untuk HuggingFace
10. streamlit_app.py - Program streamlit untuk keseluruhan project

## Problem Background
Penentuan harga tiket pesawat yang optimal dan kompetitif adalah tantangan krusial dalam industri penerbangan dan pariwisata. Harga tiket bersifat sangat dinamis dan dipengaruhi oleh banyak faktor seperti maskapai, rute, kelas layanan, waktu booking, musim, hingga tingkat permintaan. Harga yang tidak akurat dapat menyebabkan kerugian finansial bagi maskapai (jika harga terlalu rendah) atau membuat konsumen beralih ke maskapai lain (jika harga terlalu tinggi)

Bagi Online Travel Agent (OTA) seperti Traveloka, Tiket.com, atau Skyscanner, memprediksi harga tiket secara akurat sangat penting untuk memberikan rekomendasi terbaik, memaksimalkan konversi, dan menjaga kepuasan pelanggan.

## Project Output
Membangun model Machine Learning Regression yang mampu memprediksi Harga tiket pesawat (Price) secara akurat berdasarkan atribut penerbangan. 
Serta space HuggingFace untuk EDA dan prediksi

## Data
Dataset ini berisi dataset penerbangan di India dari bulan Februari - Maret 2022. Data didapat dari website Ease My Trip. Data memiliki informasi tentang maskapai penerbangan, kode penerbangan, kota asal, waktu keberangkatan, total perhentian, waktu kedatangan, kota tujuan, kelas penerbangan, durasi penerbangan, jarak antara booking dengan penerbangan, dan price sebagai targetnya. Terdapat total 2.890 baris

## Method
Project ini menggunakan model supervised learning dengan model K Neighbors Regressor, SVR, Decission Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor

## Stacks
Python, Pandas, numPy, Matplotlib, Seaborn, scikit-learn, sciPy, plotly-express, streamlit

## Reference
Huggingface URL: https://huggingface.co/spaces/afzanurhakim/Flight-Price-Prediction

