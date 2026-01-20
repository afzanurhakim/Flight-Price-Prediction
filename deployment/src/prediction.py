import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load Model ---
# Memuat model Random Forest yang sudah disetel (best_rf_model_tuned.pkl)
try:
    with open('../src/best_rf_model_tuned.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("Model berhasil dimuat.")
except FileNotFoundError:
    st.error("File 'best_rf_model_tuned.pkl' tidak ditemukan. Pastikan file model berada di direktori yang sama.")
    model = None
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    model = None

# --- 2. Streamlit Application Interface ---
def main():
    st.title('Prediksi Harga Tiket Pesawat')
    st.write('Masukkan detail penerbangan untuk memprediksi harga tiket.')

    if model is not None:
        
        # --- Input Widget ---
        st.header("Detail Penerbangan")
        
        # Kolom kategorikal
        airline = st.selectbox('Maskapai (Airline)',
                               ['AirAsia', 'Vistara', 'Air India', 'Indigo', 'GO FIRST', 'SpiceJet'])
        
        source_city = st.selectbox('Kota Asal (Source City)',
                                   ['Mumbai', 'Kolkata', 'Delhi', 'Chennai', 'Hyderabad', 'Bangalore'])
        
        destination_city = st.selectbox('Kota Tujuan (Destination City)',
                                        ['Chennai', 'Bangalore', 'Delhi', 'Mumbai', 'Kolkata', 'Hyderabad'])

        # Cek duplikasi kota
        if source_city == destination_city:
            st.warning("Kota Asal dan Kota Tujuan tidak boleh sama. Silakan pilih kota yang berbeda.")
            return # Menghentikan eksekusi jika kota sama

        flight_class = st.selectbox('Kelas (Class)',
                                    ['Economy', 'Business'])
        
        departure_time = st.selectbox('Waktu Keberangkatan (Departure Time)',
                                      ['Morning', 'Evening', 'Night', 'Afternoon', 'Early Morning', 'Late Night'])
        
        arrival_time = st.selectbox('Waktu Kedatangan (Arrival Time)',
                                    ['Morning', 'Evening', 'Night', 'Afternoon', 'Early Morning', 'Late Night'])
        
        stops = st.selectbox('Jumlah Pemberhentian (Stops)',
                             ['zero', 'one', 'two_plus'])
        
        # Kolom numerik
        duration = st.slider('Durasi Penerbangan (Duration) dalam jam', 1, 48, 4)
        
        days_left = st.slider('Sisa Hari Menuju Keberangkatan (Days Left)', 1, 50, 3)

        # --- 3. Inferencing ---
        if st.button('Prediksi Harga'):
            # Buat DataFrame dari input pengguna
            input_data = pd.DataFrame([{
                'airline': airline,
                'source_city': source_city,
                'departure_time': departure_time,
                'stops': stops,
                'arrival_time': arrival_time,
                'destination_city': destination_city,
                'class': flight_class,
                'duration': str(duration), # Convert ke string sesuai notebook
                'days_left': str(days_left) # Convert ke string sesuai notebook
            }])

            st.subheader("Data Input untuk Prediksi:")
            st.dataframe(input_data)

            # Lakukan prediksi
            try:
                y_pred_inf = model.predict(input_data)
                predicted_price = y_pred_inf[0]
                
                # Tampilkan hasil prediksi
                st.markdown('---')
                st.subheader('Hasil Prediksi Harga Tiket')
                st.success(f"Harga Tiket Diprediksi Sebesar: **{predicted_price:,.2f}**")
                st.caption("Nilai yang ditampilkan adalah prediksi harga tiket berdasarkan model machine learning.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi. Pastikan data input sudah benar: {e}")

# Jalankan fungsi utama
if __name__ == '__main__':
    main()