import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import io

# Mengatur tampilan halaman Streamlit
st.set_page_config(layout="wide")

def load_data(file_path):
    """Memuat data dari file CSV."""
    try:
        # Menggunakan encoding 'latin1' atau 'cp1252' jika 'utf-8' gagal,
        # umum pada dataset yang berasal dari sistem Windows.
        try:
            df_eda = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df_eda = pd.read_csv(file_path, encoding='latin1')
            
        # Asumsi: pastikan kolom yang akan dianalisis ada
        required_cols = ['price', 'airline', 'departure_time', 'duration', 'days_left', 'stops', 'class']
        if not all(col in df_eda.columns for col in required_cols):
            st.error(f"Data tidak lengkap. Pastikan memiliki kolom: {', '.join(required_cols)}")
            return None

        # --- Basic Preprocessing ---
        # Mengubah kolom 'days_left' ke numerik (jika belum)
        df_eda['days_left'] = pd.to_numeric(df_eda['days_left'], errors='coerce')
        
        # Mengubah kolom 'duration' ke numerik (jika belum)
        # Catatan: Jika 'duration' dalam format 'xh ym', Anda perlu logika konversi yang lebih kompleks di sini.
        df_eda['duration'] = pd.to_numeric(df_eda['duration'], errors='coerce')
        
        # Hapus baris dengan NaN yang dihasilkan dari konversi/pembersihan
        df_eda.dropna(subset=required_cols, inplace=True) 

        return df_eda
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan di: {file_path}. Pastikan 'Dataset.csv' ada di direktori yang benar.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat atau memproses data: {e}")
        return None

def main():
    st.title('Exploratory Data Analysis (EDA) of Flight Price Prediction')
    st.write('Analisis ini didasarkan pada file `Dataset.csv` yang digunakan dalam project.')
    
    # Lokasi file data - Sesuaikan jika perlu
    file_path = '../src/Dataset.csv' 
    df_eda = load_data(file_path)

    if df_eda is not None:
        
        # --- 1. Ringkasan Data Awal (Layout Vertikal) ---
        st.header('1. Ringkasan Data Awal')

        # 1.1 Sekilas Data
        st.subheader('Sekilas Data')
        st.dataframe(df_eda.head())

        st.markdown('***') 

        # 1.2 Informasi Dasar Data
        st.subheader('Informasi Dasar Data')
        # Mengambil df.info() ke dalam string
        buffer = io.StringIO()
        df_eda.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.markdown('***') 

        # 1.3 Statistik Deskriptif
        st.subheader('Statistik Deskriptif (Seluruh Kolom)')
        st.dataframe(df_eda.describe(include='all'))

        st.markdown('---')
        
        # --- 2. Analisis Variabel Target (Price) ---
        st.header('2. Analisis Variabel Target: Harga (Price)')
        
        # Set ukuran figure untuk menampilkan dua plot secara berdampingan
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Histogram (Distribusi Frekuensi)
        sns.histplot(df_eda['price'], kde=True, bins=50, ax=ax1)
        ax1.set_title('Distribusi Harga Tiket', fontsize=14)
        ax1.set_xlabel('Harga (Price)', fontsize=12)
        ax1.set_ylabel('Frekuensi (Frequency)', fontsize=12)
        ax1.grid(axis='y', alpha=0.5)

        # Plot 2: Box Plot (Deteksi Outlier dan Quartil)
        sns.boxplot(y=df_eda['price'], ax=ax2)
        ax2.set_title('Box Plot Harga Tiket', fontsize=14)
        ax2.set_ylabel('Harga (Price)', fontsize=12)

        plt.tight_layout()
        st.pyplot(fig)
        
        st.write('**Interpretasi:**')
        st.write('* **Histogram:** Distribusi harga cenderung **miring ke kanan (*right-skewed*)**, menunjukkan mayoritas harga berada di rentang bawah.')
        st.write('* **Box Plot:** Menunjukkan adanya banyak **nilai *outlier* pada harga yang lebih tinggi**, yang mungkin merupakan harga kelas bisnis atau penerbangan premium.')

        st.subheader('Statistik Deskriptif untuk Harga (Price)')
        st.dataframe(df_eda['price'].describe().to_frame())
        
        st.markdown('---')
        
        # --- 3. Harga Berdasarkan Maskapai (Airline) + ANOVA ---
        st.header('3. Analisis Harga Berdasarkan Maskapai Penerbangan (Airline)')
        
        target_col = 'price'
        group_col = 'airline'
        
        # Statistik Deskriptif Harga per Maskapai
        airline_stats = df_eda.groupby(group_col)[target_col].agg(
            ['count', 'mean', 'median', 'min', 'max']
        ).sort_values(by='mean', ascending=False)
        
        st.subheader('Statistik Harga Rata-rata dan Sebaran per Maskapai')
        st.dataframe(airline_stats)
        
        # Uji ANOVA
        # Pastikan tidak ada grup kosong (jika ada data yang sangat kotor)
        valid_groups = [group[target_col].values for name, group in df_eda.groupby(group_col) if not group.empty]
        if len(valid_groups) > 1:
            f_statistic, p_value = f_oneway(*valid_groups)
        else:
            f_statistic, p_value = 0, 1.0 # Nilai default jika ANOVA tidak mungkin

        st.subheader('Uji One-Way ANOVA untuk Harga Maskapai')
        st.code(f"P-Value: {p_value:.10f}")
        
        if p_value < 0.05:
            st.success('Kesimpulan: **P-Value sangat kecil (< 0.05)**, yang menunjukkan ada **perbedaan harga yang signifikan secara statistik antar maskapai**.')
        else:
            st.info('Kesimpulan: P-Value besar (> 0.05) atau ANOVA tidak dapat dilakukan. Tidak ada atau tidak terbukti adanya perbedaan harga yang signifikan antar maskapai.')


        # Group data berdasarkan 'airline' dan 'class', lalu hitung jumlahnya
        class_distribution = df_eda.groupby(['airline', 'class']).size().unstack(fill_value=0)

        # Tambahkan kolom Total untuk mengetahui total penerbangan per maskapai
        class_distribution['Total'] = class_distribution.sum(axis=1)

        # Urutkan berdasarkan total penerbangan
        class_distribution = class_distribution.sort_values(by='Total', ascending=False)

        st.dataframe(class_distribution)
        
        st.write('**Interpretasi:** Tabel di atas menunjukkan jumlah penerbangan Kelas Ekonomi dan Bisnis yang ditawarkan oleh setiap maskapai, diurutkan berdasarkan total frekuensi penerbangan.')

        # Visualisasi (Opsional: Bar Plot)
        st.subheader("Visualisasi Distribusi Kelas")
        
        # Normalisasi untuk mendapatkan persentase dalam bar plot tumpuk
        class_distribution_norm = class_distribution[['Economy', 'Business']].apply(lambda x: x / x.sum(), axis=1).sort_values(by='Business', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        class_distribution_norm[['Economy', 'Business']].plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        
        ax.set_title('Persentase Kelas Layanan per Maskapai', fontsize=16)
        ax.set_xlabel('Maskapai Penerbangan', fontsize=12)
        ax.set_ylabel('Proporsi', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Class')
        plt.tight_layout()
        st.pyplot(fig)

        # Visualisasi Box Plot Harga per Maskapai
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=group_col, y=target_col, data=df_eda, order=airline_stats.index, ax=ax)
        ax.set_title('Box Plot Harga Tiket Berdasarkan Maskapai', fontsize=16)
        ax.set_xlabel('Maskapai Penerbangan', fontsize=12)
        ax.set_ylabel('Harga Tiket (Price)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('---')

        # --- 4. Harga Berdasarkan Waktu Keberangkatan (Departure_Time) ---
        st.header('4. Analisis Harga Berdasarkan Waktu Keberangkatan (Departure Time)')
        
        # Statistik Deskriptif Harga per Waktu Keberangkatan
        time_stats = df_eda.groupby('departure_time')['price'].agg(
            ['count']
        ).sort_values(by='count', ascending=False)

        st.subheader('Statistik Harga Berdasarkan Waktu Keberangkatan')
        st.dataframe(time_stats)

        # Visualisasi Box Plot Harga per Waktu Keberangkatan
        time_order = time_stats.index 
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            x='departure_time',
            y='price',
            data=df_eda,
            order=time_order,
            ax=ax
        )

        ax.set_title('Perbandingan Distribusi Harga Tiket Berdasarkan Waktu Keberangkatan', fontsize=14)
        ax.set_xlabel('Waktu Keberangkatan', fontsize=12)
        ax.set_ylabel('Harga Tiket (Price)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('---')

        # --- 5. Hubungan Durasi (Duration) vs. Harga (Price) ---
        st.header('5. Hubungan Durasi Penerbangan (Duration) dan Harga Tiket')
        
        duration_col = 'duration'
        price_col = 'price'

        # Hitung Koefisien Korelasi
        correlation = df_eda[duration_col].corr(df_eda[price_col])

        st.subheader('Koefisien Korelasi (Durasi vs Harga)')
        st.code(f"Koefisien Korelasi Pearson: {correlation:.4f}")
        
        if abs(correlation) > 0.3:
            st.info(f'Interpretasi: Terdapat hubungan linear yang sedang ({abs(correlation):.2f}) antara Durasi dan Harga.')
        else:
            st.info(f'Interpretasi: **Hubungan linear antara Durasi dan Harga cenderung lemah ({abs(correlation):.2f})**.')

        # Visualisasi Scatter Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(
            x=duration_col, 
            y=price_col, 
            data=df_eda, 
            scatter_kws={'alpha': 0.3}, 
            line_kws={'color': 'red', 'linewidth': 2},
            ax=ax
        )

        ax.set_title('Hubungan antara Durasi Penerbangan dan Harga Tiket', fontsize=14)
        ax.set_xlabel('Durasi Penerbangan (Jam)', fontsize=12)
        ax.set_ylabel('Harga Tiket (Price)', fontsize=12)
        plt.grid(axis='both', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('---')

        # --- 6. Harga Berdasarkan Jumlah Pemberhentian (Stops) ---
        st.header('6. Analisis Harga Berdasarkan Jumlah Pemberhentian (Stops)')
        
        group_col = 'stops'
        target_col = 'price'

        # Statistik Deskriptif Harga per Jumlah Pemberhentian
        stops_stats = df_eda.groupby(group_col)[target_col].agg(
            ['count']
        ).sort_values(by='count', ascending=False)

        st.subheader('Statistik Harga Berdasarkan Jumlah Pemberhentian (Stops)')
        st.dataframe(stops_stats)
        
        # Visualisasi Box Plot Harga per Jumlah Pemberhentian
        stops_order = stops_stats.index 
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            x=group_col,
            y=target_col,
            data=df_eda,
            order=stops_order,
            palette='viridis', 
            ax=ax
        )

        ax.set_title('Perbandingan Distribusi Harga Tiket Berdasarkan Jumlah Pemberhentian', fontsize=14)
        ax.set_xlabel('Jumlah Pemberhentian (Stops)', fontsize=12)
        ax.set_ylabel('Harga Tiket (Price)', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader('Analisis Mendalam: Hubungan Stops, Price, dan Airline/Class')

        zero_stops_airlines = df_eda[df_eda['stops'] == 'zero']['airline'].value_counts().head(5)
        st.write('**Maskapai yang Mendominasi Penerbangan Langsung (Zero Stops):**')
        st.dataframe(zero_stops_airlines.to_frame())

        one_stop_airlines = df_eda[df_eda['stops'] == 'one']['airline'].value_counts().head(5)
        st.write('**Maskapai yang Mendominasi Penerbangan One Stop:**')
        st.dataframe(one_stop_airlines.to_frame())

        st.markdown('---')

        # --- 7. Harga Berdasarkan Sisa Hari Keberangkatan (Days_left) - Line Plot ---
        st.header('7. Rata-rata Harga vs. Sisa Hari Menuju Keberangkatan (Line Plot)')
        
        # Menghitung rata-rata harga untuk setiap nilai days_left
        avg_price_by_days_left = df_eda.groupby('days_left')['price'].mean().reset_index()

        # Memvisualisasikan tren rata-rata harga
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='days_left', y='price', data=avg_price_by_days_left, marker='o', color='red', ax=ax)
        ax.set_title('Rata-rata Harga vs. Sisa Hari Menuju Keberangkatan', fontsize=16)
        ax.set_xlabel('Sisa Hari Sampai Keberangkatan (days_left)', fontsize=12)
        ax.set_ylabel('Rata-rata Harga Tiket (price)', fontsize=12)
        ax.invert_xaxis() # Membalik sumbu X
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('---')
        
        # --- 8. Harga Berdasarkan Segmentasi Sisa Hari (Days_left Group) ---
        st.header('8. Distribusi Harga Berdasarkan Kelompok Sisa Hari Keberangkatan')
        
        # Define Bins and Create Group Column
        bins = [0, 7, 30, df_eda['days_left'].max() + 1] # Menggunakan max days_left + 1 sebagai batas atas
        labels = ['< 7 Days (Last Minute)', '8 - 30 Days (Medium Term)', f'> 30 Days (Early Booking)']
        
        # Membuat kolom baru 'days_left_group' berdasarkan segmentasi
        # observed=True digunakan untuk pd.cut dalam kasus Pandas versi baru
        df_eda['days_left_group'] = pd.cut(
            df_eda['days_left'], 
            bins=bins, 
            labels=labels, 
            right=False, # Interval [start, end)
            include_lowest=True
        )

        # Calculate Descriptive Statistics
        # observed=True digunakan untuk groupby pada categorical data
        price_stats_group = df_eda.groupby('days_left_group', observed=True)['price'].agg(
            ['mean', 'median', 'count']
        ).reset_index()

        st.subheader('Statistik Harga Berdasarkan Kelompok Hari Tersisa')
        st.dataframe(price_stats_group)

        # Visualization: Box Plot for Price Distribution across Groups
        fig, ax = plt.subplots(figsize=(12, 7))
        order = ['< 7 Days (Last Minute)', '8 - 30 Days (Medium Term)', f'> 30 Days (Early Booking)']
        sns.boxplot(
            x='days_left_group', 
            y='price', 
            data=df_eda, 
            order=order,
            palette='viridis',
            ax=ax
        )

        ax.set_title('Distribusi Harga Berdasarkan Kelompok Sisa Hari Keberangkatan', fontsize=16)
        ax.set_xlabel('Kelompok Sisa Hari', fontsize=12)
        ax.set_ylabel('Harga Tiket (price)', fontsize=12)
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)


if __name__ == '__main__':
    main()