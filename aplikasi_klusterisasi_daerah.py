import streamlit as st
import pandas as pd
import random
import math
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import folium
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.pyplot as plt
from kneed import KneeLocator
import seaborn as sns
import math
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Pengelompokkan Daerah Rawan Stunting dan Gizi Buruk",
    page_icon='https://i.imgur.com/Fe489Ox.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(point1, point2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

# Inisialisasi centroid menggunakan metode K-Means++
def initialize_centroids_kmeans_plus_plus(data, k):
    centroids = []
    
    # Langkah 1: Pilih centroid pertama secara acak
    first_centroid = random.choice(data)
    centroids.append(first_centroid)
    st.write(f"Centroid Pertama Terpilih:")
    st.dataframe(np.array(first_centroid).reshape(1, -1))
    
    # Langkah 2: Pilih centroid selanjutnya berdasarkan jarak kuadrat
    for i in range(1, k):
        distances = []
        for point in data:
            min_distance = min([euclidean_distance(point, centroid) for centroid in centroids])
            distances.append(min_distance ** 2)
        
        # Konversi jarak menjadi probabilitas
        total_distance = sum(distances)
        probabilities = [distance / total_distance for distance in distances]
        
        # Pilih centroid berikutnya berdasarkan distribusi probabilitas
        next_centroid_index = random.choices(range(len(data)), probabilities)[0]
        next_centroid = data[next_centroid_index]
        centroids.append(next_centroid)
        
        st.write(f"Centroid Ke-{i + 1} Terpilih:")
        st.dataframe(np.array(next_centroid).reshape(1, -1))
    
    return centroids

# Inisialisasi centroid secara acak
def initialize_centroids_kmeans(data, k):
    centroids = random.sample(data, k)
    print("Centroid Awal Terpilih:")
    print(pd.DataFrame(centroids))
    return centroids

# Fungsi untuk mengelompokkan data berdasarkan centroid
def assign_clusters(data, centroids):
    clusters = []
    distances_all = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = distances.index(min(distances))
        clusters.append(cluster)
        distances_all.append(distances)
    return clusters, distances_all

# Fungsi untuk memperbarui centroid
def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = [data[j] for j in range(len(data)) if clusters[j] == i]
        new_centroid = [sum(dim)/len(cluster_points) for dim in zip(*cluster_points)] if cluster_points else random.choice(data)
        new_centroids.append(new_centroid)
    return new_centroids

# Algoritma K-Means konvensional
def kmeans(data, features_name, k, max_iterations=100):
    start_time = time.time()
    st.write(f"Nilai K = {k}")
    centroids = initialize_centroids_kmeans_plus_plus(data, k)
    previous_clusters = None
    iteration_count = 0

    for iteration in range(max_iterations):
        st.write(f"Iterasi {iteration + 1}")
        clusters, distances_all = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        # Menampilkan pusat centroid
        centroid_df = pd.DataFrame(centroids, columns = features_name)
        centroid_df.index = [f'{i+1}' for i in range(k)]
        st.write("Pusat centroid")
        st.write(centroid_df)

        # Menampilkan jarak terhadap pusat centroid dan group
        distance_df = pd.DataFrame(data, columns = features_name)
        for i in range(k):
            distance_df[f'C{i+1}'] = [dist[i] for dist in distances_all]
        distance_df['Cluster'] = [f'{c+1}' for c in clusters]
        st.write("\nJarak Terhadap Pusat centroid")
        st.write(distance_df)

        iteration_count += 1

        if previous_clusters == clusters:
            st.write(f"\nKelompok tidak berubah. Iterasi dihentikan.")
            break
        else:
            st.write(f"\nKelompok berubah. Iterasi dilanjutkan.")
        
        centroids = new_centroids
        previous_clusters = clusters

    end_time = time.time()
    computation_time = end_time - start_time

    # Perhitungan SSE
    sse = 0
    for i in range(k):
        cluster_points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
        if len(cluster_points) > 0:
            sse += np.sum(np.square(cluster_points - centroids[i]))

    return distance_df, centroids, computation_time, iteration_count, sse

# Menentukan nilai optimal k menggunakan metode Elbow
def optimal_k_elbow_method(data, features_names, max_k):
    sse_values = []
    for k in range(1, max_k+1):
        print(" ")
        print(f"Jumlah K =",k)
        _, _, _, _, sse = kmeans(data,features_names,k)
        sse_values.append(sse)

    # Plot kurva Elbow
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k+1), sse_values, 'bo-', color='blue')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal k')
    st.pyplot(fig)

    # Menentukan titik elbow secara otomatis
    kl = KneeLocator(range(1, max_k+1), sse_values, curve="convex", direction="decreasing")

    # Periksa apakah kl.elbow ditemukan
    if kl.elbow is None:
        st.write("Titik elbow tidak ditemukan. Menggunakan nilai k default 3.")
        optimal_k = 3  # Nilai default jika elbow tidak ditemukan
    else:
        optimal_k = kl.elbow
        st.write(f"Nilai optimal k menurut metode elbow adalah: {optimal_k}")

    return optimal_k

# Algoritma K-Means konvensional
def kmeans_konvensional(data, features_name, k, max_iterations=100):
    start_time = time.time()
    st.write(f"Nilai K = {k}")
    centroids = initialize_centroids_kmeans(data, k)
    previous_clusters = None
    iteration_count = 0

    for iteration in range(max_iterations):
        st.write(f"Iterasi {iteration + 1}")
        clusters, distances_all = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        # Menampilkan pusat centroid
        centroid_df = pd.DataFrame(centroids, columns = features_name)
        centroid_df.index = [f'{i+1}' for i in range(k)]
        st.write("Pusat centroid")
        st.write(centroid_df)

        # Menampilkan jarak terhadap pusat centroid dan group
        distance_df = pd.DataFrame(data, columns = features_name)
        for i in range(k):
            distance_df[f'C{i+1}'] = [dist[i] for dist in distances_all]
        distance_df['Cluster'] = [f'{c+1}' for c in clusters]
        st.write("\nJarak Terhadap Pusat centroid")
        st.write(distance_df)

        iteration_count += 1

        if previous_clusters == clusters:
            st.write(f"\nKelompok tidak berubah. Iterasi dihentikan.")
            break
        else:
            st.write(f"\nKelompok berubah. Iterasi dilanjutkan.")
        
        centroids = new_centroids
        previous_clusters = clusters

    end_time = time.time()
    computation_time = end_time - start_time

    # Perhitungan SSE
    sse = 0
    for i in range(k):
        cluster_points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
        if len(cluster_points) > 0:
            sse += np.sum(np.square(cluster_points - centroids[i]))

    return distance_df, centroids, computation_time, iteration_count, sse

# Menentukan nilai optimal k menggunakan metode Elbow
def optimal_k_elbow_method_kmeans(data, features_names, max_k):
    sse_values = []
    for k in range(1, max_k+1):
        print(" ")
        print(f"Jumlah K =",k)
        _, _, _, _, sse = kmeans_konvensional(data,features_names,k)
        sse_values.append(sse)

    # Plot kurva Elbow
    fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k+1), sse_values, 'bo-', color='blue')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal k')
    st.pyplot(fig)

    # Menentukan titik elbow secara otomatis
    kl = KneeLocator(range(1, max_k+1), sse_values, curve="convex", direction="decreasing")

    # Periksa apakah kl.elbow ditemukan
    if kl.elbow is None:
        st.write("Titik elbow tidak ditemukan. Menggunakan nilai k default 3.")
        optimal_k_kmeans = 3  # Nilai default jika elbow tidak ditemukan
    else:
        optimal_k_kmeans = kl.elbow
        st.write(f"Nilai optimal k menurut metode elbow adalah: {optimal_k_kmeans}")

    return optimal_k_kmeans

# Fungsi untuk menghitung SSW (Sum of Squared Within) dengan rata-rata jarak
def calculate_ssw(data_dbi):
    # Buat Kolom Baru untuk mengambil Jarak akhir yang di sesuaikan dengan Cluster yang di dapatkan
    data_dbi['Final_Distance'] = data_dbi.apply(lambda row: row[f'C{row.Cluster}'], axis=1)

    # Menghitung Rata-rata dari setiap jarak dalam satu cluster atau (SSW)
    ssw = data_dbi.groupby('Cluster')['Final_Distance'].mean()

    return ssw

# Fungsi untuk menghitung SSB (Sum of Squared Between)
def calculate_ssb(centroids, n_clusters):
    ssb = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            dist_centroids = euclidean_distance(centroids[i], centroids[j])
            ssb[i, j] = dist_centroids
            ssb[j, i] = dist_centroids  # Karena matriks SSB simetris
    return ssb

# Fungsi untuk menghitung Rasio (SSW + SSW) / SSB
def calculate_rij(ssw, ssb, n_clusters):
    rij = np.zeros((n_clusters, n_clusters))
    max_ratios = np.zeros(n_clusters)
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            if ssb[i, j] != 0:
                rij[i, j] = (ssw[i] + ssw[j]) / ssb[i, j]
                rij[j, i] = rij[i, j]  # Karena matriks simetris
        # Simpan rasio maksimum dari setiap klaster
        max_ratios= np.max(rij)
    return rij, max_ratios

# Fungsi untuk menghitung DBI
def dbi(max_ratios,n_clusters):
    return max_ratios/n_clusters



# Streamlit Title
st.title("Clustering of PMT and High-Risk Areas for Stunting and Malnutrition")
tab1, tab2, tab3 = st.tabs(['Deskripsi',"Implementasi Pengelompokkan","Hasil Optimal Pengelompokkan"])

with tab1:
    # Menampilkan gambar di tengah menggunakan HTML <img>
    st.write("""<h2 style = "text-align: center;"><img src="https://unair.ac.id/wp-content/uploads/2023/02/IL-by-OASE.png" width="600" height="300"><br></h2>""",unsafe_allow_html=True)
    st.header("Deskripsi Aplikasi")
    st.write("""<p style = "text-align: justify;">Aplikasi Ini digunakan untuk proses pengelompokkan PMT (Pemberian Makanan Tambahan) berdasarkan data Daftar Status Gizi Balita yang di ambil melalui Puskesmas. Hasil dari pengelompokkan PMT kemudian dilakukan pengelompokkan kembali untuk mengetahui penanganan stunting dan gizi berdasarkan jumlah balita pada masing-masing cluster PMT dan juga Persentase Stunting ditingkat Desa/Kelurahan. <p>""",unsafe_allow_html=True)

with tab2:
    data_upload = st.file_uploader("Masukkan Data yang ingin dilakukan Pengelompokkan")
    maks_k = st.slider("Masukkan K yang akan di ujikan",1,10,4)
    submit = st.button("Submit")
    if submit:
        subBab1, subBab2 = st.tabs(["K-Means++","K-Means"])

        with subBab1:
            subtab1, subtab2, subtab3, subtab4, subtab5, subtab6 = st.tabs(['Dataset', "Preprocessing", "K-Means++", "Hasil Akhir K-Means++", "Evalusi Cluster", "Analisis Cluster"])

            with subtab1:
                data_input = pd.read_csv(data_upload)
                st.subheader('Dataset Percobaan')
                st.dataframe(data_input)

            with subtab2:
                # Preprocessing Dataset
                # Hapus Fitur Tidak Relevan
                delete_fitur_input = ['No', 'NIK', 'Nama', 'Nama Ortu', 'Prov', 'Kab/Kota', 'Kec', 'Pukesmas', 'Posyandu', 'RT', 'RW', 'Alamat']
                data_relevan_input = data_input.drop(delete_fitur_input, axis=1)

                delete_fitur_sementara_input = ['Desa/Kel']
                data_sementara_input = data_relevan_input.drop(delete_fitur_sementara_input, axis=1)

                st.write("Dataset yang digunakan:")
                st.dataframe(data_sementara_input)

                # Label Encoding (Tranformasi Data)
                le = LabelEncoder()
                categorical_columns_input = data_sementara_input.select_dtypes(include=['object']).columns
                for column in categorical_columns_input:
                    data_sementara_input[column] = le.fit_transform(data_sementara_input[column]) + 1
                
                data_transform_input = data_sementara_input
                st.write("Data setelah Tranformasi:")
                st.dataframe(data_transform_input)

                # Normalisasi
                scaler = MinMaxScaler()
                scaled_input = scaler.fit_transform(data_transform_input)
                features_names_input = data_transform_input.columns.copy()
                #features_names.remove('label')
                scaled_features_input = pd.DataFrame(scaled_input, columns=features_names_input)

                st.write("Data setelah Normalisasi:")

                st.dataframe(scaled_features_input)

            with subtab3:
                # Eksekusi K-Means++
                data_for_clustering_input = scaled_features_input.values.tolist()
                features_input = scaled_features_input.columns.copy()
                
                optimal_k = optimal_k_elbow_method(data_for_clustering_input,features_input, maks_k)
                if optimal_k is not None:
                    final_cluster_data_input, final_centroids_input, computation_time_input, iteration_count_input, sse_input = kmeans(data_for_clustering_input, features_input, optimal_k)
                else:
                    st.write("Nilai optimal k tidak valid.")

            with subtab4:
                # Menampilkan Hasil Clusterisasi
                st.write(f"Jumlah iterasi hingga konvergen: {iteration_count_input}")
                st.write(f"Waktu komputasi yang dibutuhkan: {computation_time_input:.4f} detik")
                hasil_input = final_cluster_data_input
                st.dataframe(hasil_input)
            
            with subtab5:
                # Menampilkan Hasil Evaluasi Cluster
                # 1. Menghitung Sum Square Error (SSE)
                st.write(f"Sum of Squared Errors (SSE): {sse_input}")
                
                # 2. Menghitung Davies-Bouldin Index (DBI)
                # Jumlah cluster
                n_clusters_input = optimal_k  # jumlah klaster

                # Konversi dan centroids_akhir ke float
                centroids_input = np.array(final_centroids_input, dtype=float)

                # Mengambil data hasil perhitungan jarak pada setiap centroid serta kolom cluster
                data_dbi_input = final_cluster_data_input.iloc[:, 5:]

                # Konversi labels ke integer
                labels_input = np.array(final_cluster_data_input['Cluster'], dtype=int)

                # # Hitung SSW
                ssw_input = calculate_ssw(data_dbi_input)

                # Hitung SSB
                ssb_input = calculate_ssb(centroids_input, n_clusters_input)

                # Hitung Rasio R_ij dan rasio maksimum
                rij_input, max_ratios_input = calculate_rij(ssw_input, ssb_input, n_clusters_input)

                # Hitung Davies-Bouldin Index
                dbi_value_input = dbi(max_ratios_input,n_clusters_input)

                st.write('Tahapan Perhitungan DBI :')
                st.write(f'Jumlah Cluster: {n_clusters_input}')

                # Tampilkan SSW
                st.write('SSW (Sum of Squared Within) untuk setiap cluster:')
                st.dataframe(ssw_input)

                # Tampilkan SSB sebagai matriks
                st.write('SSB (Sum of Squared Between) antar centroid:')
                st.dataframe(pd.DataFrame(ssb_input, columns=[f'Cluster {i+1}' for i in range(n_clusters_input)], index=[f'Cluster {i+1}' for i in range(n_clusters_input)]))

                # Tampilkan Rasio (SSW + SSW) / SSB sebagai matriks
                st.write('Rasio (SSW + SSW) / SSB antar cluster:')
                st.dataframe(pd.DataFrame(rij_input, columns=[f'Cluster {i+1}' for i in range(n_clusters_input)], index=[f'Cluster {i+1}' for i in range(n_clusters_input)]))

                # Tampilkan rasio maksimum dan DBI
                st.write(f'Rasio maksimum untuk setiap cluster: {max_ratios_input}')
                st.write(f'Davies-Bouldin Index: {dbi_value_input}')
                
            
            with subtab6:
                # Menampilkan Hasil Analisis Cluster
                # 1. Menghitung mean dan standar deviasi dari setiap cluster
                hasil_gabung_input = pd.concat([final_cluster_data_input.iloc[:, :8],final_cluster_data_input['Cluster']],axis=1)
                cluster_stats_input = hasil_gabung_input.groupby('Cluster').agg(['mean', 'std'])
                st.write("Mean dan Standar Deviasi per Cluster:")
                st.write(cluster_stats_input)

                # 2. Menghitung Korelasi dari setip cluster
                # Gabungkan data sebelum di normalisasi dengan cluster untuk dilakukan analisis
                st.write("Korelasi data per Cluster:")
                data_gabung_input= pd.concat([data_transform_input,hasil_gabung_input['Cluster']],axis=1)
                # Ambil jumlah cluster
                num_clusters = data_gabung_input['Cluster'].nunique()

                # Hitung jumlah baris dan kolom untuk grid layout
                cols = 1  # Misal kita ingin 3 heatmap per baris
                rows = math.ceil(num_clusters / cols)  # Hitung jumlah baris yang dibutuhkan

                # Buat figure dengan subplots
                fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))  # Lebar dan tinggi dari figure
                axes = axes.flatten()  # Flatten agar bisa digunakan untuk indexing

                # Looping melalui setiap cluster
                for idx, cluster in enumerate(data_gabung_input['Cluster'].unique()):
                    # Mengambil data berdasarkan cluster
                    cluster_data = data_gabung_input[data_gabung_input['Cluster'] == cluster][['Umur dalam Bulan', 'Berat Badan (Kg)','Tinggi Badan (Cm)', 'BB/U', 'TB/U', 'BB/TB']]

                    # Menghitung matriks korelasi
                    correlation_matrix = cluster_data.corr()

                    # Menampilkan heatmap pada subplot yang sesuai
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=axes[idx])

                    # Set judul pada setiap heatmap
                    axes[idx].set_title(f"Heatmap Korelasi Cluster {cluster}")

                # Hapus subplot kosong jika jumlah cluster kurang dari grid
                for ax in axes[num_clusters:]:
                    fig.delaxes(ax)

                # Atur layout agar tidak tumpang tindih
                plt.tight_layout()
                plt.show()
                st.pyplot(fig)

        with subBab2:
            subtab1, subtab2, subtab3, subtab4, subtab5, subtab6 = st.tabs(['Dataset', "Preprocessing", "K-Means", "Hasil Akhir K-Means", "Evalusi Cluster", "Analisis Cluster"])

            with subtab1:
                st.subheader('Dataset Percobaan')
                st.dataframe(data_input)

            with subtab2:
                # Preprocessing Dataset
                # Hapus Fitur Tidak Relevan
                delete_fitur_input_kmeans = ['No', 'NIK', 'Nama', 'Nama Ortu', 'Prov', 'Kab/Kota', 'Kec', 'Pukesmas', 'Posyandu', 'RT', 'RW', 'Alamat']
                data_relevan_input_kmeans = data_input.drop(delete_fitur_input_kmeans, axis=1)

                delete_fitur_sementara_input_kmeans = ['Desa/Kel']
                data_sementara_input_kmeans = data_relevan_input_kmeans.drop(delete_fitur_sementara_input_kmeans, axis=1)

                st.write("Dataset yang digunakan:")
                st.dataframe(data_sementara_input_kmeans)

                # Label Encoding (Tranformasi Data)
                le = LabelEncoder()
                categorical_columns_input_kmeans = data_sementara_input_kmeans.select_dtypes(include=['object']).columns
                for column in categorical_columns_input_kmeans:
                    data_sementara_input_kmeans[column] = le.fit_transform(data_sementara_input_kmeans[column]) + 1
                
                data_transform_input_kmeans = data_sementara_input_kmeans
                st.write("Data setelah Tranformasi:")
                st.dataframe(data_transform_input_kmeans)

                # Normalisasi
                scaler = MinMaxScaler()
                scaled_input_kmeans = scaler.fit_transform(data_transform_input_kmeans)
                features_names_input_kmeans = data_transform_input_kmeans.columns.copy()
                #features_names.remove('label')
                scaled_features_input_kmeans = pd.DataFrame(scaled_input_kmeans, columns=features_names_input_kmeans)

                st.write("Data setelah Normalisasi:")

                st.dataframe(scaled_features_input_kmeans)

            with subtab3:
                # Eksekusi K-Means++
                data_for_clustering_input_kmeans = scaled_features_input_kmeans.values.tolist()
                features_input_kmeans = scaled_features_input_kmeans.columns.copy()
                
                optimal_k_kmeans = optimal_k_elbow_method_kmeans(data_for_clustering_input_kmeans,features_input_kmeans, maks_k)
                if optimal_k_kmeans is not None:
                    final_cluster_data_input_kmeans, final_centroids_input_kmeans, computation_time_input_kmeans, iteration_count_input_kmeans, sse_input_kmeans = kmeans_konvensional(data_for_clustering_input_kmeans, features_input_kmeans, optimal_k)
                else:
                    st.write("Nilai optimal k tidak valid.")

            with subtab4:
                # Menampilkan Hasil Clusterisasi
                st.write(f"Jumlah iterasi hingga konvergen: {iteration_count_input_kmeans}")
                st.write(f"Waktu komputasi yang dibutuhkan: {computation_time_input_kmeans:.4f} detik")
                hasil_input_kmeans = final_cluster_data_input_kmeans
                st.dataframe(hasil_input_kmeans)
            
            with subtab5:
                # Menampilkan Hasil Evaluasi Cluster
                # 1. Menghitung Sum Square Error (SSE)
                st.write(f"Sum of Squared Errors (SSE): {sse_input_kmeans}")
                
                # 2. Menghitung Davies-Bouldin Index (DBI)
                # Jumlah cluster
                n_clusters_input_kmeans = optimal_k_kmeans  # jumlah klaster

                # Konversi dan centroids_akhir ke float
                centroids_input_kmeans = np.array(final_centroids_input_kmeans, dtype=float)

                # Mengambil data hasil perhitungan jarak pada setiap centroid serta kolom cluster
                data_dbi_input_kmeans = final_cluster_data_input_kmeans.iloc[:, 5:]

                # Konversi labels ke integer
                labels_input_kmeans = np.array(final_cluster_data_input_kmeans['Cluster'], dtype=int)

                # # Hitung SSW
                ssw_input_kmeans = calculate_ssw(data_dbi_input_kmeans)

                # Hitung SSB
                ssb_input_kmeans = calculate_ssb(centroids_input_kmeans, n_clusters_input_kmeans)

                # Hitung Rasio R_ij dan rasio maksimum
                rij_input_kmeans, max_ratios_input_kmeans = calculate_rij(ssw_input_kmeans, ssb_input_kmeans, n_clusters_input_kmeans)

                # Hitung Davies-Bouldin Index
                dbi_value_input_kmeans = dbi(max_ratios_input_kmeans,n_clusters_input_kmeans)

                st.write('Tahapan Perhitungan DBI :')
                st.write(f'Jumlah Cluster: {n_clusters_input_kmeans}')

                # Tampilkan SSW
                st.write('SSW (Sum of Squared Within) untuk setiap cluster:')
                st.dataframe(ssw_input_kmeans)

                # Tampilkan SSB sebagai matriks
                st.write('SSB (Sum of Squared Between) antar centroid:')
                st.dataframe(pd.DataFrame(ssb_input_kmeans, columns=[f'Cluster {i+1}' for i in range(n_clusters_input_kmeans)], index=[f'Cluster {i+1}' for i in range(n_clusters_input_kmeans)]))

                # Tampilkan Rasio (SSW + SSW) / SSB sebagai matriks
                st.write('Rasio (SSW + SSW) / SSB antar cluster:')
                st.dataframe(pd.DataFrame(rij_input_kmeans, columns=[f'Cluster {i+1}' for i in range(n_clusters_input_kmeans)], index=[f'Cluster {i+1}' for i in range(n_clusters_input_kmeans)]))

                # Tampilkan rasio maksimum dan DBI
                st.write(f'Rasio maksimum untuk setiap cluster: {max_ratios_input_kmeans}')
                st.write(f'Davies-Bouldin Index: {dbi_value_input_kmeans}')
                
            
            with subtab6:
                # Menampilkan Hasil Analisis Cluster
                # 1. Menghitung mean dan standar deviasi dari setiap cluster
                hasil_gabung_input_kmeans = pd.concat([final_cluster_data_input_kmeans.iloc[:, :8],final_cluster_data_input_kmeans['Cluster']],axis=1)
                cluster_stats_input_kmeans = hasil_gabung_input_kmeans.groupby('Cluster').agg(['mean', 'std'])
                st.write("Mean dan Standar Deviasi per Cluster:")
                st.write(cluster_stats_input_kmeans)

                # 2. Menghitung Korelasi dari setip cluster
                # Gabungkan data sebelum di normalisasi dengan cluster untuk dilakukan analisis
                st.write("Korelasi data per Cluster:")
                data_gabung_input_kmeans= pd.concat([data_transform_input_kmeans,hasil_gabung_input_kmeans['Cluster']],axis=1)
                # Ambil jumlah cluster
                num_clusters = data_gabung_input_kmeans['Cluster'].nunique()

                # Hitung jumlah baris dan kolom untuk grid layout
                cols = 1  # Misal kita ingin 3 heatmap per baris
                rows = math.ceil(num_clusters / cols)  # Hitung jumlah baris yang dibutuhkan

                # Buat figure dengan subplots
                fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))  # Lebar dan tinggi dari figure
                axes = axes.flatten()  # Flatten agar bisa digunakan untuk indexing

                # Looping melalui setiap cluster
                for idx, cluster in enumerate(data_gabung_input_kmeans['Cluster'].unique()):
                    # Mengambil data berdasarkan cluster
                    cluster_data = data_gabung_input_kmeans[data_gabung_input_kmeans['Cluster'] == cluster][['Umur dalam Bulan', 'Berat Badan (Kg)','Tinggi Badan (Cm)', 'BB/U', 'TB/U', 'BB/TB']]

                    # Menghitung matriks korelasi
                    correlation_matrix = cluster_data.corr()

                    # Menampilkan heatmap pada subplot yang sesuai
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=axes[idx])

                    # Set judul pada setiap heatmap
                    axes[idx].set_title(f"Heatmap Korelasi Cluster {cluster}")

                # Hapus subplot kosong jika jumlah cluster kurang dari grid
                for ax in axes[num_clusters:]:
                    fig.delaxes(ax)

                # Atur layout agar tidak tumpang tindih
                plt.tight_layout()
                plt.show()
                st.pyplot(fig)    
                

with tab3:
    # Halaman Hasil Pengujian Metode Terbaik Dengan Data Daftar Status Gizi Balita dari Puskesmas
    pilih = st.selectbox("Pilihlah Tahun Yang Akan Anda Tampilkan",
    ("2019", "2020", "2021", "2022", "2023"),)

    if pilih == "2019":
        data_input = pd.read_csv("https://raw.githubusercontent.com/HanifSantoso05/Dataset_Skripsi/main/2019.csv")
        # Centroid Pertama Terpilih
        centroid_1_PMT = np.array([[0.0, 1.0, 0.422785, 0.682175, 0.666667, 0.333333, 0.75, 1.0]])
        # Centroid Ke-2 terpilih
        centroid_2_PMT = np.array([[0.0, 0.0, 0.287764, 0.527188, 0.666667, 0.666667, 0.25, 1.0]])
        # Centroid Ke-3 terpilih
        centroid_3_PMT = np.array([[1.0, 0.454545, 0.557806, 0.682175, 0.666667, 0.666667, 0.5, 1.0]])
        # Centroid Ke-4 terpilih
        centroid_4_PMT = np.array([[1.0, 0.090909, 0.537553, 0.369293, 0.666667, 0.666667, 0.5, 0.0]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_PMT = np.vstack([centroid_1_PMT, centroid_2_PMT, centroid_3_PMT, centroid_4_PMT])

        # Centroid Pertama Terpilih
        centroid_1_daerah = np.array([[0.351852, 0.333333, 0.185185, 0.016949, 0.346682]])
        # Centroid Ke-2 terpilih
        centroid_2_daerah = np.array([[0.277778, 0.111111, 0.055556, 0.016949, 0.888132]])
        # Centroid Ke-3 terpilih
        centroid_3_daerah = np.array([[0.148148, 0.203704, 0.259259, 0.152542, 0.0]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_daerah = np.vstack([centroid_1_daerah, centroid_2_daerah, centroid_3_daerah])

    elif pilih == "2020":
        data_input = pd.read_csv("https://raw.githubusercontent.com/HanifSantoso05/Dataset_Skripsi/main/2020.csv")
        # Centroid Pertama Terpilih
        centroid_1_PMT = np.array([[0.0, 0.090909, 0.254545, 0.488131, 0.666667, 0.666667, 0.25, 1.0]])
        # Centroid Ke-2 terpilih
        centroid_2_PMT = np.array([[1.0, 0.636364, 0.373554, 0.3521, 0.666667, 0.666667, 0.5, 0.0]])
        # Centroid Ke-3 terpilih
        centroid_3_PMT = np.array([[1.0, 0.090909, 0.28595, 0.412355, 0.666667, 0.666667, 0.25, 1.0]])
        # Centroid Ke-4 terpilih
        centroid_4_PMT = np.array([[0.0, 0.727273, 0.459504, 0.505478, 0.666667, 0.666667, 0.25, 0.0]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_PMT = np.vstack([centroid_1_PMT, centroid_2_PMT, centroid_3_PMT, centroid_4_PMT])
        
        # Centroid Pertama Terpilih
        centroid_1_daerah = np.array([[0.5, 0.34375, 0.592593, 0.212121  , 0.305402]])
        # Centroid Ke-2 terpilih
        centroid_2_daerah = np.array([[0.392857, 0.0625, 0.37037, 0.242424, 0.0]])
        # Centroid Ke-3 terpilih
        centroid_3_daerah = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_daerah = np.vstack([centroid_1_daerah, centroid_2_daerah, centroid_3_daerah])

    elif pilih == "2021":
        data_input = pd.read_csv("https://raw.githubusercontent.com/HanifSantoso05/Dataset_Skripsi/main/2021.csv")
        # Centroid Pertama Terpilih
        centroid_1_PMT = np.array([[1.0, 0.818182, 0.498634, 0.695122, 0.666667, 0.5, 0.25, 0.0]])
        # Centroid Ke-2 terpilih
        centroid_2_PMT = np.array([[0.0, 0.454545, 0.188525, 0.548063, 0.0, 0.5, 0.0, 0.0]])
        # Centroid Ke-3 terpilih
        centroid_3_PMT = np.array([[1.0, 0.636364, 0.430328, 0.740316, 0.666667, 0.5, 0.25, 1.0]])
        # Centroid Ke-4 terpilih
        centroid_4_PMT = np.array([[0.0, 0.636364, 0.672814, 0.809541, 0.666667, 0.5, 0.5, 1.0]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_PMT = np.vstack([centroid_1_PMT, centroid_2_PMT, centroid_3_PMT, centroid_4_PMT])
        
        # Centroid Pertama Terpilih
        centroid_1_daerah = np.array([[0.277778, 0.0, 0.0, 0.2, 0.0]])
        # Centroid Ke-2 terpilih
        centroid_2_daerah = np.array([[0.777778, 1.0, 1.0, 0.35, 0.300466]])
        # Centroid Ke-3 terpilih
        centroid_3_daerah = np.array([[0.0, 0.5, 0.272727, 0.6, 0.616579]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_daerah = np.vstack([centroid_1_daerah, centroid_2_daerah, centroid_3_daerah])

    elif pilih == "2022":
        data_input = pd.read_csv("https://raw.githubusercontent.com/HanifSantoso05/Dataset_Skripsi/main/2022.csv")
        # Centroid Pertama Terpilih
        centroid_1_PMT = np.array([[0.0, 0.181818, 0.297565, 0.5241, 0.666667, 0.666667, 0.0, 1.0]])
        # Centroid Ke-2 terpilih
        centroid_2_PMT = np.array([[1.0, 1.0, 0.619482, 0.369048, 0.666667, 0.333333, 0.4, 0.0]])
        # Centroid Ke-3 terpilih
        centroid_3_PMT = np.array([[0.0, 0.636364, 0.776256, 0.560105, 1.0, 0.666667, 0.4, 1.0]])
        # Centroid Ke-4 terpilih
        centroid_4_PMT = np.array([[1.0, 0.545455, 0.174277, 0.307491, 0.333333, 0.333333, 0.0, 1.0]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_PMT = np.vstack([centroid_1_PMT, centroid_2_PMT, centroid_3_PMT, centroid_4_PMT])
        
        # Centroid Pertama Terpilih
        centroid_1_daerah = np.array([[1.0, 1.0, 1.0, 1.0, 0.476418]])
        # Centroid Ke-2 terpilih
        centroid_2_daerah = np.array([[0.096774,  0.029412, 0.411765, 0.0, 1.0]])
        # Centroid Ke-3 terpilih
        centroid_3_daerah = np.array([[0.516129, 0.205882, 0.823529, 0.393939, 0.769586]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_daerah = np.vstack([centroid_1_daerah, centroid_2_daerah, centroid_3_daerah])

    elif pilih == "2023":
        data_input = pd.read_csv("https://raw.githubusercontent.com/HanifSantoso05/Dataset_Skripsi/main/2023.csv")
        # Centroid Pertama Terpilih
        centroid_1_PMT = np.array([[1.0, 0.090909, 0.364415, 0.156584, 0.666667, 0.333333, 0.5, 1.0]])
        # Centroid Ke-2 terpilih
        centroid_2_PMT = np.array([[0.0, 0.909091, 0.630828, 0.591394, 0.666667, 0.666667, 0.25, 0.0]])
        # Centroid Ke-3 terpilih
        centroid_3_PMT = np.array([[0.0, 0.181818, 0.679353, 0.576836, 0.666667, 0.666667, 0.5, 1.0]])
        # Centroid Ke-4 terpilih
        centroid_4_PMT = np.array([[1.0, 0.636364, 0.627973, 0.731155, 0.666667, 0.666667, 0.25, 0.0]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_PMT = np.vstack([centroid_1_PMT, centroid_2_PMT, centroid_3_PMT, centroid_4_PMT])
        
        # Centroid Pertama Terpilih
        centroid_1_daerah = np.array([[0.666667, 0.6, 0.428571, 0.333333, 0.0]])
        # Centroid Ke-2 terpilih
        centroid_2_daerah = np.array([[1.0, 0.4, 0.857143, 0.666667, 0.352871]])
        # Centroid Ke-3 terpilih
        centroid_3_daerah = np.array([[0.0, 0.2, 0.0, 0.0, 0.200495]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_daerah = np.vstack([centroid_1_daerah, centroid_2_daerah, centroid_3_daerah])

    else:
        data_input = pd.read_csv("https://raw.githubusercontent.com/HanifSantoso05/Dataset_Skripsi/main/2019.csv")
        # Centroid Pertama Terpilih
        centroid_1_PMT = np.array([[0.0, 1.0, 0.422785, 0.682175, 0.666667, 0.333333, 0.75, 1.0]])
        # Centroid Ke-2 terpilih
        centroid_2_PMT = np.array([[0.0, 0.0, 0.287764, 0.527188, 0.666667, 0.666667, 0.25, 1.0]])
        # Centroid Ke-3 terpilih
        centroid_3_PMT = np.array([[1.0, 0.454545, 0.557806, 0.682175, 0.666667, 0.666667, 0.5, 1.0]])
        # Centroid Ke-4 terpilih
        centroid_4_PMT = np.array([[1.0, 0.090909, 0.537553, 0.369293, 0.666667, 0.666667, 0.5, 0.0]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_PMT = np.vstack([centroid_1_PMT, centroid_2_PMT, centroid_3_PMT, centroid_4_PMT])

        # Centroid Pertama Terpilih
        centroid_1_daerah = np.array([[0.351852, 0.333333, 0.185185, 0.016949, 0.346682]])
        # Centroid Ke-2 terpilih
        centroid_2_daerah = np.array([[0.277778, 0.111111, 0.055556, 0.016949, 0.888132]])
        # Centroid Ke-3 terpilih
        centroid_3_daerah = np.array([[0.148148, 0.203704, 0.259259, 0.152542, 0.0]])
        # Gabungkan ketiga centroid menjadi satu array
        initial_centroids_daerah = np.vstack([centroid_1_daerah, centroid_2_daerah, centroid_3_daerah])

    

    subTab1,subTab2 = st.tabs(["Cluster PMT","Cluster Daerah"])
    with subTab1:        
        subtab1, subtab2, subtab3, subtab4, subtab5, subtab6 = st.tabs(["Preprocessing", "K-Means++", "Hasil Akhir K-Means++", "Evalusi Cluster", "Analisis Cluster", "Hasil Pemetaan Pengelompokkan PMT"])
        with subtab1:
            # Menampilkan Data Tahun
            st.write('Data Tahun ',pilih)
            st.dataframe(data_input)

            # Preprocessing Dataset
            # 1. Menghapus Fitur Tidak Relevan
            delete_fitur = ['No', 'NIK', 'Nama', 'Nama Ortu', 'Prov', 'Kab/Kota', 'Kec', 'Pukesmas', 'Posyandu', 'RT', 'RW', 'Alamat']
            data_relevan = data_input.drop(delete_fitur, axis=1)

            delete_fitur_sementara = ['Desa/Kel']
            data_sementara = data_relevan.drop(delete_fitur_sementara, axis=1)

            st.write("Dataset yang digunakan:")
            st.dataframe(data_sementara)

            # 2. Label Encoding (Transformasi Data)
            # Definisi kan setiap kolom yang akan di lakukan tranformasi
            J_K = {
                "L	": 1,
                "P	": 2
            }

            bb_u_mapping = {
                "Berat badan sangat kurang": 1,
                "Berat badan kurang": 2,
                "Berat badan normal": 3,
                "Resiko berat badan lebih": 4
            }

            tb_u_mapping = {
                "Sangat Pendek": 1,
                "Pendek": 2,
                "Normal": 3,
                "Tinggi": 4
            }

            bb_tb_mapping = {
                "Gizi buruk": 1,
                "Gizi kurang": 2,
                "Gizi baik": 3,
                "Beresiko gizi lebih": 4,
                "Gizi Lebih": 5,
                "Obesitas" : 6
            }

            naik_bb_mapping = {
                "Naik": 1,
                "Tidak Naik": 2
            }

            # Terapkan setiap variable yang di definisikan dengan dataset relevan yang akan di transformasi
            le = LabelEncoder()
            data_sementara['JK'] = le.fit_transform(data_sementara['JK']) + 1
            data_sementara['BB/U'] = data_sementara['BB/U'].map(bb_u_mapping)
            data_sementara['TB/U'] = data_sementara['TB/U'].map(tb_u_mapping)
            data_sementara['BB/TB'] = data_sementara['BB/TB'].map(bb_tb_mapping)
            data_sementara['Naik Berat Badan'] = data_sementara['Naik Berat Badan'].map(naik_bb_mapping)

            # Tampilkan hasil dari transformasi
            data_transform = data_sementara
            st.write("Data setelah Tranformasi:")
            st.dataframe(data_transform)

            # 3. Normalisasi
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data_transform)
            features_names = data_transform.columns.copy()
            #features_names.remove('label')
            scaled_features = pd.DataFrame(scaled, columns=features_names)

            st.write("Data setelah Normalisasi:")

            st.dataframe(scaled_features)

        with subtab2:
            def euclidean_distance_fix(point1, point2):
                return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

            def menjalankan_pengelompokkan_fix(data, centroids):
                clusters = []
                jarak_keseluruhan = []
                for point in data:
                    jarak = [euclidean_distance_fix(point, centroid) for centroid in centroids]
                    cluster = jarak.index(min(jarak))
                    clusters.append(cluster)
                    jarak_keseluruhan.append(jarak)
                return clusters, jarak_keseluruhan


            # Fungsi untuk memperbarui centroid
            def update_centroids_fix(data, clusters, k):
                keseluruhan_centroids_baru = []
                for i in range(k):
                    cluster_points = [data[j] for j in range(len(data)) if clusters[j] == i]
                    centroid_baru = [sum(dim)/len(cluster_points) for dim in zip(*cluster_points)] if cluster_points else random.choice(data)
                    keseluruhan_centroids_baru.append(centroid_baru)
                return keseluruhan_centroids_baru

            # Algoritma K-Means konvensional
            def kmeans_fix(data, features_name, k, centroids, max_iterations=100):
                start_time = time.time() 
                previous_clusters = None
                iteration_count = 0

                for iteration in range(max_iterations):
                    st.write(f"Iterasi {iteration + 1}")
                    clusters, distances_all = menjalankan_pengelompokkan_fix(data, centroids)
                    new_centroids = update_centroids_fix(data, clusters, k)

                    # Menampilkan pusat centroid
                    centroid_df = pd.DataFrame(centroids, columns = features_name)
                    centroid_df.index = [f'{i+1}' for i in range(k)]
                    st.write("Pusat centroid")
                    st.write(centroid_df)

                    # Menampilkan jarak terhadap pusat centroid dan group
                    distance_df = pd.DataFrame(data, columns = features_name)
                    for i in range(k):
                        distance_df[f'C{i+1}'] = [dist[i] for dist in distances_all]
                    distance_df['Cluster'] = [f'{c+1}' for c in clusters]
                    st.write("\nJarak Terhadap Pusat centroid")
                    st.write(distance_df)

                    iteration_count += 1

                    if previous_clusters == clusters:
                        st.write(f"\nKelompok tidak berubah. Iterasi dihentikan.")
                        break
                    else:
                        st.write(f"\nKelompok berubah. Iterasi dilanjutkan.")
                    
                    centroids = new_centroids
                    previous_clusters = clusters

                end_time = time.time()
                computation_time = end_time - start_time

                # Perhitungan SSE
                sse = 0
                for i in range(k):
                    cluster_points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
                    if len(cluster_points) > 0:
                        sse += np.sum(np.square(cluster_points - centroids[i]))

                return distance_df, centroids, computation_time, iteration_count, sse

            # Eksekusi Metode Pengelompokkan 
            data_for_clustering = scaled_features.values.tolist() 
            features = scaled_features.columns.copy()
            k = 4
            final_cluster_data, final_centroids, computation_time, iteration_count, sse = kmeans_fix(data_for_clustering, features, k, initial_centroids_PMT)


        with subtab3:
            # Menampilkan Hasil Clsuterisasi    
            st.write(f"Jumlah iterasi hingga konvergen: {iteration_count}")
            st.write(f"Waktu komputasi yang dibutuhkan: {computation_time:.4f} detik")
            centroid_akhir = pd.DataFrame(final_centroids)
            labels = final_cluster_data['Cluster']
            labels = np.array(labels, dtype=int)
            hasil = final_cluster_data
            st.dataframe(hasil)


            # Misalnya data Anda berbentuk numpy array atau dataframe
            X = final_cluster_data.iloc[:, :8]
            y = hasil['Cluster'].values.astype(int)  # Cluster dalam bentuk angka atau label
            centroids = centroid_akhir

            # Mapping angka kluster ke nama kluster (jika diperlukan)
            cluster_names = {1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3', 4: 'Cluster 4'}  # Sesuaikan dengan kluster Anda

            # Menggunakan PCA untuk mereduksi dari 5 dimensi ke 2 dimensi
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            # Transformasi centroid dari 5D ke 2D menggunakan PCA yang sudah di-fit
            centroids_pca = pca.transform(centroids)

            # Plot hasil PCA dari data dan centroid
            plt.figure(figsize=(6, 6))

            # Plot setiap cluster dengan label dinamis
            for cluster in np.unique(y):
                plt.scatter(X_pca[y == cluster, 0], X_pca[y == cluster, 1], 
                            label=cluster_names.get(cluster, f'Cluster {cluster}'), s=50)

            # Plot centroid
            plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', s=200, alpha=0.75, marker='x', label='Centroids')

            # Tambahkan nilai centroid ke plot
            for i, (x, y) in enumerate(centroids_pca):
                plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=9, ha='right', color='black')

            plt.title("Plot Hasil Pengelompokkan PMT")
            plt.xlabel("Data")
            plt.ylabel("Cluster")
            plt.legend()
            plt.show()
            col1, col2, col3 = st.columns([3, 1, 1])  
            with col1:
                st.pyplot(plt)

            
        with subtab4:
            # Menampilkan Hasil Evaluasi Cluster
            # 1. Menghitung Sum Square Error (SSE)
            st.write(f"Sum of Squared Errors (SSE): \n{sse}")

            # 2. Menghitung Davies-Bouldin Index (DBI)
            # Jumlah cluster
            n_clusters = k  # jumlah klaster

            # Konversi centroids_akhir ke float
            centroids = np.array(centroid_akhir, dtype=float)

            # Mengambil data hasil perhitungan jarak pada setiap centroid serta kolom cluster
            data_dbi = final_cluster_data.iloc[:, 8:]

            # Konversi labels ke integer
            labels = np.array(final_cluster_data['Cluster'], dtype=int)

            # # Hitung SSW
            ssw = calculate_ssw(data_dbi)

            # Hitung SSB
            ssb = calculate_ssb(centroids, n_clusters)

            # Hitung Rasio R_ij dan rasio maksimum
            rij, max_ratios = calculate_rij(ssw, ssb, n_clusters)

            # Hitung Davies-Bouldin Index
            dbi_value = dbi(max_ratios,n_clusters)

            # Tampilkan hasil dalam bentuk dataframe
            st.write('Tahapan Perhitungan DBI :')
            st.write(f'Jumlah Cluster: {n_clusters}')

            # Tampilkan SSW
            st.write('SSW (Sum of Squared Within) untuk setiap cluster:')
            st.dataframe(ssw)

            # Tampilkan SSB sebagai matriks
            st.write('SSB (Sum of Squared Between) antar centroid:')
            st.dataframe(pd.DataFrame(ssb, columns=[f'Cluster {i+1}' for i in range(n_clusters)], index=[f'Cluster {i+1}' for i in range(n_clusters)]))

            # Tampilkan Rasio (SSW + SSW) / SSB sebagai matriks
            st.write('Rasio (SSW + SSW) / SSB antar cluster:')
            st.dataframe(pd.DataFrame(rij, columns=[f'Cluster {i+1}' for i in range(n_clusters)], index=[f'Cluster {i+1}' for i in range(n_clusters)]))

            # Tampilkan rasio maksimum dan DBI
            st.write(f'Rasio maksimum untuk setiap cluster: {max_ratios}')
            st.write(f'Davies-Bouldin Index: {dbi_value}')

        with subtab5:
            # Menampilkan Hasil Analisis Cluster
            # 1. Menghitung mean dan standar deviasi dari setiap cluster
            data_gabung= pd.concat([data_transform,hasil['Cluster']],axis=1)
            cluster_stats = data_gabung.groupby('Cluster').agg(['mean', 'std'])
            st.write("Mean dan Standar Deviasi per Cluster:")
            st.write(cluster_stats)

            # 2. Menghitung Korelasi dari setip cluster
            # Gabungkan data sebelum di normalisasi dengan cluster untuk dilakukan analisis
            st.write("Korelasi data per Cluster:")
            data_gabung= pd.concat([data_transform,hasil['Cluster']],axis=1)
            # Ambil jumlah cluster
            num_clusters = data_gabung['Cluster'].nunique()

            # Hitung jumlah baris dan kolom untuk grid layout
            cols = 1  # Misal ingin bentuk 3 heatmap per baris
            rows = math.ceil(num_clusters / cols)  # Hitung jumlah baris yang dibutuhkan

            # Buat figure dengan subplots
            fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))  # Lebar dan tinggi dari figure
            axes = axes.flatten()  # Flatten agar bisa digunakan untuk indexing

            # Looping melalui setiap cluster
            for idx, cluster in enumerate(data_gabung['Cluster'].unique()):
                # Mengambil data berdasarkan cluster
                cluster_data = data_gabung[data_gabung['Cluster'] == cluster][['Umur dalam Bulan', 'Berat Badan (Kg)','Tinggi Badan (Cm)', 'BB/U', 'TB/U', 'BB/TB']]

                # Menghitung matriks korelasi
                correlation_matrix = cluster_data.corr()

                # Menampilkan heatmap pada subplot yang sesuai
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=axes[idx], annot_kws={"size": 16})

                # Set judul pada setiap heatmap
                axes[idx].set_title(f"Heatmap Korelasi Cluster {cluster}")

            # Hapus subplot kosong jika jumlah cluster kurang dari grid
            for ax in axes[num_clusters:]:
                fig.delaxes(ax)

            # Atur layout agar tidak tumpang tindih
            plt.tight_layout()
            plt.show()
            st.pyplot(fig)
        
        with subtab6:
            st.header("Pemetaan Hasil Pengelompokkan PMT")
            # # Persiapan Dataset
            data_jumlah_cluster = pd.concat([data_relevan['Desa/Kel'], hasil['Cluster']], axis=1)

            # Mengelompokkan data berdasarkan 'Desa/Kel' dan 'Cluster' untuk menghitung jumlah cluster di setiap desa/kel
            cluster_counts = data_jumlah_cluster.groupby(['Desa/Kel', 'Cluster']).size().unstack(fill_value=0)

            # Menambahkan kolom jumlah total anak di masing-masing desa
            cluster_counts['Total Anak'] = cluster_counts.sum(axis=1)

            # Reset indeks untuk meratakan DataFrame (tidak ada indeks multi-level)
            cluster_counts = cluster_counts.reset_index()

            # Menampilkan Hasil data jumlah masing-masing cluster PMT dalam satu Desa/Kel
            st.write("A. Data banyaknya jumlah anak pada setiap cluster PMT dalam satu Desa/Kel, beserta totalnya")
            st.dataframe(cluster_counts)
            st.write("Jumlah keseluruhan anak berdasarkan data tahun",pilih,"=",str(cluster_counts['Total Anak'].sum()),"Anak")

            # Upload data koordinat
            st.write("B. Penggabungan Data Jumlah Balita Masing-masing Cluster dengan Koordinat Wilayah")
            koor = pd.read_csv("https://raw.githubusercontent.com/HanifSantoso05/Dataset_Skripsi/main/koordinat_wilayah.csv")
            data_koordinat = pd.concat([cluster_counts.drop(columns=['Total Anak']), koor['polygons']], axis=1)
            st.dataframe(data_koordinat)

            # Pemetaan Masing-masing Cluster
            st.write("1. Pemetaan Jumlah Anak Pada Cluster 1")
            # Fungsi untuk mengonversi string polygon menjadi daftar tuple
            def parse_polygon(polygon_str):
                points = polygon_str.replace('|', ',').split(' ')
                return [(float(point.split(',')[1]), float(point.split(',')[0])) for point in points]
            
            # Menghitung total jumlah anak di seluruh desa pada cluster 1
            total_anak_cluster_1 = data_koordinat['1'].sum()

            
            # Menentukan lokasi peta rata-rata
            average_lat_cluster_1 = data_koordinat['polygons'].apply(lambda x: parse_polygon(x)[0][0]).mean()
            average_lon_cluster_1 = data_koordinat['polygons'].apply(lambda x: parse_polygon(x)[0][1]).mean()

            m_cluster_1 = folium.Map(location=[average_lat_cluster_1, average_lon_cluster_1], zoom_start=12)

            # Normalisasi jumlah anak untuk intensitas warna
            max_jumlah_anak_cluster_1 = data_koordinat['1'].max()
            min_jumlah_anak_cluster_1 = data_koordinat['1'].min()

            for _, row in data_koordinat.iterrows():
                polygon_points_cluster_1 = parse_polygon(row['polygons'])
                jumlah_anak_cluster_1 = row['1']
                
                # Mengatur intensitas warna berdasarkan jumlah anak
                opacity_cluster_1 = 0.2 + 0.5 * (jumlah_anak_cluster_1 - min_jumlah_anak_cluster_1) / (max_jumlah_anak_cluster_1 - min_jumlah_anak_cluster_1)
                
                # Buat konten popup
                popup_content = f"<b>Desa/Kel:</b> {row['Desa/Kel']}<br><b>Jumlah Anak:</b> {jumlah_anak_cluster_1}"
                
                folium.Polygon(
                    locations=polygon_points_cluster_1,
                    color='blue',  # Outline color set to match fill color
                    fill=True,
                    fill_color='blue',  # Warna dasar, intensitas diatur oleh opacity
                    fill_opacity=opacity_cluster_1,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(m_cluster_1)

            # Menambahkan informasi total jumlah anak di pojok kanan atas menggunakan CSS
            title_html_cluster_1 = f'''
                <div style="position: fixed; 
                            top: 10px; right: 10px; width: 200px; height: 30px; 
                            background-color: white; z-index:9999; 
                            font-size: 14px; font-weight: bold; color: blue;
                            border:2px solid blue; padding: 4px;">
                    Total Anak Cluster 1: {total_anak_cluster_1}
                </div>
                '''

            # Tambahkan HTML ke peta
            m_cluster_1.get_root().html.add_child(folium.Element(title_html_cluster_1))

            # Render peta ke dalam objek HTML dan tampilkan di Streamlit
            map_html_cluster_1 = m_cluster_1._repr_html_()
            st.components.v1.html(map_html_cluster_1, width=700, height=300)


            # Pemetaan Masing-masing Cluster
            st.write("2. Pemetaan Jumlah Anak Pada Cluster 2")

            # Menghitung total jumlah anak di seluruh desa pada cluster 2
            total_anak_cluster_2 = data_koordinat['2'].sum()

            # Menentukan lokasi peta rata-rata
            average_lat_cluster_2 = data_koordinat['polygons'].apply(lambda x: parse_polygon(x)[0][0]).mean()
            average_lon_cluster_2 = data_koordinat['polygons'].apply(lambda x: parse_polygon(x)[0][1]).mean()

            m_cluster_2 = folium.Map(location=[average_lat_cluster_2, average_lon_cluster_2], zoom_start=12)

            # Normalisasi jumlah anak untuk intensitas warna
            max_jumlah_anak_cluster_2 = data_koordinat['2'].max()
            min_jumlah_anak_cluster_2 = data_koordinat['2'].min()

            for _, row in data_koordinat.iterrows():
                polygon_points_cluster_2 = parse_polygon(row['polygons'])
                jumlah_anak_cluster_2 = row['2']
                
                # Mengatur intensitas warna berdasarkan jumlah anak
                opacity_cluster_2 = 0.2 + 0.5 * (jumlah_anak_cluster_2 - min_jumlah_anak_cluster_2) / (max_jumlah_anak_cluster_2 - min_jumlah_anak_cluster_2)
                
                # Buat konten popup
                popup_content = f"<b>Desa/Kel:</b> {row['Desa/Kel']}<br><b>Jumlah Anak:</b> {jumlah_anak_cluster_2}"
                
                folium.Polygon(
                    locations=polygon_points_cluster_2,
                    color='green',  # Outline color set to match fill color
                    fill=True,
                    fill_color='green',  # Warna dasar, intensitas diatur oleh opacity
                    fill_opacity=opacity_cluster_2,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(m_cluster_2)

            # Menambahkan informasi total jumlah anak di pojok kanan atas menggunakan CSS
            title_html_cluster_2 = f'''
                <div style="position: fixed; 
                            top: 10px; right: 10px; width: 200px; height: 30px; 
                            background-color: white; z-index:9999; 
                            font-size: 14px; font-weight: bold; color: green;
                            border:2px solid green; padding: 4px;">
                    Total Anak Cluster 2: {total_anak_cluster_2}
                </div>
                '''

            # Tambahkan HTML ke peta
            m_cluster_2.get_root().html.add_child(folium.Element(title_html_cluster_2))

            # Render peta ke dalam objek HTML dan tampilkan di Streamlit
            map_html_cluster_2 = m_cluster_2._repr_html_()
            st.components.v1.html(map_html_cluster_2, width=700, height=300)

        
            # Pemetaan Masing-masing Cluster
            st.write("3. Pemetaan Jumlah Anak Pada Cluster 3")

            # Menghitung total jumlah anak di seluruh desa pada cluster 3
            total_anak_cluster_3 = data_koordinat['3'].sum()

            # Menentukan lokasi peta rata-rata
            average_lat_cluster_3 = data_koordinat['polygons'].apply(lambda x: parse_polygon(x)[0][0]).mean()
            average_lon_cluster_3 = data_koordinat['polygons'].apply(lambda x: parse_polygon(x)[0][1]).mean()

            m_cluster_3 = folium.Map(location=[average_lat_cluster_3, average_lon_cluster_3], zoom_start=13)

            # Normalisasi jumlah anak untuk intensitas warna
            max_jumlah_anak_cluster_3 = data_koordinat['3'].max()
            min_jumlah_anak_cluster_3 = data_koordinat['3'].min()

            for _, row in data_koordinat.iterrows():
                polygon_points_cluster_3 = parse_polygon(row['polygons'])
                jumlah_anak_cluster_3 = row['3']
                
                # Mengatur intensitas warna berdasarkan jumlah anak
                opacity_cluster_3 = 0.2 + 0.5 * (jumlah_anak_cluster_3 - min_jumlah_anak_cluster_3) / (max_jumlah_anak_cluster_3 - min_jumlah_anak_cluster_3)
                
                # Buat konten popup
                popup_content = f"<b>Desa/Kel:</b> {row['Desa/Kel']}<br><b>Jumlah Anak:</b> {jumlah_anak_cluster_3}"
                
                folium.Polygon(
                    locations=polygon_points_cluster_3,
                    color='orange',  # Outline color set to match fill color
                    fill=True,
                    fill_color='orange',  # Warna dasar, intensitas diatur oleh opacity
                    fill_opacity=opacity_cluster_3,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(m_cluster_3)

            # Menambahkan informasi total jumlah anak di pojok kanan atas menggunakan CSS
            title_html_cluster_3 = f'''
                <div style="position: fixed; 
                            top: 10px; right: 10px; width: 200px; height: 30px; 
                            background-color: white; z-index:9999; 
                            font-size: 14px; font-weight: bold; color: orange;
                            border:2px solid orange; padding: 4px;">
                    Total Anak Cluster 3: {total_anak_cluster_3}
                </div>
                '''

            # Tambahkan HTML ke peta
            m_cluster_3.get_root().html.add_child(folium.Element(title_html_cluster_3))

            # Render peta ke dalam objek HTML dan tampilkan di Streamlit
            map_html_cluster_3 = m_cluster_3._repr_html_()
            st.components.v1.html(map_html_cluster_3, width=700, height=300)


            # Pemetaan Masing-masing Cluster
            st.write("4. Pemetaan Jumlah Anak Pada Cluster 4")

            # Menghitung total jumlah anak di seluruh desa pada cluster 4
            total_anak_cluster_4 = data_koordinat['4'].sum()

            # Menentukan lokasi peta rata-rata
            average_lat_cluster_4 = data_koordinat['polygons'].apply(lambda x: parse_polygon(x)[0][0]).mean()
            average_lon_cluster_4 = data_koordinat['polygons'].apply(lambda x: parse_polygon(x)[0][1]).mean()

            m_cluster_4 = folium.Map(location=[average_lat_cluster_4, average_lon_cluster_4], zoom_start=14)

            # Normalisasi jumlah anak untuk intensitas warna
            max_jumlah_anak_cluster_4 = data_koordinat['4'].max()
            min_jumlah_anak_cluster_4 = data_koordinat['4'].min()

            for _, row in data_koordinat.iterrows():
                polygon_points_cluster_4 = parse_polygon(row['polygons'])
                jumlah_anak_cluster_4 = row['4']
                
                # Mengatur intensitas warna berdasarkan jumlah anak
                opacity_cluster_4 = 0.2 + 0.5 * (jumlah_anak_cluster_4 - min_jumlah_anak_cluster_4) / (max_jumlah_anak_cluster_4 - min_jumlah_anak_cluster_4)
                
                # Buat konten popup
                popup_content = f"<b>Desa/Kel:</b> {row['Desa/Kel']}<br><b>Jumlah Anak:</b> {jumlah_anak_cluster_4}"
                
                folium.Polygon(
                    locations=polygon_points_cluster_4,
                    color='red',  # Outline color set to match fill color
                    fill=True,
                    fill_color='red',  # Warna dasar, intensitas diatur oleh opacity
                    fill_opacity=opacity_cluster_4,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(m_cluster_4)

            # Menambahkan informasi total jumlah anak di pojok kanan atas menggunakan CSS
            title_html_cluster_4 = f'''
                <div style="position: fixed; 
                            top: 10px; right: 10px; width: 200px; height: 30px; 
                            background-color: white; z-index:9999; 
                            font-size: 14px; font-weight: bold; color: red;
                            border:2px solid red; padding: 4px;">
                    Total Anak Cluster 4: {total_anak_cluster_4}
                </div>
                '''

            # Tambahkan HTML ke peta
            m_cluster_4.get_root().html.add_child(folium.Element(title_html_cluster_4))

            # Render peta ke dalam objek HTML dan tampilkan di Streamlit
            map_html_cluster_4 = m_cluster_4._repr_html_()
            st.components.v1.html(map_html_cluster_4, width=700, height=300)




    with subTab2:
        subtab1, subtab2, subtab3, subtab4, subtab5, subtab6, subtab7= st.tabs(['Dataset', "Preprocessing", "K-Means++", "Hasil Akhir K-Means++", "Evalusi Cluster", "Analisis Cluster", "Hasil Pemetaan Pengelompokkan Daerah"])

        with subtab1:
            # Persiapan Dataset
            data_jumlah_cluster = pd.concat([data_relevan['Desa/Kel'], hasil['Cluster']], axis=1)

            # Mengelompokkan data berdasarkan 'Desa/Kel' dan 'Cluster' untuk menghitung jumlah cluster di setiap desa/kel
            cluster_counts = data_jumlah_cluster.groupby(['Desa/Kel', 'Cluster']).size().unstack(fill_value=0)

            # Reset indeks untuk meratakan DataFrame (tidak ada indeks multi-level)
            cluster_counts = cluster_counts.reset_index()

            # Menampilkan Hasil data jumlah masing-masing cluster PMT dalam satu Desa/Kel
            st.write("Data banyaknya jumlah setiap cluster dalam satu Desa/Kel")
            st.dataframe(cluster_counts)

            # Menghitung Persentase Stunting berdasarkan data daftar status gizi balita dengan variabel TB/U dikategorikan "Pendek" dan "Sangat Pendek"
            # 1: Hitung jumlah anak dan jumlah anak stunting (Pendek dan Sangat Pendek) per Desa
            total_count = data_input.groupby('Desa/Kel').size()
            stunting_count = data_input[data_input['TB/U'].isin(['Pendek', 'Sangat Pendek'])].groupby('Desa/Kel').size()

            # 2: Hitung Persentase Stunting
            stunting_percentage = (stunting_count / total_count) * 100

            # Masukkan keseluruhan anak, jumlah anak Pendek dan Sangat Pendek serta persentase stunting dalam satu dataframe
            stunting_summary = pd.DataFrame({
                'Total Anak': total_count,
                'Jumlah Pendek/Sangat Pende': stunting_count,
                'Persentase Stunting': stunting_percentage
            }).fillna(0)

            # Tampilkan Hasil dataframe
            stunting_summary.reset_index(inplace=True)
            st.write('Data Presentase Stunting')
            st.dataframe(stunting_summary)

            # Gabungkan data jumlah masing-masing cluster dengan data persentase stunting
            data_daerah = pd.concat([cluster_counts,stunting_summary['Persentase Stunting']], axis=1)

            # tampilkan data hasil penggabungan
            st.write('Gabungan Ke dua data untuk proses klusterisasi daerah rawan stunting dan gizi buruk')
            st.dataframe(data_daerah)
        
        with subtab2:
            # Normalisasi data
            scaler = MinMaxScaler()
            scaled_daerah = scaler.fit_transform(data_daerah.drop(columns=['Desa/Kel']))
            features_names_daerah = data_daerah.drop(columns=['Desa/Kel']).columns.copy()
            #features_names.remove('label')
            scaled_features_daerah = pd.DataFrame(scaled_daerah, columns=features_names_daerah)
            st.write('Data Hasil Normalisasi')
            st.dataframe(scaled_features_daerah)

        with subtab3:
            # Eksekusi metode pengelompokkan
            data_for_clustering_daerah = scaled_features_daerah.values.tolist()
            st.dataframe(scaled_features_daerah)
            features_daerah = scaled_features_daerah.columns.copy()
            k_daerah = 3
            final_cluster_data_daerah, final_centroids_daerah, computation_time_daerah, iteration_count_daerah, sse_daerah = kmeans_fix(data_for_clustering_daerah, features_daerah, k_daerah, initial_centroids_daerah)

        with subtab4:
            st.write(f"Jumlah iterasi hingga konvergen: {iteration_count_daerah}")
            st.write(f"Waktu komputasi yang dibutuhkan: {computation_time_daerah:.4f} detik")
            hasil_daerah = final_cluster_data_daerah.drop(columns=['C1','C2','C3'])
            st.dataframe(hasil_daerah)
            
            # Misalnya data Anda berbentuk numpy array atau dataframe
            X_daerah = hasil_daerah.drop(columns=['Cluster'])
            y_daerah = hasil_daerah['Cluster'].values.astype(int)  # Cluster dalam bentuk angka atau label
            centroids_daerah = final_centroids_daerah

            # Mapping angka kluster ke nama kluster (jika diperlukan)
            cluster_names_daerah = {1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3'}  # Sesuaikan dengan kluster Anda

            # Menggunakan PCA untuk mereduksi dari 5 dimensi ke 2 dimensi
            pca_daerah = PCA(n_components=2)
            X_pca_daerah = pca_daerah.fit_transform(X_daerah)

            # Transformasi centroid dari 5D ke 2D menggunakan PCA yang sudah di-fit
            centroids_pca_daerah = pca_daerah.transform(centroids_daerah)

            # Plot hasil PCA dari data dan centroid
            plt.figure(figsize=(4, 4))

            # Plot setiap cluster dengan label dinamis
            for cluster in np.unique(y_daerah):
                plt.scatter(X_pca_daerah[y_daerah == cluster, 0], X_pca_daerah[y_daerah == cluster, 1], 
                            label=cluster_names_daerah.get(cluster, f'Cluster {cluster}'), s=50)

            # Plot centroid
            plt.scatter(centroids_pca_daerah[:, 0], centroids_pca_daerah[:, 1], c='black', s=200, alpha=0.75, marker='x', label='Centroids')

            # Tambahkan nilai centroid ke plot
            for i, (x_daerah, y_daerah) in enumerate(centroids_pca_daerah):
                plt.text(x_daerah, y_daerah, f'({x_daerah:.2f}, {y_daerah:.2f})', fontsize=9, ha='right', color='black')

            plt.title("Plot Hasil Pengelompokkan Daerah")
            plt.xlabel("Data")
            plt.ylabel("Cluster")
            plt.legend()
            plt.show()
            col1, col2, col3 = st.columns([3, 1, 1])  
            with col1:
                st.pyplot(plt)
        
        with subtab5:
            # Menampilkan Hasil Evaluasi Cluster
            # 1. Menghitung Sum Square Error (SSE)
            st.write(f"Sum of Squared Errors (SSE): {sse_daerah}")
            
            # 2. Menghitung Davies-Bouldin Index (DBI)
            # Jumlah cluster
            n_clusters_daerah = k_daerah  # jumlah klaster

            # Konversi dan centroids_akhir ke float
            centroids_daerah = np.array(final_centroids_daerah, dtype=float)

            # Mengambil data hasil perhitungan jarak pada setiap centroid serta kolom cluster
            data_dbi_daerah = final_cluster_data_daerah.iloc[:, 5:]

            # Konversi labels ke integer
            labels_daerah = np.array(final_cluster_data_daerah['Cluster'], dtype=int)

            # # Hitung SSW
            ssw_daerah = calculate_ssw(data_dbi_daerah)

            # Hitung SSB
            ssb_daerah = calculate_ssb(centroids_daerah, n_clusters_daerah)

            # Hitung Rasio R_ij dan rasio maksimum
            rij_daerah, max_ratios_daerah = calculate_rij(ssw_daerah, ssb_daerah, n_clusters_daerah)

            # Hitung Davies-Bouldin Index
            dbi_value_daerah = dbi(max_ratios_daerah,n_clusters_daerah)

            st.write('Tahapan Perhitungan DBI :')
            st.write(f'Jumlah Cluster: {n_clusters_daerah}')

            # Tampilkan SSW
            st.write('SSW (Sum of Squared Within) untuk setiap cluster:')
            st.dataframe(ssw_daerah)

            # Tampilkan SSB sebagai matriks
            st.write('SSB (Sum of Squared Between) antar centroid:')
            st.dataframe(pd.DataFrame(ssb_daerah, columns=[f'Cluster {i+1}' for i in range(n_clusters_daerah)], index=[f'Cluster {i+1}' for i in range(n_clusters_daerah)]))

            # Tampilkan Rasio (SSW + SSW) / SSB sebagai matriks
            st.write('Rasio (SSW + SSW) / SSB antar cluster:')
            st.dataframe(pd.DataFrame(rij_daerah, columns=[f'Cluster {i+1}' for i in range(n_clusters_daerah)], index=[f'Cluster {i+1}' for i in range(n_clusters_daerah)]))

            # Tampilkan rasio maksimum dan DBI
            st.write(f'Rasio maksimum untuk setiap cluster: {max_ratios_daerah}')
            st.write(f'Davies-Bouldin Index: {dbi_value_daerah}')
        
        with subtab6:
            # Menampilkan hasil analisis cluster
            # Gabungkan data sebelum di normalisasi dengan cluster untuk dilakukan analisis
            data_gabung_daerah= pd.concat([data_daerah.drop(columns=['Desa/Kel']),hasil_daerah['Cluster']],axis=1)

            # 1. Analisis cluster dengan nilai mean dan standar deviasi
            cluster_stats = data_gabung_daerah.groupby('Cluster').agg(['mean', 'std'])
            st.write("Mean dan Standar Deviasi per Cluster:")
            st.write(cluster_stats)

            # 2. Analisis cluster dengan uji korelasi antar data pada masing-masing cluster
            st.write("Uji Korelasi masing-masing cluster:")

            # Ambil jumlah cluster
            num_clusters = data_gabung_daerah['Cluster'].nunique()

            # Hitung jumlah baris dan kolom untuk grid layout
            cols = 1  # Misal ingin 3 heatmap per baris
            rows = math.ceil(num_clusters / cols)  # Hitung jumlah baris yang dibutuhkan

            # Buat figure dengan subplots
            fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))  # Lebar dan tinggi dari figure
            axes = axes.flatten()  # Flatten agar bisa digunakan untuk indexing

            # Looping melalui setiap cluster
            for idx, cluster in enumerate(data_gabung_daerah['Cluster'].unique()):
                # Mengambil data berdasarkan cluster
                cluster_data = data_gabung_daerah[data_gabung_daerah['Cluster'] == cluster][features_names_daerah]

                # Menghitung matriks korelasi
                correlation_matrix = cluster_data.corr()

                # Menampilkan heatmap pada subplot yang sesuai
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=axes[idx], annot_kws={"size": 16})

                # Set judul pada setiap heatmap
                axes[idx].set_title(f"Heatmap Korelasi Cluster {cluster}")

            # Hapus subplot kosong jika jumlah cluster kurang dari grid
            for ax in axes[num_clusters:]:
                fig.delaxes(ax)

            # Atur layout agar tidak tumpang tindih
            plt.tight_layout()
            st.pyplot(fig)

        with subtab7:
            # Pemetaan Hasil Pengelompokkan Daerah
            st.header("Pemetaan Hasil Pengelompokkan Daerah")
            st.write("A. Data Hasil Pengelompokkan Daerah")
            data_sig = pd.concat([data_daerah, hasil_daerah['Cluster']], axis=1)
            st.dataframe(data_sig)

            st.write("B. Penggabungan Data Hasil Pengelompokkan Daerah dengan Koordinat Wilayah")
            koor = pd.read_csv("https://raw.githubusercontent.com/HanifSantoso05/Dataset_Skripsi/main/koordinat_wilayah.csv")
            data_koordinat_daerah = pd.concat([data_sig, koor['polygons']], axis=1)
            st.dataframe(data_koordinat_daerah)

            st.write("C. Peta Hasil Pengelompokkan Daerah Rawan Gizi Buruk dan Stunting")
            
            def get_cluster_color(cluster):
                if pilih == "2019":
                    if cluster == "1":
                        return 'orange'
                    elif cluster == "2":
                        return 'red'
                    elif cluster == "3":
                        return 'green'
                    return 'grey' 
                elif pilih == "2020":
                    if cluster == "1":
                        return 'orange'
                    elif cluster == "2":
                        return 'green'
                    elif cluster == "3":
                        return 'red'
                    return 'grey'
                elif pilih == "2021":
                    if cluster == "1":
                        return 'orange'
                    elif cluster == "2":
                        return 'green'
                    elif cluster == "3":
                        return 'red'
                    return 'grey'
                elif pilih == "2022":
                    if cluster == "1":
                        return 'green'
                    elif cluster == "2":
                        return 'red'
                    elif cluster == "3":
                        return 'orange'
                    return 'grey'
                elif pilih == "2023":
                    if cluster == "1":
                        return 'green'
                    elif cluster == "2":
                        return 'orange'
                    elif cluster == "3":
                        return 'red'
                    return 'grey'

            # Membuat peta folium berpusat pada lokasi rata-rata dari data yang dipilih
            average_lat_daerah = data_koordinat_daerah['polygons'].apply(lambda x: parse_polygon(x)[0][0]).mean()
            average_lon_daerah = data_koordinat_daerah['polygons'].apply(lambda x: parse_polygon(x)[0][1]).mean()

            m_daerah = folium.Map(location=[average_lat_daerah, average_lon_daerah], zoom_start=12)

            # Tambahkan polygon untuk setiap kelurahan yang terpilih dengan warna berdasarkan cluster
            for _, row in data_koordinat_daerah.iterrows():
                polygon_points = parse_polygon(row['polygons'])
                popup_columns = data_sig.columns

                # Buat konten popup secara dinamis
                popup_content = ""
                for col in popup_columns:
                    popup_content += f"<b>{col}:</b> {row[col]}<br>"

                # Ambil warna sesuai dengan cluster
                color = get_cluster_color(row['Cluster'])                
                
                folium.Polygon(
                    locations=polygon_points,
                    color=color,  # Gunakan warna dari fungsi get_cluster_color
                    fill=True,
                    fill_opacity=0.4,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(m_daerah)

            # Render peta ke dalam objek HTML
            map_html_daerah = m_daerah._repr_html_()

            # Tampilkan peta di Streamlit dengan st.components.v1.html
            st.components.v1.html(map_html_daerah, width=700, height=500)
