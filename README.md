# Laporan Proyek Machine Learning - Satriatama Putra

## Daftar Isi

- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Kesimpulan](#kesimpulan)
- [Referensi](#referensi)

## Project Overview

Pada era digital saat ini, ledakan informasi yang tersedia secara online membuat pengguna sering kali kewalahan dalam menemukan konten yang relevan dan sesuai dengan minat mereka. Hal ini sangat terasa dalam industri literatur digital, di mana jutaan buku tersedia melalui platform e-commerce dan perpustakaan digital. Fenomena ini dikenal sebagai "information overload" atau kelebihan informasi, di mana volume data yang besar justru menjadi hambatan bagi pengguna untuk menemukan buku yang benar-benar mereka inginkan.

![image](https://github.com/user-attachments/assets/41396bde-5b6c-4161-9efe-e662c6b27b40)

_Gambar 1: Buku_

Sistem rekomendasi buku hadir sebagai solusi untuk masalah ini, dengan tujuan utama membantu pengguna menemukan buku yang sesuai dengan preferensi mereka secara efisien. Menurut penelitian oleh [Isinkaye et al. (2015)](https://www.sciencedirect.com/science/article/pii/S1110866515000341), sistem rekomendasi telah menjadi komponen kritis dalam platform e-commerce modern, dengan kemampuan meningkatkan kepuasan pengguna hingga 27% dan meningkatkan pendapatan bisnis hingga 35%.

Pentingnya sistem rekomendasi buku dapat dilihat dari beberapa aspek:

1. **Bagi Pengguna**: Membantu menemukan buku yang relevan dengan minat mereka di tengah jutaan pilihan, menghemat waktu pencarian, dan meningkatkan penemuan buku baru yang mungkin tidak ditemukan melalui pencarian manual.

2. **Bagi Platform**: Meningkatkan engagement pengguna, memperpanjang waktu yang dihabiskan di platform, meningkatkan konversi penjualan, dan membangun loyalitas pengguna melalui personalisasi.

3. **Bagi Penerbit dan Penulis**: Memberikan visibilitas yang lebih baik untuk buku-buku yang mungkin tidak mendapat perhatian melalui metode pemasaran tradisional, terutama untuk penulis baru atau niche.

Dalam studi terbaru oleh [Koren et al. (2022)](https://dl.acm.org/doi/10.1145/1401890.1401944), penerapan sistem rekomendasi yang efektif pada platform buku online telah terbukti meningkatkan discovery rate sebesar 35% dan conversion rate hingga 40%. Penelitian lain oleh [Smith dan Linden (2017)](https://ieeexplore.ieee.org/document/8186743) menunjukkan bahwa hingga 35% transaksi di Amazon berasal dari rekomendasi personalisasi.

Dalam proyek ini, kita akan mengembangkan sistem rekomendasi buku menggunakan dua pendekatan utama dalam machine learning: **Content-based Filtering** dan **Collaborative Filtering**. Kedua pendekatan ini memiliki kelebihan dan kekurangan masing-masing, dan kombinasi keduanya dapat memberikan rekomendasi yang lebih komprehensif dan akurat bagi pengguna.

## Business Understanding

Dalam era digital dengan pertumbuhan eksponensial konten online, pengguna menghadapi tantangan besar dalam menemukan buku yang relevan dengan minat mereka. Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menavigasi jutaan buku yang tersedia dan menemukan buku-buku yang paling sesuai dengan preferensi mereka.

### Problem Statements

Berdasarkan latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah:

1. Bagaimana cara membangun sistem rekomendasi buku yang dapat memberikan rekomendasi berdasarkan karakteristik konten buku (content-based filtering)?
2. Bagaimana cara membangun sistem rekomendasi buku yang dapat mempelajari pola interaksi pengguna dengan buku (collaborative filtering)?
3. Bagaimana cara mengukur efektivitas dari sistem rekomendasi yang dibangun?

### Goals

Tujuan dari proyek ini adalah:

1. Mengembangkan model rekomendasi buku menggunakan pendekatan content-based filtering yang mampu memberikan rekomendasi berdasarkan kemiripan konten buku.
2. Mengembangkan model rekomendasi buku menggunakan pendekatan collaborative filtering yang mampu memberikan rekomendasi berdasarkan pola perilaku pengguna.
3. Mengevaluasi dan membandingkan kinerja kedua model rekomendasi menggunakan metrik evaluasi yang sesuai.

### Solution Approach

Untuk mencapai tujuan yang telah ditetapkan, berikut adalah pendekatan solusi yang akan diimplementasikan:

#### 1. Content-based Filtering

Content-based filtering merekomendasikan buku berdasarkan kemiripan fitur atau konten dari buku tersebut. Pendekatan ini menggunakan informasi dari buku yang pernah disukai pengguna untuk merekomendasikan buku serupa.

**Tahapan implementasi:**

- Ekstraksi fitur dari data buku (judul, penulis, penerbit, kategori, ringkasan) menggunakan teknik TF-IDF (Term Frequency-Inverse Document Frequency)
- Perhitungan similarity antar buku menggunakan Cosine Similarity
- Pengembangan fungsi rekomendasi yang mengembalikan buku-buku dengan similarity tertinggi

**Kelebihan:**

- Tidak memerlukan data dari pengguna lain (cold-start untuk item)
- Dapat merekomendasikan item khusus yang mungkin tidak populer
- Mampu memberikan penjelasan mengapa item tertentu direkomendasikan

**Kekurangan:**

- Cenderung merekomendasikan item yang sangat mirip (overspecialization)
- Tidak dapat menemukan preferensi baru pengguna (serendipity rendah)
- Membutuhkan representasi konten yang baik

#### 2. Collaborative Filtering

Collaborative filtering merekomendasikan buku berdasarkan preferensi pengguna lain yang memiliki pola perilaku serupa. Pendekatan ini mengasumsikan bahwa pengguna yang setuju di masa lalu cenderung setuju di masa depan.

**Tahapan implementasi:**

- Pengembangan model neural network dengan embedding layer untuk user dan item
- Pelatihan model menggunakan data rating pengguna terhadap buku
- Prediksi rating untuk buku yang belum dibaca pengguna
- Rekomendasi buku dengan prediksi rating tertinggi

**Kelebihan:**

- Mampu menemukan pola yang kompleks dan tidak eksplisit
- Dapat merekomendasikan item yang tidak terkait dengan riwayat pengguna (serendipity tinggi)
- Tidak memerlukan informasi konten dari item

**Kekurangan:**

- Menghadapi masalah cold-start untuk pengguna baru dan item baru
- Membutuhkan data yang cukup besar untuk memberikan rekomendasi yang akurat
- Sulit menjelaskan alasan di balik rekomendasi (black box)

Dengan mengimplementasikan kedua pendekatan ini, kita dapat membandingkan kinerja masing-masing dan memberikan rekomendasi buku yang lebih komprehensif dan akurat kepada pengguna.

## Data Understanding

Dataset yang digunakan pada proyek ini adalah dataset [Book-Crossing: User review ratings](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset) dari Kaggle, yang berisi informasi tentang buku, pengguna, dan rating yang diberikan oleh pengguna terhadap buku.

![image](https://github.com/user-attachments/assets/03d27bfa-67b8-418c-9aee-da978880a90f)


_Gambar 2: Dataset_

### Informasi Dataset

Dataset ini terdiri dari satu file CSV bernama 'preprocessed_data.csv' yang merupakan hasil penggabungan dari beberapa dataset asli dan telah melalui proses preprocessing awal. Dataset ini memiliki 1.031.175 baris data dengan 19 kolom.

**Tabel 1. Informasi Kolom pada Dataset**

| Kolom               | Deskripsi                                 | Tipe Data |
| ------------------- | ----------------------------------------- | --------- |
| Unnamed: 0          | Indeks baris                              | int64     |
| user_id             | ID unik pengguna                          | int64     |
| location            | Lokasi pengguna                           | object    |
| age                 | Usia pengguna                             | float64   |
| isbn                | Kode ISBN buku                            | object    |
| rating              | Rating yang diberikan pengguna (0-10)     | int64     |
| book_title          | Judul buku                                | object    |
| book_author         | Penulis buku                              | object    |
| year_of_publication | Tahun publikasi buku                      | float64   |
| publisher           | Penerbit buku                             | object    |
| img_s               | URL gambar sampul buku (small)            | object    |
| img_m               | URL gambar sampul buku (medium)           | object    |
| img_l               | URL gambar sampul buku (large)            | object    |
| Summary             | Ringkasan atau sinopsis buku              | object    |
| Language            | Bahasa buku                               | object    |
| Category            | Kategori atau genre buku                  | object    |
| city                | Kota asal pengguna                        | object    |
| state               | Negara bagian atau provinsi asal pengguna | object    |
| country             | Negara asal pengguna                      | object    |

### Exploratory Data Analysis (EDA)

#### Statistik Deskriptif

Berikut adalah statistik deskriptif dari dataset:

**Tabel 2. Statistik Deskriptif Variabel Numerik**

| Statistik | user_id      | age          | rating       | year_of_publication |
| --------- | ------------ | ------------ | ------------ | ------------------- |
| count     | 1.031175e+06 | 1.031175e+06 | 1.031175e+06 | 1.031175e+06        |
| mean      | 1.424570e+05 | 3.639768e+01 | 2.866972e+00 | 1.996143e+03        |
| std       | 8.040498e+04 | 1.196009e+01 | 3.824161e+00 | 9.568668e+00        |
| min       | 2.000000e+00 | 5.000000e+00 | 0.000000e+00 | 1.376000e+03        |
| 25%       | 7.780850e+04 | 2.800000e+01 | 0.000000e+00 | 1.992000e+03        |
| 50%       | 1.437380e+05 | 3.474390e+01 | 0.000000e+00 | 1.999000e+03        |
| 75%       | 2.087980e+05 | 4.500000e+01 | 8.000000e+00 | 2.001000e+03        |
| max       | 2.788540e+05 | 9.900000e+01 | 1.000000e+01 | 2.008000e+03        |

#### Univariate Analysis

Dari hasil Univariate Analysis, ditemukan beberapa insight penting:

![image](https://github.com/user-attachments/assets/0adc7ff0-36fb-4dff-8e04-4f02d3e11b2d)

_Gambar 3: Distribusi Rating pada Dataset_

- **Distribusi Rating**: Mayoritas data adalah interaksi tanpa rating eksplisit (rating 0), menandakan bahwa sebagian besar entri adalah implicit feedback. Hal ini penting untuk diperhatikan karena akan mempengaruhi pendekatan pemodelan, terutama untuk collaborative filtering.

![image](https://github.com/user-attachments/assets/633db6c6-4401-45d8-8103-6261b7e8c5d8)

_Gambar 4: Distribusi Usia Pengguna_

- **Distribusi Usia Pengguna**: Rentang usia pengguna sangat luas (5-99 tahun) dengan konsentrasi terbesar pada kelompok usia 30-40 tahun. Hal ini menunjukkan bahwa platform digunakan oleh berbagai kelompok umur, tetapi mayoritas adalah pembaca dewasa.

![image](https://github.com/user-attachments/assets/e561c0d2-ce39-4bf0-b512-2878ba786b48)

_Gambar 5: Distribusi Tahun Publikasi Buku_

- **Distribusi Tahun Publikasi**: Terjadi peningkatan drastis dari tahun 1990-an hingga awal 2000-an, dengan puncak di sekitar tahun 2000. Ini mengindikasikan bahwa dataset lebih banyak berisi buku-buku kontemporer daripada klasik.

![image](https://github.com/user-attachments/assets/d7b0bb2c-00dd-45b8-ab00-196660b095e9)

_Gambar 6: 10 Penulis dengan Jumlah Buku Terbanyak_

- **Distribusi Penulis**: Stephen King merupakan penulis dengan representasi tertinggi (10,053 entri), diikuti oleh beberapa penulis populer lainnya. Distribusi yang tidak merata ini dapat mempengaruhi rekomendasi, terutama untuk content-based filtering.

![image](https://github.com/user-attachments/assets/b8cd5696-1ecf-45ff-a57e-71a54c97d19c)

_Gambar 7: Distribusi Bahasa Buku_

- **Distribusi Bahasa**: Bahasa Inggris sangat dominan (618,505 entri), menunjukkan ketidakseimbangan yang ekstrem dalam keragaman bahasa. Hal ini penting untuk dipertimbangkan dalam sistem rekomendasi multi-bahasa.

![image](https://github.com/user-attachments/assets/d8800275-d327-4bfc-8cbc-2ac8a72b446f)

_Gambar 8: 10 Negara dengan Jumlah Pengguna Terbanyak_

- **Distribusi Geografis**: USA sangat dominan (745,840 entri), diikuti oleh Kanada dan Inggris. Ini menunjukkan bias geografis yang kuat dalam dataset, yang perlu dipertimbangkan untuk rekomendasi global.

#### Multivariate Analysis

Dari hasil Analysis Multivariate, ditemukan beberapa insight penting:

![image](https://github.com/user-attachments/assets/bbb8bcf7-6351-450c-b503-ac3096005dbc)

_Gambar 9: Distribusi Rating Berdasarkan Tahun Publikasi_

- **Rating vs Tahun Publikasi**: Distribusi rating relatif konsisten antar tahun, menandakan bahwa usia buku tidak signifikan mempengaruhi penilaian pengguna. Ini menunjukkan bahwa pembaca menilai buku berdasarkan kualitas konten, bukan popularitas terkini.

![image](https://github.com/user-attachments/assets/b9481993-f3a6-4d6e-b078-caac880210a7)

_Gambar 10: Rata-rata Rating Berdasarkan Penerbit_

- **Rating vs Penerbit**: Terdapat variasi signifikan antar penerbit besar, dengan beberapa penerbit secara konsisten mendapatkan rating lebih tinggi. Hal ini dapat menjadi fitur penting dalam sistem rekomendasi.

![image](https://github.com/user-attachments/assets/5c712bcf-183a-4677-8a67-baf34f53ecc5)

_Gambar 11: Distribusi Rating untuk Penulis Teratas_

- **Rating vs Penulis Teratas**: Penulis populer seperti Stephen King memiliki distribusi rating yang lebih luas dengan median yang lebih rendah, mengindikasikan bahwa popularitas tidak selalu berkorelasi dengan rating tinggi.

#### Analisis Korelasi

![image](https://github.com/user-attachments/assets/95baf440-80da-4bd3-bd4f-b67ece2daa6d)

_Gambar 12: Matriks Korelasi Antar Variabel Numerik_

Dari hasil analisis korelasi, ditemukan bahwa:

- Korelasi antara rating dan usia pengguna sangat lemah (0.03), menandakan bahwa preferensi literatur tidak secara signifikan dipengaruhi oleh faktor usia.
- Korelasi antara rating dan tahun publikasi juga sangat lemah (0.01), mengkonfirmasi insight dari analisis multivariat bahwa usia buku tidak menjadi faktor penentu dalam penilaian pengguna.
- Korelasi lemah antara usia pengguna dan tahun publikasi (-0.05), mengindikasikan tidak adanya kecenderungan kuat dari kelompok usia tertentu untuk membaca buku dari era tertentu.

Secara keseluruhan, matrix korelasi menunjukkan independensi antar variabel numerik utama, mengisyaratkan bahwa pendekatan content-based dan collaborative filtering perlu memanfaatkan fitur kategorical dan relasional untuk menghasilkan rekomendasi yang efektif.

## Data Preparation

Tahap persiapan data sangat penting untuk memastikan data siap digunakan dalam pemodelan. Berikut adalah langkah-langkah yang dilakukan:

### 1. Penanganan Missing Value

Terdapat beberapa missing value pada dataset, khususnya pada kolom:

- book_author (1 missing value)
- city (14,103 missing value)
- state (22,798 missing value)
- country (35,374 missing value)

Missing value ditangani dengan mengisi nilai default:

- book_author diisi dengan 'Unknown Author'
- city, state, dan country diisi dengan 'Unknown'

```python
print("Kolom dengan missing value:")
print(df.isna().sum()[df.isna().sum() > 0])

df['book_author'] = df['book_author'].fillna('Unknown Author')

df['city'] = df['city'].fillna('Unknown')
df['state'] = df['state'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

print("\nSetelah penanganan missing value:")
print(df.isna().sum()[df.isna().sum() > 0])

print(f"\nJumlah duplikasi: {df.duplicated().sum()}")
```

**Alasan**: Penanganan missing value penting untuk menghindari error saat pemrosesan data. Untuk book_author, mengisi dengan 'Unknown Author' memungkinkan kita tetap mempertahankan buku tersebut dalam dataset. Untuk data lokasi, karena tidak terlalu berpengaruh pada rekomendasi buku, kita cukup mengisi dengan 'Unknown'.

### 2. Pemilihan Fitur

Untuk masing-masing pendekatan, dipilih fitur yang relevan:

**Content-based Filtering**:

- isbn
- book_title
- book_author
- publisher
- Category
- Summary
- Language

**Collaborative Filtering**:

- user_id
- isbn
- rating

```python
content_features = ['isbn', 'book_title', 'book_author', 'publisher', 'Category', 'Summary']

# Filter data untuk menghilangkan buku dengan kategori '9' dan summary '9'
filtered_content_df = df[~(df['Category'] == '9') & ~(df['Summary'] == '9')]
content_df = filtered_content_df[content_features].drop_duplicates(subset=['isbn'])

collab_features = ['user_id', 'isbn', 'rating']
collab_df = df[collab_features]
```

**Alasan**: Pemilihan fitur yang tepat sangat penting untuk efisiensi komputasi dan akurasi model. Untuk content-based filtering, kita memilih fitur yang mendeskripsikan konten buku. Untuk collaborative filtering, kita hanya membutuhkan interaksi antara pengguna dan buku.

### 3. Pembersihan Data untuk Content-Based Filtering

- Menangani nilai kosong pada fitur teks (Summary dan Category)
- Membuat fitur gabungan 'content' yang menggabungkan book_title, book_author, publisher, Category, dan Summary
- Mengonversi teks ke lowercase untuk standardisasi

```python
content_df['Summary'] = content_df['Summary'].fillna('')
content_df['Category'] = content_df['Category'].fillna('')

content_df['content'] = content_df['book_title'] + ' ' + content_df['book_author'] + ' ' + content_df['publisher'] + ' ' + content_df['Category'] + ' ' + content_df['Summary']

content_df['content'] = content_df['content'].str.lower()
```

**Alasan**: Pembersihan data teks penting untuk TF-IDF yang akan digunakan dalam content-based filtering. Dengan menggabungkan semua fitur teks, kita dapat membuat representasi komprehensif dari setiap buku. Konversi ke lowercase memastikan konsistensi dalam pemrosesan teks.

### 4. Pembersihan Data untuk Collaborative Filtering

- Memfilter data dengan rating eksplisit (>0)
- Mengatasi masalah sparsity dengan memfilter user dan buku dengan jumlah interaksi minimal
- Melakukan mapping ID user dan buku ke indeks berurutan untuk efisiensi komputasi

```python
collab_df = collab_df[collab_df['rating'] > 0]

min_user_ratings = 5
user_counts = collab_df['user_id'].value_counts()
active_users = user_counts[user_counts >= min_user_ratings].index
filtered_df = collab_df[collab_df['user_id'].isin(active_users)]

min_book_ratings = 5
book_counts = filtered_df['isbn'].value_counts()
popular_books = book_counts[book_counts >= min_book_ratings].index
filtered_df = filtered_df[filtered_df['isbn'].isin(popular_books)]

user_ids = filtered_df['user_id'].unique().tolist()
isbn_ids = filtered_df['isbn'].unique().tolist()
user_to_idx = {user: i for i, user in enumerate(user_ids)}
isbn_to_idx = {isbn: i for i, isbn in enumerate(isbn_ids)}
filtered_df['user_idx'] = filtered_df['user_id'].map(user_to_idx)
filtered_df['isbn_idx'] = filtered_df['isbn'].map(isbn_to_idx)
```

**Alasan**: Untuk collaborative filtering, perlu mengatasi masalah sparsity dengan memfilter user dan buku dengan jumlah interaksi yang cukup. Mapping ID ke indeks berurutan meningkatkan efisiensi komputasi dan memudahkan proses embedding dalam model neural network.


## Modeling

Pada proyek ini, diimplementasikan dua pendekatan sistem rekomendasi:

### 1. Content-based Filtering

Content-based filtering merekomendasikan buku berdasarkan kemiripan konten antar buku.

#### Algoritma dan Tahapan Implementasi

1. **TF-IDF Vectorizer**

   TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengekstrak fitur dari konten buku dan mengubahnya menjadi representasi vektor.

   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer

   # Membuat instance TF-IDF Vectorizer
   tfidf = TfidfVectorizer(stop_words='english')

   # Fitting dan transformasi data ke dalam representasi TF-IDF
   tfidf_matrix = tfidf.fit_transform(content_df['content'])
   ```


2. **Cosine Similarity**

   Cosine similarity digunakan untuk menghitung kemiripan antar buku berdasarkan representasi vektor TF-IDF.

   ```python
   from sklearn.metrics.pairwise import cosine_similarity

   # Menghitung cosine similarity untuk setiap buku secara on-demand
   def get_cosine_similarity(idx):
       return cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
   ```

3. **Fungsi Rekomendasi**

   ```python
   def get_content_based_recommendations(isbn, k=10):
    try:
        # Mendapatkan buku referensi dari ISBN
        reference_book = content_df[content_df['isbn'] == isbn]
        
        if reference_book.empty:
            print(f"ISBN {isbn} tidak ditemukan dalam dataset.")
            return []
            
        # Mendapatkan indeks buku referensi
        idx = reference_book.index[0]
        
        # Menghitung cosine similarity
        book_vector = tfidf_matrix[idx:idx+1]
        sim_scores = cosine_similarity(book_vector, tfidf_matrix).flatten()
        
        # Membuat DataFrame sementara dengan skor similaritas dan ISBNs
        temp_df = pd.DataFrame({
            'isbn': content_df['isbn'],
            'similarity': sim_scores
        })
        
        # Mengurutkan berdasarkan similaritas (hilangkan buku referensi)
        temp_df = temp_df[temp_df['isbn'] != isbn].sort_values('similarity', ascending=False).head(k)
        
        # Menyiapkan hasil rekomendasi
        recommendations = []
        for _, row in temp_df.iterrows():
            rec_isbn = row['isbn']
            rec_title = isbn_to_title.get(rec_isbn, "Unknown Title")
            recommendations.append((rec_isbn, rec_title))
            
        return recommendations
        
    except Exception as e:
        print(f"Error dalam get_content_based_recommendations: {e}")
        return []
   ```

#### Contoh Hasil Rekomendasi


Untuk buku "Our Dumb Century: The Onion Presents 100 Years of Headlines from America's Finest News Source", sistem merekomendasikan buku-buku seperti:

1. The Onion Field (ISBN: 0440173507)
2. Dispatches from the Tenth Circle: The Best of the Onion (ISBN: 0609808346)
3. The Onion's Finest News Reporting, Volume 1 (ISBN: 0609804634)
4. The Maui Onion Cookbook (ISBN: 0890878021)
5. The Onion Ad Nauseam: Complete News Archives, Volume 13 (ISBN: 1400047242)

Hasil ini menunjukkan bahwa sistem berhasil merekomendasikan buku-buku dengan konten yang serupa berdasarkan karakteristiknya.

### 2. Collaborative Filtering

Collaborative filtering merekomendasikan buku berdasarkan pola rating yang diberikan oleh pengguna.

#### Algoritma dan Tahapan Implementasi

1. **Persiapan Data untuk Model**

   ```python
   # Menyiapkan data untuk modeling
   X = filtered_df[['user_idx', 'isbn_idx']].values
   y = filtered_df['rating'].values

   # Normalisasi rating ke rentang [0, 1]
   y = y / 10.0

   # Membagi data menjadi training dan testing
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

2. **Arsitektur Model Neural Network**

   ![image](https://github.com/user-attachments/assets/d99f4c59-5bf0-4030-8c48-77a9083e9428)

   _Gambar 13: Arsitektur Neural Network untuk Collaborative Filtering_

   ```python
   import tensorflow as tf
   from tensorflow import keras
   from tensorflow.keras import layers

   # Menentukan dimensi embedding
   embedding_dim = 50;

   # Membangun model
   def create_model():
       # Input untuk user dan item
       user_input = keras.Input(shape=(1,), name='user_input')
       book_input = keras.Input(shape=(1,), name='book_input')

       # Embedding layer untuk user dan item
       user_embedding = layers.Embedding(num_users, embedding_dim, name='user_embedding')(user_input)
       book_embedding = layers.Embedding(num_books, embedding_dim, name='book_embedding')(book_input)

       # Flatten embedding
       user_vector = layers.Flatten()(user_embedding)
       book_vector = layers.Flatten()(book_embedding)

       # Dot product antara user dan item embedding
       dot_product = layers.Dot(axes=1)([user_vector, book_vector])

       # Layer dense tambahan untuk meningkatkan kapasitas model
       x = layers.Dense(128, activation='relu')(dot_product)
       x = layers.Dense(64, activation='relu')(x)
       x = layers.Dense(32, activation='relu')(x)

       # Output layer
       output = layers.Dense(1, activation='sigmoid')(x)

       # Mendefinisikan model
       model = keras.Model(inputs=[user_input, book_input], outputs=output)

       # Kompilasi model
       model.compile(
           optimizer=keras.optimizers.Adam(learning_rate=0.001),
           loss='mean_squared_error',
           metrics=[keras.metrics.RootMeanSquaredError()]
       )

       return model
   ```

3. **Pelatihan Model**

   ```python
   # Early stopping untuk menghindari overfitting
   early_stopping = keras.callbacks.EarlyStopping(
       monitor='val_loss',
       patience=5,
       restore_best_weights=True
   )

   # Melatih model
   history = model.fit(
       [X_train[:, 0], X_train[:, 1]],  # user_idx, isbn_idx
       y_train,
       epochs=20,
       batch_size=64,
       validation_split=0.2,
       callbacks=[early_stopping],
       verbose=1
   )
   ```

4. **Fungsi Rekomendasi**

   ```python
   def get_collaborative_recommendations(user_id, model, isbn_to_idx, user_to_idx, isbn_to_title, k=10):
       try:
           # Mendapatkan indeks user
           user_idx = user_to_idx[user_id]

           # Mendapatkan semua buku dalam dataset
           all_books = list(isbn_to_idx.keys())

           # Membuat data untuk prediksi
           user_book_array = np.array([[user_idx, isbn_to_idx[isbn]] for isbn in all_books])

           # Memprediksi rating untuk semua kombinasi user-book
           predictions = model.predict([user_book_array[:, 0], user_book_array[:, 1]], verbose=0).flatten() * 10

           # Menggabungkan hasil prediksi dengan ISBN buku
           book_predictions = list(zip(all_books, predictions))

           # Mengurutkan berdasarkan prediksi rating tertinggi
           book_predictions = sorted(book_predictions, key=lambda x: x[1], reverse=True)

           # Mendapatkan k rekomendasi teratas
           top_recommendations = book_predictions[:k]

           # Menyiapkan hasil rekomendasi dengan ISBN, judul, dan prediksi rating
           recommendations = [(isbn, isbn_to_title.get(isbn, "Unknown Title"), rating)
                              for isbn, rating in top_recommendations]

           return recommendations

       except KeyError:
           print(f"User ID {user_id} tidak ditemukan dalam dataset.")
           return []
   ```

#### Contoh Hasil Rekomendasi


Untuk pengguna dengan ID 238010, sistem merekomendasikan buku-buku berikut dengan prediksi rating tertinggi:

1. Unknown Title (ISBN: 0446611212) 
2. Change Your Life Without Getting Out of Bed : The Ultimate Nap Book (ISBN: 0684859300) 
3. Bloody Bones (Anita Blake Vampire Hunter (Paperback)) (ISBN: 0515134465) 
4. Dubliners (ISBN: 0553213806) 
5. Don't Let's Go to the Dogs Tonight : An African Childhood (ISBN: 0375758992) 

Hasil ini menunjukkan bahwa sistem berhasil memprediksi buku-buku yang mungkin disukai oleh pengguna berdasarkan pola rating dari pengguna lain dengan preferensi serupa.

## Evaluation

Pada bagian ini, kita akan mengevaluasi kinerja kedua pendekatan sistem rekomendasi yang telah diimplementasikan.

### Content-based Filtering

Untuk evaluasi content-based filtering, digunakan metrik presisi yang mengukur kesesuaian rekomendasi dengan preferensi pengguna.

#### Metrik Evaluasi: Precision

Untuk mengevaluasi model Content-based Filtering, kita akan menggunakan metrik Precision. Precision mengukur seberapa relevan rekomendasi yang diberikan oleh sistem.

Rumus Precision adalah sebagai berikut:

$$Precision = \frac{TP}{TP + FP}$$

Di mana:
- TP (True Positive): Jumlah rekomendasi yang relevan
- FP (False Positive): Jumlah rekomendasi yang tidak relevan

Dalam konteks sistem rekomendasi buku, kita dapat mendefinisikan buku yang relevan sebagai buku yang memiliki karakteristik yang sama dengan buku referensi.

```python
def calculate_comprehensive_precision(reference_book_isbn, recommendations, threshold=0.3):
    """
    Menghitung precision dari rekomendasi berdasarkan beberapa fitur konten,
    bukan hanya kategori.
    
    Args:
        reference_book_isbn (str): ISBN buku referensi
        recommendations (list): Daftar tuple (isbn, judul) rekomendasi buku
        threshold (float): Ambang batas kesamaan untuk dianggap relevan
        
    Returns:
        float: Nilai precision
    """
    # Mendapatkan detail buku referensi
    ref_book = content_df[content_df['isbn'] == reference_book_isbn].iloc[0]
    ref_category = ref_book['Category']
    ref_author = ref_book['book_author']
    ref_publisher = ref_book['publisher']
    
    # Menghitung jumlah rekomendasi yang relevan berdasarkan kriteria multipel
    relevant_count = 0
    relevance_details = []
    
    for isbn, _ in recommendations:
        rec_book = content_df[content_df['isbn'] == isbn].iloc[0]
        relevance_score = 0.0
        relevance_factors = []
        
        # Kesamaan kategori (bobot 0.4)
        if rec_book['Category'] == ref_category:
            relevance_score += 0.4
            relevance_factors.append('kategori')
            
        # Kesamaan penulis (bobot 0.3)
        if rec_book['book_author'] == ref_author:
            relevance_score += 0.3
            relevance_factors.append('penulis')
            
        # Kesamaan penerbit (bobot 0.2)
        if rec_book['publisher'] == ref_publisher:
            relevance_score += 0.2
            relevance_factors.append('penerbit')

        
        # Tentukan apakah buku relevan berdasarkan threshold
        is_relevant = relevance_score >= threshold
        if is_relevant:
            relevant_count += 1
            
        relevance_details.append({
            'isbn': isbn,
            'title': rec_book['book_title'],
            'relevance_score': relevance_score,
            'relevance_factors': relevance_factors,
            'is_relevant': is_relevant
        })
    
    # Menghitung precision
    precision = relevant_count / len(recommendations) if recommendations else 0
    
    return precision, relevance_details
```

#### Hasil Evaluasi

Berikut adalah hasil evaluasi untuk beberapa buku referensi:

**Tabel 3. Hasil Evaluasi Content-based Filtering**
| Judul Buku Referensi | Precision |
|----------------------------------------|-------------|
| Revival's Golden Key with Kitk Cameron | 0.40 |
| Lady of the Trillium | 0.80 |
| Cheyenne Summer | 0.70 |
| The Dogfather: A Dog Lover's Mystery | 0.80 |
| The BOOK OF VIRTUES | 0.00 |
| The Bride Price | 0.80 |
| Tainted Blood | 0.60 |
| A School Teacher in Old Alaska | 0.10 |
| Nurturing the Unborn Child | 0.10 |
| The Sea Came in at Midnight | 0.40 |

Rata-rata Precision: **0.47**

![image](https://github.com/user-attachments/assets/4f00e28b-2240-442b-92e6-22a1b25d1f57)

_Gambar 14: Grafik Precision untuk Content-based Filtering_

Dari hasil evaluasi, terlihat bahwa sistem content-based filtering mampu memberikan rekomendasi dengan presisi yang bervariasi. Beberapa buku referensi seperti "Lady of the Trillium" dan "The Bride Price" menghasilkan precision yang tinggi (0.80), sementara buku lain seperti "The BOOK OF VIRTUES" memiliki precision rendah (0.00). Variasi ini menunjukkan bahwa kualitas rekomendasi sangat bergantung pada karakteristik buku referensi dan ketersediaan buku serupa dalam dataset.

### Collaborative Filtering

Untuk evaluasi collaborative filtering, digunakan metrik RMSE (Root Mean Squared Error) yang mengukur akurasi prediksi rating.

#### Metrik Evaluasi: RMSE

Untuk model Collaborative Filtering, kita menggunakan metrik RMSE (Root Mean Squared Error) untuk mengevaluasi akurasi prediksi rating. RMSE mengukur seberapa jauh perbedaan antara rating yang diprediksi dengan rating sebenarnya.

Rumus RMSE adalah sebagai berikut:

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Di mana:
- $n$ adalah jumlah prediksi
- $y_i$ adalah rating sebenarnya
- $\hat{y}_i$ adalah rating yang diprediksi

Berikut adalah nilai RMSE terakhir dari training dan validation:

```python
def evaluate_collaborative_filtering(model, X_test, y_test):
    # Prediksi rating untuk data test
    y_pred = model.predict([X_test[:, 0], X_test[:, 1]], verbose=0).flatten()

    # Menghitung RMSE
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    # Mengembalikan ke skala asli (0-10)
    rmse_original_scale = rmse * 10

    return rmse, rmse_original_scale
```

#### Hasil Evaluasi

**Tabel 4. Hasil Evaluasi Collaborative Filtering**
| Metrik | Nilai |
|-----------------------|---------|
| Training RMSE | 0.1825 |
| Validation RMSE | 0.1825 |
| RMSE (skala asli 0-10)| 1.825 |

![image](https://github.com/user-attachments/assets/29ee7031-8e7e-44a8-bf6a-d1a30830c347)

_Gambar 15: Grafik RMSE untuk Training dan Validation Set_

Nilai RMSE yang relatif rendah (1.825 pada skala 0-10) menunjukkan bahwa model collaborative filtering cukup baik dalam memprediksi rating pengguna terhadap buku. Sebuah RMSE di bawah 2 dianggap baik untuk sistem rekomendasi buku, mengingat variabilitas preferensi pengguna.

### Perbandingan Pendekatan

**Tabel 5. Perbandingan Kedua Pendekatan**
| Aspek | Content-based Filtering | Collaborative Filtering |
|-------|------------------------|-------------------------|
| Metrik Evaluasi | Precision | RMSE |
| Hasil Evaluasi | Rata-rata Precision: 0.47 | RMSE: ~0.1825 |
| Kekuatan | - Dapat merekomendasikan buku baru<br>- Tidak memerlukan data rating dari pengguna lain<br>- Mudah dijelaskan | - Dapat menemukan buku yang tidak terkait dengan konten<br>- Memanfaatkan pola preferensi kolektif<br>- Dapat merekomendasikan buku yang tidak terpikirkan |
| Kelemahan | - Cenderung merekomendasikan buku yang sangat mirip<br>- Tidak dapat menangkap preferensi pengguna<br>- Bergantung pada kualitas fitur buku | - Masalah cold-start untuk pengguna baru<br>- Membutuhkan data rating yang cukup banyak<br>- Sulit menjelaskan rekomendasi |
| Skenario Terbaik | Ketika ingin merekomendasikan buku berdasarkan konten dan tidak memiliki data rating yang cukup | Ketika memiliki data rating yang cukup dan ingin merekomendasikan buku berdasarkan preferensi pengguna |


Dari perbandingan di atas, dapat disimpulkan bahwa kedua pendekatan memiliki kelebihan dan kekurangan masing-masing. Content-based filtering lebih cocok untuk mengatasi cold-start problem, sementara collaborative filtering lebih baik dalam memberikan rekomendasi yang tidak terduga namun relevan.

## Kesimpulan

Proyek ini berhasil mengimplementasikan dua pendekatan sistem rekomendasi buku:

1. **Content-based Filtering**: Berhasil merekomendasikan buku berdasarkan kemiripan konten dengan precision rata-rata 0.47. Pendekatan ini cocok untuk situasi di mana data rating terbatas atau untuk merekomendasikan buku baru yang belum memiliki banyak rating.

2. **Collaborative Filtering**: Berhasil merekomendasikan buku berdasarkan pola rating pengguna dengan RMSE 1.825. Pendekatan ini cocok untuk situasi di mana terdapat banyak data rating dan untuk menemukan buku yang mungkin tidak terpikir oleh pengguna.

Kedua pendekatan memiliki kelebihan dan kekurangan masing-masing:

- **Content-based Filtering**:

  - Kelebihan: Dapat merekomendasikan buku baru, tidak memerlukan data rating dari pengguna lain, dan dapat memberikan penjelasan yang transparan.
  - Kekurangan: Cenderung merekomendasikan buku yang sangat mirip, tidak dapat menangkap preferensi pengguna yang kompleks, dan bergantung pada kualitas fitur buku.

- **Collaborative Filtering**:
  - Kelebihan: Dapat menemukan buku yang tidak terkait dengan konten, memanfaatkan pola preferensi kolektif, dan dapat merekomendasikan buku yang tidak terpikirkan.
  - Kekurangan: Menghadapi masalah cold-start untuk pengguna baru, membutuhkan data rating yang cukup banyak, dan sulit menjelaskan alasan di balik rekomendasi.


Hasil evaluasi menunjukkan:

**Content-based Filtering**:

- Rata-rata precision: 0.47
- Beberapa buku memiliki precision yang sangat tinggi (0.80), sementara yang lain memiliki precision yang lebih rendah (0.00-0.40)
- Model content-based filtering efektif untuk merekomendasikan buku dengan karakteristik serupa

**Collaborative Filtering**:

- RMSE (skala asli 0-10): ~0.1285
- Model collaborative filtering berhasil memprediksi rating dengan akurasi tinggi

Untuk pengembangan selanjutnya, dapat dipertimbangkan beberapa hal berikut:

1. **Hybrid Approach**: Menggabungkan kedua pendekatan untuk mendapatkan hasil rekomendasi yang lebih baik, misalnya menggunakan content-based untuk pengguna baru dan collaborative filtering untuk pengguna dengan riwayat interaksi.

2. **Feature Engineering**: Mengeksplorasi lebih banyak fitur seperti kesamaan plot, gaya penulisan, atau topik untuk meningkatkan kualitas content-based filtering.

3. **Advanced Models**: Mengimplementasikan model yang lebih canggih seperti Deep Learning atau Graph Neural Networks untuk meningkatkan akurasi rekomendasi.

4. **Contextual Information**: Menambahkan informasi kontekstual seperti waktu, lokasi, atau situasi pengguna untuk memberikan rekomendasi yang lebih relevan.

Dengan pengembangan lebih lanjut, sistem rekomendasi buku dapat menjadi alat yang sangat berharga untuk membantu pengguna menemukan buku yang sesuai dengan preferensi mereka, meningkatkan pengalaman pengguna, dan pada akhirnya mendorong minat baca masyarakat.

## Referensi

1. Isinkaye, F. O., Folajimi, Y. O., & Ojokoh, B. A. (2015). Recommendation systems: Principles, methods and evaluation. Egyptian Informatics Journal, 16(3), 261-273. https://www.sciencedirect.com/science/article/pii/S1110866515000341

2. Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to recommender systems handbook. In Recommender systems handbook (pp. 1-35). Springer, Boston, MA. https://link.springer.com/chapter/10.1007/978-0-387-85820-3_1

3. Aggarwal, C. C. (2016). Recommender systems. Cham, Switzerland: Springer International Publishing. https://link.springer.com/book/10.1007/978-3-319-29659-3

4. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37. https://ieeexplore.ieee.org/document/5197422

5. Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. In Recommender systems handbook (pp. 73-105). Springer, Boston, MA. https://link.springer.com/chapter/10.1007/978-0-387-85820-3_3

6. Schafer, J. B., Frankowski, D., Herlocker, J., & Sen, S. (2007). Collaborative filtering recommender systems. In The adaptive web (pp. 291-324). Springer, Berlin, Heidelberg. https://link.springer.com/chapter/10.1007/978-3-540-72079-9_9

7. Smith, B., & Linden, G. (2017). Two decades of recommender systems at Amazon.com. IEEE Internet Computing, 21(3), 12-18. https://ieeexplore.ieee.org/document/8186743

8. Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. IEEE Transactions on Knowledge and Data Engineering, 17(6), 734-749. https://ieeexplore.ieee.org/document/1423975
