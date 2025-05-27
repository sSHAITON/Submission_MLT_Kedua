import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

df = pd.read_csv('preprocessed_data.csv')

df.head()

df.info()

df.describe()

df.describe(include='object')

category_counts = df['Category'].value_counts()
print(f"Jumlah nilai unik: {len(category_counts)}")
print("\nDistribusi kategori:")
print(category_counts.head(10))
num_category_9 = df[df['Category'] == '9'].shape[0]
total_books = df.shape[0]
percentage = (num_category_9 / total_books) * 100
print(f"\nBuku dengan kategori '9': {num_category_9} dari {total_books} ({percentage:.2f}%)")
summary_counts = df['Summary'].value_counts()
print(f"Jumlah nilai unik: {len(category_counts)}")
print("\nDistribusi summary:")
print(summary_counts.head(10))
num_category_9 = df[df['Summary'] == '9'].shape[0]
total_books = df.shape[0]
percentage = (num_category_9 / total_books) * 100
print(f"\nBuku dengan Summary '9': {num_category_9} dari {total_books} ({percentage:.2f}%)")

missing_values = df.isna().sum()
print("Missing Values:")
print(missing_values)

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

plt.figure(figsize=(8, 5))
sns.countplot(x='rating', data=df)
plt.title('Distribusi Rating')
plt.xlabel('Rating')
plt.ylabel('Jumlah')
plt.show()
print("Rata-rata rating:", df['rating'].mean())
print("Median rating:", df['rating'].median())

plt.figure(figsize=(10, 6))
sns.histplot(df['age'].dropna(), kde=True, bins=30)
plt.title('Distribusi Usia Pengguna')
plt.xlabel('Usia')
plt.ylabel('Jumlah')
plt.show()

plt.figure(figsize=(12, 6))
df['year_of_publication'].value_counts().sort_index().plot(kind='line')
plt.title('Distribusi Tahun Publikasi Buku')
plt.xlabel('Tahun Publikasi')
plt.ylabel('Jumlah Buku')
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(12, 6))
top_authors = df['book_author'].value_counts().head(10)
sns.barplot(x=top_authors.values, y=top_authors.index)
plt.title('10 Penulis dengan Jumlah Buku Terbanyak')
plt.xlabel('Jumlah Buku')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
language_counts = df['Language'].value_counts().head(10)
sns.barplot(x=language_counts.values, y=language_counts.index)
plt.title('10 Bahasa Teratas dalam Dataset')
plt.xlabel('Jumlah Buku')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
country_counts = df['country'].value_counts().head(10)
sns.barplot(x=country_counts.values, y=country_counts.index)
plt.title('10 Negara dengan Jumlah Pengguna Terbanyak')
plt.xlabel('Jumlah Pengguna')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='rating', y='year_of_publication', data=df[df['rating'] > 0].sample(10000))
plt.title('Distribusi Rating Berdasarkan Tahun Publikasi')
plt.xlabel('Rating')
plt.ylabel('Tahun Publikasi')
plt.show()

plt.figure(figsize=(12, 6))
publisher_ratings = df.groupby('publisher')['rating'].agg(['mean', 'count'])
publisher_ratings = publisher_ratings[publisher_ratings['count'] > 1000].sort_values('mean', ascending=False).head(10)
sns.barplot(x=publisher_ratings['mean'].values, y=publisher_ratings.index)
plt.title('Rata-rata Rating oleh 10 Penerbit Teratas (min. 1000 rating)')
plt.xlabel('Rata-rata Rating')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
top_authors = df['book_author'].value_counts().head(10).index
author_data = df[df['book_author'].isin(top_authors)]
sns.boxplot(x='book_author', y='rating', data=author_data)
plt.title('Distribusi Rating oleh 5 Penulis Teratas')
plt.xlabel('Penulis')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
numeric_df = df[['rating', 'age', 'year_of_publication']].dropna()
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasi antar Variabel Numerik')
plt.tight_layout()
plt.show()


# Melihat kembali kolom dengan missing value
print("Kolom dengan missing value:")
print(df.isna().sum()[df.isna().sum() > 0])
# Menangani missing value pada book_author
df['book_author'] = df['book_author'].fillna('Unknown Author')
# Untuk kolom lokasi, kita dapat mengisinya dengan 'Unknown' karena tidak terlalu berpengaruh pada model rekomendasi buku
df['city'] = df['city'].fillna('Unknown')
df['state'] = df['state'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')
# Verifikasi bahwa tidak ada lagi missing value
print("\nSetelah penanganan missing value:")
print(df.isna().sum()[df.isna().sum() > 0])
# Verifikasi tidak ada duplikasi (meskipun sudah diverifikasi sebelumnya)
print(f"\nJumlah duplikasi: {df.duplicated().sum()}")

# Memilih fitur untuk content-based filtering
content_features = ['isbn', 'book_title', 'book_author', 'publisher', 'Category', 'Summary']
# Filter data untuk menghilangkan buku dengan kategori '9' dan summary '9'
filtered_content_df = df[~(df['Category'] == '9') & ~(df['Summary'] == '9')]
content_df = filtered_content_df[content_features].drop_duplicates(subset=['isbn'])
# Memilih fitur untuk collaborative filtering (tidak perlu difilter)
collab_features = ['user_id', 'isbn', 'rating']
collab_df = df[collab_features]
# Melihat hasil pemilihan fitur
print("Dataset untuk content-based filtering (setelah menghilangkan anomali):")
print(content_df.head())
print(f"Jumlah buku unik: {content_df.shape[0]}")
print(f"Persentase data yang dipertahankan: {content_df.shape[0]/df['isbn'].nunique()*100:.2f}%")
# Melihat distribusi kategori setelah filtering
print("\nDistribusi kategori setelah filtering:")
category_counts = content_df['Category'].value_counts().head(10)
print(category_counts)
print("\nDataset untuk collaborative filtering:")
print(collab_df.head())
print(f"Jumlah interaksi user-item: {collab_df.shape[0]}")

# Menangani nilai kosong pada fitur teks
content_df['Summary'] = content_df['Summary'].fillna('')
content_df['Category'] = content_df['Category'].fillna('')
# Membuat fitur gabungan untuk TF-IDF
content_df['content'] = content_df['book_title'] + ' ' + content_df['book_author'] + ' ' + content_df['publisher'] + ' ' + content_df['Category'] + ' ' + content_df['Summary']
# Konversi ke lowercase
content_df['content'] = content_df['content'].str.lower()
print("Contoh fitur gabungan untuk TF-IDF:")
print(content_df['content'].head())

# Filter data dengan rating eksplisit (>0) untuk collaborative filtering
collab_df = collab_df[collab_df['rating'] > 0]
# Mengecek distribusi rating setelah filtering
print("Distribusi rating setelah filtering:")
print(collab_df['rating'].value_counts().sort_index())
# Melihat jumlah user dan item setelah filtering
print(f"Jumlah user: {collab_df['user_id'].nunique()}")
print(f"Jumlah buku: {collab_df['isbn'].nunique()}")
print(f"Jumlah interaksi: {collab_df.shape[0]}")

# Menghitung jumlah rating per user dan per buku
user_counts = collab_df['user_id'].value_counts()
book_counts = collab_df['isbn'].value_counts()
# Filter user yang telah memberikan rating minimal 5 buku
min_user_ratings = 5
active_users = user_counts[user_counts >= min_user_ratings].index
filtered_df = collab_df[collab_df['user_id'].isin(active_users)]
# Filter buku yang telah mendapatkan rating minimal dari 5 user
min_book_ratings = 5
popular_books = book_counts[book_counts >= min_book_ratings].index
filtered_df = filtered_df[filtered_df['isbn'].isin(popular_books)]
print("Setelah filtering untuk mengurangi sparsity:")
print(f"Jumlah user: {filtered_df['user_id'].nunique()}")
print(f"Jumlah buku: {filtered_df['isbn'].nunique()}")
print(f"Jumlah interaksi: {filtered_df.shape[0]}")

# Membuat mapping ID untuk user dan buku
user_ids = filtered_df['user_id'].unique().tolist()
isbn_ids = filtered_df['isbn'].unique().tolist()
# Membuat dictionary untuk mapping
user_to_idx = {user: i for i, user in enumerate(user_ids)}
isbn_to_idx = {isbn: i for i, isbn in enumerate(isbn_ids)}
# Memetakan ID ke indeks berurutan
filtered_df['user_idx'] = filtered_df['user_id'].map(user_to_idx)
filtered_df['isbn_idx'] = filtered_df['isbn'].map(isbn_to_idx)
print("Data setelah mapping ID:")
print(filtered_df.head())

# Library untuk Content-based Filtering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Library untuk Collaborative Filtering
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

content_df = content_df.reset_index(drop=True)
indices = pd.Series(content_df.index, index=content_df['isbn']).drop_duplicates()
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(content_df['content'])
print(f"Dimensi matriks TF-IDF: {tfidf_matrix.shape}")
print(f"Jumlah fitur/kata unik: {len(tfidf.get_feature_names_out())}")

isbn_to_title = dict(zip(content_df['isbn'], content_df['book_title']))

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


valid_isbns = list(indices.index)
if len(valid_isbns) > 0:
    sample_isbn = valid_isbns[15]
    sample_title = isbn_to_title.get(sample_isbn, "Unknown Title")
    
    print(f"Buku Referensi: {sample_title} (ISBN: {sample_isbn})")
    print("\nRekomendasi Buku Berdasarkan Konten:")
    
    recommendations = get_content_based_recommendations(sample_isbn)
    for i, (rec_isbn, rec_title) in enumerate(recommendations, 1):
        print(f"{i}. {rec_title} (ISBN: {rec_isbn})")
else:
    print("Tidak ada ISBN valid dalam indices.")

complete_isbn_to_title = dict(zip(df['isbn'], df['book_title']))

# Menyiapkan data untuk modeling
X = filtered_df[['user_idx', 'isbn_idx']].values
y = filtered_df['rating'].values
# Normalisasi rating ke rentang [0, 1]
y = y / 10.0
# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Mendapatkan jumlah user dan buku
num_users = len(user_to_idx)
num_books = len(isbn_to_idx)
print(f"Jumlah user: {num_users}")
print(f"Jumlah buku: {num_books}")
print(f"Jumlah data training: {len(X_train)}")
print(f"Jumlah data testing: {len(X_test)}")

# Menentukan dimensi embedding
embedding_dim = 50
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
# Membuat model
model = create_model()
# Menampilkan summary model
model.summary()

# Early stopping untuk menghindari overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
# Melatih model
history = model.fit(
    [X_train[:, 0], X_train[:, 1]],
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

def get_collaborative_recommendations(user_id, model, isbn_to_idx, user_to_idx, k=10):
    """
    Memberikan rekomendasi buku untuk pengguna tertentu berdasarkan collaborative filtering
    
    Parameters:
    user_id (int): ID pengguna
    model: Model yang telah dilatih
    isbn_to_idx (dict): Mapping dari ISBN ke indeks
    user_to_idx (dict): Mapping dari user ID ke indeks
    k (int): Jumlah rekomendasi yang diinginkan
    
    Returns:
    list: Daftar ISBN dan judul buku yang direkomendasikan beserta prediksi rating
    """
    try:
        # Mendapatkan indeks user
        user_idx = user_to_idx[user_id]
        
        # Mendapatkan semua buku dalam dataset
        all_books = list(isbn_to_idx.keys())
        
        # Membuat data untuk prediksi
        user_book_array = np.array([[user_idx, isbn_to_idx[isbn]] for isbn in all_books])
        
        # Memprediksi rating untuk semua kombinasi user-book
        predictions = model.predict([user_book_array[:, 0], user_book_array[:, 1]], verbose=0).flatten() * 10  # Mengembalikan ke skala asli
        
        # Menggabungkan hasil prediksi dengan ISBN buku
        book_predictions = list(zip(all_books, predictions))
        
        # Mengurutkan berdasarkan prediksi rating tertinggi
        book_predictions = sorted(book_predictions, key=lambda x: x[1], reverse=True)
        
        # Mendapatkan k rekomendasi teratas
        top_recommendations = book_predictions[:k]
        
        # Menyiapkan hasil rekomendasi dengan ISBN, judul, dan prediksi rating
        recommendations = [(isbn, isbn_to_title.get(isbn, "Unknown Title"), rating) for isbn, rating in top_recommendations]
        
        return recommendations
    
    except KeyError:
        print(f"User ID {user_id} tidak ditemukan dalam dataset.")
        return []

# Memilih user acak sebagai contoh
sample_user_id = np.random.choice(user_ids)
print(f"User ID: {sample_user_id}")
print("\nRekomendasi Buku Berdasarkan Collaborative Filtering:")
recommendations = get_collaborative_recommendations(sample_user_id, model, isbn_to_idx, user_to_idx)
for i, (rec_isbn, rec_title, pred_rating) in enumerate(recommendations, 1):
    print(f"{i}. {rec_title} (ISBN: {rec_isbn}) ")

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

import random
precision_results = {}
sample_books = random.sample(list(content_df['isbn'].unique()), 10)  # Use unique ISBNs directly
for isbn in sample_books:
    title = isbn_to_title.get(isbn, "Unknown Title")
    recommendations = get_content_based_recommendations(isbn)
    
    if recommendations:
        precision = calculate_precision(isbn, recommendations)
        precision_results[title] = precision
        print(f"Precision for '{title}': {precision:.2f}")
# Calculate average precision
avg_precision = sum(precision_results.values()) / len(precision_results) if precision_results else 0
print(f"\nAverage precision: {avg_precision:.2f}")

import matplotlib.pyplot as plt
# Membuat grafik batang untuk precision
plt.figure(figsize=(12, 10))
books = list(precision_results.keys())
precision_values = list(precision_results.values())
# Membuat warna yang berbeda berdasarkan nilai precision
colors = ['green' if p >= 0.8 else 'orange' if p >= 0.5 else 'red' for p in precision_values]
bars = plt.bar(books, precision_values, color=colors)
plt.axhline(y=avg_precision, color='blue', linestyle='--', label=f'Rata-rata Precision: {avg_precision:.2f}')
# Menambahkan label pada sumbu x dan y
plt.xlabel('Buku Referensi')
plt.ylabel('Precision')
plt.title('Precision dari Content-based Filtering untuk Berbagai Buku Referensi')
# Menambahkan nilai precision di atas setiap bar
for bar, precision in zip(bars, precision_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{precision:.2f}',
            ha='center', va='bottom', rotation=0)
plt.ylim(0, 1.1)  # Mengatur rentang sumbu y
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.legend()
plt.show()

# Evaluasi model pada data testing
test_loss, test_rmse = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test RMSE (scaled): {test_rmse:.4f}")
print(f"Test RMSE (original scale): {test_rmse * 10:.4f}")
# Visualisasi hasil pelatihan
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.subplot(1, 2, 2)
plt.plot(history.history['root_mean_squared_error'], label='Training RMSE')
plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.title('Training and Validation RMSE')
plt.tight_layout()
plt.show()

print(f"Training RMSE: {history.history['root_mean_squared_error'][-1]:.4f}")
print(f"Validation RMSE: {history.history['val_root_mean_squared_error'][-1]:.4f}")