
<div align="center">

# 🍄 **Mushroom Classification App**
### _AI-Powered Mushroom Classifier using Streamlit & Machine Learning_

![Mushroom Banner](https://i.pinimg.com/1200x/1d/06/85/1d068548986d7a0e11353ad7f70f2d26.jpg)

</div>

---

## 📘 **Deskripsi Proyek**
Aplikasi **Mushroom Classification App** dikembangkan sebagai bagian dari **Kursus Data Science** untuk **Skema Associate Data Scientist**.  Tujuan utama proyek ini adalah untuk mengklasifikasikan jamur menjadi **edible (dapat dimakan)** atau **poisonous (beracun)** berdasarkan fitur morfologinya menggunakan pendekatan **Machine Learning**.

Model yang digunakan:
- 🔹 **Logistic Regression**
- 🔹 **Random Forest Classifier**
- 🔹 **XGBoost Classifier**

> Hasil evaluasi menunjukkan bahwa model **XGBoost** memberikan akurasi tertinggi sebesar **100%**, diikuti **Logistic Regression (99.9%)** dan **Random Forest (92.6%)**.

---

## 📊 **Dataset Overview**
Dataset mencakup 22 fitur kategori seperti:  
**cap-shape**, **cap-surface**, **cap-color**, **odor**, **gill-size**, **ring-type**, dan lainnya.  

**Distribusi kelas:**

| Kelas | Keterangan | Jumlah Data |
|:------|:------------|-------------:|
| `e` | Edible 🍽️ | 4208 |
| `p` | Poisonous ☠️ | 3916 |

---

## 🧠 **Pengembangan Model**
Model dikembangkan menggunakan pustaka **Scikit-learn** dan **XGBoost** dengan pipeline berikut:

1️⃣ **Preprocessing Data** – Encoding fitur kategori dan membagi data train-test  
2️⃣ **Training Model** – Menggunakan tiga algoritma utama  
3️⃣ **Evaluation** – Mengukur akurasi, precision, recall, dan F1-score  

**📈 Hasil Evaluasi Model**

| Model | Akurasi |
|:------|:--------:|
| Logistic Regression | 99.9% |
| Random Forest | 92.6% |
| XGBoost | 🏆 **100%** |

---

## 🌐 **Antarmuka Web Aplikasi**

Aplikasi **Mushroom Classification App** dibangun menggunakan framework **Streamlit**, yang memungkinkan pengguna untuk melakukan **klasifikasi jamur secara interaktif** tanpa perlu menulis kode secara manual.  Antarmuka aplikasi dirancang sederhana dan intuitif agar mudah digunakan oleh pengguna dari berbagai latar belakang, baik akademik maupun non-teknis.


### 🏠 **Halaman Utama (Home Page)**  
![Home Page](https://github.com/user-attachments/assets/ca148225-42c1-47ab-b38a-fcb804d2357c)  
> Halaman utama menampilkan **gambaran umum proyek**, tujuan aplikasi, serta navigasi menuju fitur-fitur utama.  
> Di bagian bawah halaman juga terdapat **informasi pengembang dan deskripsi singkat aplikasi**.

---

### 📂 **Halaman Upload Dataset**  
![Upload Dataset](https://github.com/user-attachments/assets/e324b6d4-4105-4b78-88cf-07617dc65dff)  
> Pada halaman ini, pengguna dapat **mengunggah dataset jamur dalam format CSV**.  
> Setelah diunggah, sistem akan menampilkan **pratinjau data** agar pengguna dapat memastikan format dan isi data sudah sesuai.

---

### 🤖 **Halaman Klasifikasi (Classifier Page)**  
![Classifier Page](https://github.com/user-attachments/assets/b984fc71-59bd-4cc9-a211-d3d75a52dca1)  
> Bagian ini memungkinkan pengguna untuk **memilih fitur (features)** dan **target (label)** yang akan digunakan untuk proses klasifikasi.  
> Pengguna juga dapat memilih algoritma yang diinginkan, seperti **Logistic Regression**, **Random Forest**, atau **XGBoost**, untuk melihat perbandingan performa antar model.

---

### 📊 **Halaman Hasil Klasifikasi**  
![Result Page](https://github.com/user-attachments/assets/d60738c9-c81e-47df-bc99-1e8b73a4ba9b)  
> Setelah proses klasifikasi selesai, halaman ini menampilkan **hasil prediksi** beserta **tingkat akurasi model**.  
> Hasil disajikan dalam bentuk **tabel prediksi dan metrik evaluasi** untuk membantu pengguna memahami performa dari model yang digunakan.



✨ Antarmuka ini dirancang agar mendukung proses **eksperimen machine learning secara interaktif**, sekaligus memperlihatkan bagaimana **AI dapat digunakan untuk klasifikasi berbasis data nyata**.


---


## ⚙️ **Cara Menjalankan Program**

### 🔧 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/mushroom-classification-app.git
cd mushroom-classification-app
````

### 💾 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 🚀 3️⃣ Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

---

## 🧩 **Teknologi yang Digunakan**

| Kategori           | Tools / Library       |
| :----------------- | :-------------------- |
| Bahasa Pemrograman | Python 3.x            |
| Data Handling      | Pandas, NumPy         |
| Machine Learning   | Scikit-learn, XGBoost |
| Visualisasi        | Matplotlib, Seaborn   |
| Web Framework      | Streamlit             |

---

## 🗂️ **Struktur Proyek**

```
📂 Mushroom-Classification-App
 ┣ 📁 data               # Dataset jamur
 ┣ 📁 models             # Model Machine Learning
 ┣ 📁 images             # Gambar hasil visualisasi
 ┣ 📜 app.py             # File utama Streamlit
 ┣ 📜 requirements.txt   # Dependencies
 ┗ 📜 README.md          # Dokumentasi proyek
```

---

## 📈 **Kesimpulan**

Aplikasi **Mushroom Classification App** memanfaatkan **kecerdasan buatan** untuk membantu proses **grading dan klasifikasi jamur** dengan cepat dan akurat.
Dengan akurasi **100%** dari model **XGBoost**, aplikasi ini membuktikan efektivitas AI dalam **data-driven decision-making** di bidang biologi dan keamanan pangan.

---

<div align="center">

✨ **Dibuat oleh:**
👩‍💻 [Zahwa Genoveva](https://github.com/zahwagenoveva)
🎓 *Universitas Gunadarma – LSP, Skema Associate Data Scientist*

</div>

