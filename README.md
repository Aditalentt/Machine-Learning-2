# Machine-Learning-2

Prediksi Tinggi Gelombang Signifikan menggunakan Machine Learning
Repositori ini berisi skrip eksperimen untuk memprediksi tinggi
gelombang signifikan (Hsig(m)) berdasarkan data angin dan kondisi
laut. Skrip mencakup eksplorasi data, rekayasa fitur temporal,
perbandingan model regresi, pendekatan heterogeneous stacking, sistem
prediksi adaptif untuk kondisi gelombang tinggi, serta simulasi pengaruh
penambahan data latih.
Catatan: Skrip sumber dibuat dari Google Colab. Penyesuaian path
dataset dan pemeriksaan kolom diperlukan sebelum dijalankan di
lingkungan lain.

Tujuan
- Mengeksplorasi kualitas data dan korelasi antarvariabel numerik.
- Membentuk fitur temporal dari kecepatan angin.
- Memprediksi tinggi gelombang signifikan menggunakan model regresi.
- Membandingkan model baseline dengan heterogeneous stacking.
- Menguji pendekatan hibrida adaptif yang memberi perhatian lebih pada
  sampel gelombang tinggi.
- Mengevaluasi model secara keseluruhan dan pada subset gelombang
  tinggi (Hsig > 1.5 m).
Dataset
Skrip membaca dataset CSV dari path berikut:
df = pd.read_csv('/content/angin')
Ubah path tersebut sesuai lokasi file dataset Anda. File CSV harus
memiliki kolom-kolom yang digunakan dalam skrip, antara lain:
- WindSpeed(knots)
- WindDir(deg)
- WindSeaDir(deg)
- WavePeriod(s)
- SurfCurrentSpd(cm/s)
- Hsig(m)
Pastikan juga nama kolom persis sama dengan yang tercantum pada kode.
Skrip tidak menyertakan dataset, sehingga dataset perlu disediakan
secara terpisah.
Persiapan lingkungan
Gunakan Python 3 dan instal pustaka berikut:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
Skrip mengimpor LGBMRegressor, tetapi LightGBM tidak digunakan pada
rangkaian model komparatif yang didefinisikan. Jika impor tersebut tetap
dipertahankan, instal juga:
pip install lightgbm
Alur eksperimen
1. Eksplorasi data
- Memuat data CSV ke dalam DataFrame.
- Menampilkan beberapa baris awal.
- Memeriksa jumlah nilai kosong pada setiap kolom.
- Membuat heatmap korelasi untuk kolom numerik.
2. Prapemrosesan dan rekayasa fitur
- Membuat fitur SurfCurrentSpd_Capped dengan membatasi nilai
  kecepatan arus permukaan yang lebih besar dari 100 menjadi 100.
- Mengubah arah angin dan arah gelombang angin dari derajat menjadi
  komponen sinus dan kosinus agar sifat melingkar sudut dapat
  direpresentasikan.
- Membuat fitur jeda kecepatan angin (WindSpeed_Lag1 dan
  WindSpeed_Lag3).
- Membuat rata-rata bergerak dan standar deviasi bergerak kecepatan
  angin menggunakan rolling(window=2).
- Menghapus baris yang mengandung nilai kosong setelah pembentukan
  fitur.
Perhatian: Nama fitur rolling pada skrip menyebut 6h, tetapi
ukuran jendela yang diterapkan adalah 2 baris. Makna enam jam hanya
tepat apabila interval antarobservasi mendukung interpretasi tersebut.
Pastikan interval waktu dan urutan kronologis dataset benar sebelum
pemodelan.
3. Fitur dan target
Fitur model meliputi kecepatan dan arah angin, arah gelombang, periode
gelombang, kecepatan arus permukaan yang dibatasi, serta fitur lag dan
rolling kecepatan angin.

Target prediksi adalah:
y = df_clean['Hsig(m)']
4. Pembagian data
Data dibagi secara berurutan dengan 80% bagian awal sebagai data latih
dan 20% bagian akhir sebagai data uji. Pengacakan dinonaktifkan agar
urutan data tetap terjaga.
5. Model yang dievaluasi
- Baseline XGBoost --- XGBRegressor.
- Baseline Ridge --- regresi Ridge dengan standardisasi fitur.
- Heterogeneous Stacking --- menggabungkan KNN, Ridge, dan
  XGBoost, dengan Ridge sebagai final estimator.
- Adaptive Reinforcement Hybrid --- menggabungkan prediksi XGBoost
  dan Ridge. Model Ridge dilatih dengan bobot sampel lebih besar untuk
  observasi dengan Hsig(m) > 1.5; prediksi dicampur secara adaptif
  ketika salah satu prediksi model melebihi 1.2 m.
Dalam skrip, bobot sampel untuk observasi gelombang tinggi adalah 5,
sedangkan sampel lainnya berbobot 1. Pada kondisi yang memenuhi aturan
ambang prediksi, hasil akhir menggunakan 80% prediksi Ridge dan 20%
prediksi XGBoost.
6. Evaluasi dan visualisasi
Metrik yang dihitung meliputi:
- RMSE (Root Mean Squared Error): mengukur besarnya kesalahan
  prediksi dengan penalti lebih besar untuk kesalahan besar.
- MAE (Mean Absolute Error): rata-rata nilai absolut kesalahan
  prediksi.
- R²: mengukur proporsi variasi target yang dijelaskan model.
- RMSE gelombang ekstrem: RMSE pada data uji dengan
  Hsig(m) > 1.5 m.
Skrip juga membuat visualisasi perbandingan model, perbandingan nilai
aktual dan prediksi, scatter plot aktual versus prediksi, dan simulasi
bertahap penambahan jumlah data latih.
Cara menjalankan
1. Siapkan dataset CSV dengan kolom yang sesuai.
2. Ubah path pd.read_csv('/content/angin') ke path dataset yang
   benar.
3. Pastikan data terurut secara kronologis dan interval observasinya
   diketahui.
4. Instal pustaka yang diperlukan.
5. Jalankan skrip di Jupyter Notebook, Google Colab, atau lingkungan
   Python yang sesuai.
6. Periksa metrik dan grafik yang dihasilkan.
Catatan penting sebelum menggunakan hasil
- Periksa nilai waktu dan urutan data. Fitur lag dan rolling
  mengasumsikan urutan observasi yang benar.
- Periksa jumlah sampel gelombang ekstrem. Perhitungan RMSE
  ekstrem memerlukan setidaknya satu sampel uji dengan
  Hsig(m) > 1.5.
- Validasi ambang 1.2 m dan 1.5 m. Ambang tersebut ditetapkan di
  kode dan perlu didukung alasan operasional atau analisis data.
- Periksa grafik hasil akhir. Salah satu bagian visualisasi
  menggunakan angka metrik yang ditulis langsung di dalam kode, bukan
  diambil otomatis dari hasil eksperimen. Angka tersebut dapat tidak
  sesuai dengan hasil terbaru jika dataset atau konfigurasi berubah.
- Hindari menyimpulkan kausalitas dari korelasi. Korelasi antara
  kecepatan angin dan tinggi gelombang tidak dengan sendirinya
  membuktikan bahwa angin lokal adalah satu-satunya penyebab
  pembentukan gelombang.
- Evaluasi pada data uji yang tidak digunakan saat pelatihan.
  Jangan memilih model atau ambang berdasarkan performa data uji
  berulang kali tanpa validasi yang terpisah.
Struktur berkas
.
├── ets_ml2_aditya_putra_ananta.py
├── README.md
└── angin                 # dataset CSV, disediakan terpisah
Status
Skrip ini merupakan kode eksperimen/pemodelan. Hasil yang diperoleh
bergantung pada dataset, kualitas data, urutan observasi, dan
konfigurasi model yang digunakan.
