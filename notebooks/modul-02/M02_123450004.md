# Laporan Praktikum M02 - Backpropagation dan Automatic Differentiation

> **Cara menggunakan template**
>
> 1. Salin berkas ini dan ganti namanya menjadi `MXX_NIM.md`.
> 2. Ganti `MXX`, `NIM`, teks `[ISI ...]`, serta contoh pada setiap tabel.
> 3. Ikuti batas halaman pada modul: maksimal 2 halaman untuk Modul 1-3,
>    3 halaman untuk Modul 4-8, dan 4 halaman untuk Modul 9.
> 4. Ekspor laporan menjadi `MXX_NIM.pdf`.
> 5. Hapus kotak petunjuk ini sebelum laporan dikumpulkan.

---

## Identitas Praktikan

| Komponen | Isian |
|---|---|
| Nama | Luthfia Laila Ramadhani |
| NIM | 123450004 |
| Kelas | RB |
| Modul | M01 - Fondasi Jaringan Saraf, FNN, Aktivasi, dan Loss |
| Tanggal praktikum | 2026-09-18 |
| Seed/varian individual | 1004|
| Device | ASUS Vivobook|

## Ringkasan Singkat

Praktikum ini membahas proses backpropagation dan automatic differentiation untuk menghitung serta memeriksa gradien pada jaringan saraf sederhana. Eksperimen dilakukan melalui perhitungan forward pass, turunan manual, perbandingan dengan autograd PyTorch, dan gradient checking menggunakan selisih hingga. Hasil menunjukkan bahwa gradien manual dan autograd memiliki nilai yang sama, sedangkan relative error maksimum pada gradient checking hanya sebesar (1.122\times10^{-10}), sehingga seluruh gradien memenuhi ambang (10^{-5}). Diagnosis training loop pada kasus XOR menunjukkan bahwa perbaikan urutan `backward()` dan `step()`, penggunaan `zero_grad()`, penggunaan `BCEWithLogitsLoss` tanpa sigmoid tambahan, serta penambahan ReLU diperlukan agar proses pembelajaran berjalan dengan benar. Setelah seluruh perbaikan diterapkan, model menghasilkan loss akhir sebesar 0.0728 dan prediksi `[0, 1, 1, 0]` yang sesuai dengan target XOR.

## 1. Tujuan dan Hipotesis

### 1.1 Tujuan

1. Menghitung gradien secara manual menggunakan backpropagation pada jaringan saraf sederhana.
2. Membandingkan hasil gradien manual dengan autograd dan gradient checking.
3. Menganalisis serta memperbaiki kesalahan pada training loop menggunakan kasus XOR.

### 1.2 Hipotesis sebelum eksperimen

| Perbandingan | Prediksi | Alasan teknis |
|---|---|---|
| Gradien manual vs autograd vs finite difference | Ketiganya identik, relative error < 10⁻⁵ | Ketiganya menghitung turunan analitik yang sama; finite difference hanya menambah galat pembulatan kecil |
| Perbaikan 1–3 vs Perbaikan 4 (tambah ReLU) | Perbaikan 4 memberi penurunan loss jauh lebih besar | XOR tidak linearly separable; tanpa non-linearitas model tetap ekuivalen satu transformasi linear |
| Gradien per-contoh vs gradien batch rata-rata | `dW2` batch ≈ rata-rata gradien dua contoh (dibagi 2) | `loss_batch` = `loss_each.mean()`, sehingga `backward()` menjumlahkan lalu membagi kontribusi tiap contoh dengan n=2 |

## 2. Data dan Protokol Eksperimen

### 2.1 Dataset dan Split

| Komponen | Nilai |
|---|---|
| Dataset | Data sintetis: (1) Kasus 1 — satu contoh tetap \(x=[2,-1]\), \(y=1\) untuk verifikasi gradien; (2) XOR — 4 titik biner \(\{0,1\}^2\) dengan target XOR untuk diagnosis training loop |
| Jumlah kelas/target | Klasifikasi biner (\(y \in \{0,1\}\)) |
| Train | Kasus 1: 1 contoh tetap. XOR: seluruh 4 titik digunakan sebagai satu batch penuh |
| Validation | Tidak ada, karena data digunakan untuk verifikasi gradien dan diagnosis training loop, bukan untuk mengukur generalisasi |
| Test | Tidak ada test set terpisah; 4 titik XOR dievaluasi langsung setelah training sesuai protokol modul |
| Cara split | Tidak dilakukan split; seluruh data digunakan secara penuh (full-batch) dengan `seed_everything(SEED)` untuk menjaga eksperimen tetap deterministik |
| Praproses utama | Tidak ada normalisasi; nilai \(x\), \(W\), dan \(b\) digunakan dalam skala yang telah ditentukan dan perhitungan menggunakan `float64` sesuai ketentuan modul |

**Pencegahan kebocoran data:** Karena tujuan modul adalah memverifikasi perhitungan gradien dan mendiagnosis training loop, bukan menguji generalisasi model, tidak dilakukan pembagian train/validation/test sehingga tidak terdapat proses pemilihan model berdasarkan data uji.

### 2.2 Konfigurasi yang dikendalikan

| Komponen | Nilai yang digunakan |
|---|---|
| Arsitektur dasar | Jaringan saraf sederhana dengan 2 neuron input, 2 neuron hidden, dan 1 neuron output |
| Loss function | `BCEWithLogitsLoss` |
| Optimizer | SGD |
| Learning rate | 0.1 |
| Batch size | Full-batch (4 data XOR) |
| Epoch/jumlah update | 400 epoch |
| Seed | 4 |
| Kriteria pemilihan model | Tidak dilakukan pemilihan model berdasarkan validation loss; hasil dievaluasi berdasarkan loss dan prediksi akhir pada data XOR |
| Batas komputasi | Tidak ditetapkan secara khusus; eksperimen menggunakan jumlah epoch dan konfigurasi yang telah ditentukan pada modul |

**Variabel yang dibuat sama untuk seluruh run:** seed, dataset XOR, arsitektur jaringan, fungsi loss, optimizer, learning rate, batch size, dan jumlah epoch.

**Variabel yang sengaja diubah:** komponen kesalahan pada training loop diperbaiki secara bertahap untuk membandingkan perubahan hasil loss hingga diperoleh konfigurasi training loop yang benar.

### 2.3 Lingkungan eksekusi

```text
Python : 3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0]
PyTorch: 2.11.0+cpu
NumPy  : 2.1.3
Device : cpu
Runtime : Colab
```

## 3. Implementasi dan Pemeriksaan Kebenaran

Pemeriksaan dilakukan untuk memastikan implementasi jaringan, parameter, dan perhitungan gradien telah berjalan sesuai dengan konfigurasi eksperimen.

| Pemeriksaan | Nilai yang diharapkan | Hasil aktual | Status |
|---|---:|---:|---|
| Shape input \(x\) | `(2,)` | `(2,)` | Lulus |
| Shape \(W_1\) | `(2, 2)` | `(2, 2)` | Lulus |
| Shape \(W_2\) | `(2,)` | `(2,)` | Lulus |
| Jumlah parameter | 9 | 9 | Lulus |
| `requires_grad` parameter | `True` | `True` | Lulus |
| Kesesuaian gradien manual dan autograd | Selisih maksimum mendekati 0 | 0 | Lulus |
| Gradient checking | Relative error < \(10^{-5}\) | \(1.122\times10^{-10}\) | Lulus |

**Temuan dari pemeriksaan:** Seluruh shape tensor dan jumlah parameter sesuai dengan arsitektur jaringan yang digunakan. Parameter yang digunakan untuk perhitungan gradien memiliki `requires_grad=True`, sehingga autograd dapat menghitung turunannya. Hasil gradient checking menunjukkan relative error maksimum sebesar \(1.122\times10^{-10}\), jauh di bawah batas toleransi \(10^{-5}\), sehingga implementasi perhitungan gradien dinyatakan benar.

## 4. Hasil Eksperimen

### 4.1 Tabel hasil utama

Hasil gradient checking diperoleh dengan membandingkan gradien manual, autograd, dan gradien numerik pada seluruh parameter jaringan.

| Run | Perubahan utama | Seed | Jumlah parameter | Max. relative error | Status |
|---|---|---:|---:|---:|---|
| `gradcheck` | Membandingkan gradien manual, autograd, dan numerik | 4 | 9 | \(1.122192\times10^{-10}\) | Lulus |

Hasil tersebut menunjukkan bahwa seluruh 9 parameter memiliki nilai gradien manual, autograd, dan numerik yang sangat dekat. Nilai relative error maksimum sebesar \(1.122192\times10^{-10}\), sehingga berada jauh di bawah batas toleransi \(10^{-5}\).

**Catatan:** Eksperimen tidak menggunakan train/validation split, sehingga metrik `Train loss`, `Val. loss`, dan `Val. metric` tidak tersedia pada file metrics. Hasil utama yang digunakan adalah nilai gradien dan relative error dari proses gradient checking.

## 5. Analisis dan Pembahasan

### 5.1 Perbandingan dengan Hipotesis

Hasil eksperimen secara umum mendukung hipotesis yang telah dibuat. Gradien manual dan autograd menghasilkan nilai yang sama dengan selisih maksimum 0. Pada gradient checking, nilai relative error maksimum sebesar \(1.122\times10^{-10}\), yang berada jauh di bawah batas toleransi \(10^{-5}\). Hasil ini menunjukkan bahwa perhitungan gradien secara analitik sesuai dengan hasil autograd dan gradien numerik.

Pada eksperimen training loop, perbaikan kesalahan secara bertahap menghasilkan perubahan loss dari konfigurasi awal hingga konfigurasi yang benar. Setelah seluruh kesalahan diperbaiki, loss turun dari \(0.59907698\) menjadi \(0.07280489\), dengan prediksi XOR yang sesuai dengan target, yaitu \([0,1,1,0]\).

### 5.2 Perbandingan Hasil dan Diagnosis

Perubahan hasil training loop dapat dilihat dari nilai loss pada setiap tahap perbaikan.

| Tahap | Loss awal | Loss akhir | Temuan |
|---|---:|---:|---|
| Perbaikan 1 | 0.72043891 | 0.69314718 | Loss hanya turun hingga sekitar 0.693 |
| Perbaikan 2 | 0.72043891 | 0.69027663 | Loss sedikit lebih rendah |
| Perbaikan 3 | 0.70428608 | 0.69325355 | Perbaikan belum menghasilkan pembelajaran yang baik |
| Perbaikan 4 | 0.59907698 | 0.07280489 | Loss turun secara signifikan dan prediksi XOR sesuai target |

Perbedaan tersebut terjadi karena beberapa kesalahan pada training loop memengaruhi proses perhitungan dan pembaruan parameter. Kesalahan yang ditemukan meliputi penggunaan sigmoid sebelum `BCEWithLogitsLoss`, urutan `opt.step()` sebelum `loss.backward()`, tidak melakukan `opt.zero_grad()`, serta tidak adanya aktivasi nonlinier di antara layer linear. Setelah seluruh kesalahan diperbaiki, proses backpropagation dan pembaruan parameter dapat berjalan sesuai konsep yang diharapkan.

### 5.3 Trade-off dan Stabilitas

Pada praktikum ini tidak dilakukan perbandingan berdasarkan waktu komputasi, jumlah parameter, atau penggunaan memori karena konfigurasi utama jaringan tidak divariasikan untuk tujuan tersebut. Analisis difokuskan pada ketepatan perhitungan gradien dan kestabilan proses training.

Hasil gradient checking menunjukkan nilai relative error maksimum sebesar \(1.122\times10^{-10}\), sehingga perbedaan antara gradien analitik dan numerik sangat kecil. Selain itu, hasil akhir training XOR menghasilkan loss sebesar \(0.07280489\) dan prediksi \([0,1,1,0]\), yang menunjukkan bahwa konfigurasi training loop yang telah diperbaiki dapat mempelajari pola XOR.

### 5.4 Anomali atau Hasil Gagal

Hasil pada tahap awal training loop menunjukkan bahwa loss tidak turun secara signifikan. Kondisi tersebut kemudian didiagnosis berasal dari empat kesalahan implementasi, yaitu penggunaan sigmoid sebelum `BCEWithLogitsLoss`, urutan operasi `step()` dan `backward()` yang tidak tepat, tidak adanya `zero_grad()`, serta tidak adanya aktivasi nonlinier di antara layer.

Setelah kesalahan tersebut diperbaiki secara bertahap, pada perbaikan terakhir loss berhasil turun dari \(0.59907698\) menjadi \(0.07280489\). Hal ini memberikan bukti bahwa perubahan hasil traning berkaitan dengan perbaikan komponen yang sebelumnya salah pada training loop.

### 5.5 Generalisasi

Eksperimen ini tidak menggunakan train-validation split karena tujuan utama praktikum adalah memverifikasi perhitungan gradien dan mendiagnosis training loop. Oleh karena itu, train-validation gap tidak dapat dianalisis. Evaluasi dilakukan menggunakan seluruh data XOR yang tersedia untuk proses training dan pemeriksaan prediksi akhir sesuai dengan protokol modul.

## 6. Jawaban Pertanyaan Modul

1. **[Mengapa relative error tidak pernah persis nol, dan berapa besar nilai yang masih Anda anggap wajar?]**  
   [Relative error tidak selalu bernilai nol karena gradien numerik dihitung menggunakan pendekatan selisih hingga (finite difference) dan dipengaruhi oleh keterbatasan presisi floating-point. Pada eksperimen, nilai relative error maksimum adalah \(1.122192\times10^{-10}\), sehingga masih jauh di bawah batas toleransi \(10^{-5}\) dan dianggap wajar.]

2. **[Apa yang terjadi pada tabel relative error bila ε diubah menjadi 10−9? Jelaskan penyebabnya.]**  
   [Ketika \(\epsilon\) diperkecil menjadi \(10^{-9}\), relative error dapat menjadi lebih besar atau kurang stabil. Hal ini terjadi karena selisih nilai fungsi yang sangat kecil lebih mudah dipengaruhi oleh pembulatan dan keterbatasan presisi floating-point, sehingga akurasi gradien numerik dapat menurun.]

3. **[Pada langkah mana gradien contoh pertama dan kedua bergabung ketika memakai batch, dan mengapa penjumlahan itu benar?]**  
   [Gradien dari contoh pertama dan kedua bergabung ketika proses `backward()` menghitung gradien terhadap batch loss. Jika loss batch dihitung sebagai rata-rata loss tiap contoh, maka gradien batch merupakan penjumlahan kontribusi gradien dari masing-masing contoh yang kemudian dibagi dengan jumlah contoh. Pada eksperimen dua contoh, diperoleh average loss sebesar \(0.1401515061\) dan \(dW_2=[-0.05689364,\ 0.14449643]\).]

4. **[Dari keempat kesalahan Bagian E, mana yang paling sulit ditemukan tanpa membandingkan angka? Jelaskan alasannya.]**  
   [Kesalahan penggunaan `Sigmoid` sebelum `BCEWithLogitsLoss` merupakan salah satu yang paling mudah terlewat karena kode tetap terlihat masuk akal dan dapat dijalankan tanpa error. Dampaknya lebih mudah terlihat melalui perubahan loss dan gradien dibandingkan hanya dari struktur kode.]

5. **[Apa perbedaan peran backpropagation dan optimizer? Jawab dalam maksimal tiga kalimat.]**  
   [Backpropagation digunakan untuk menghitung gradien loss terhadap setiap parameter jaringan menggunakan aturan rantai. Optimizer menggunakan gradien tersebut untuk memperbarui nilai parameter agar loss dapat diminimalkan. Jadi, backpropagation menentukan arah perubahan parameter, sedangkan optimizer melakukan pembaruannya.]

## 7. Kesimpulan dan Keterbatasan

### 7.1 Kesimpulan

Praktikum ini berhasil menerapkan perhitungan gradien secara manual dan membandingkannya dengan autograd serta gradien numerik. Hasil gradient checking menunjukkan relative error maksimum sebesar \(1.122\times10^{-10}\), sehingga hasil perhitungan gradien berada di bawah batas toleransi \(10^{-5}\). Pada kasus XOR, perbaikan training loop secara bertahap menghasilkan loss akhir sebesar \(0.07280489\) dengan prediksi \([0,1,1,0]\) yang sesuai dengan target. Hasil tersebut menunjukkan bahwa backpropagation, autograd, dan proses pembaruan parameter telah berjalan sesuai dengan konsep yang dipelajari.

### 7.2 Keterbatasan

- Eksperimen menggunakan satu seed, yaitu seed = 4, sehingga kestabilan hasil terhadap variasi seed belum dapat dianalisis.
- Dataset XOR hanya terdiri dari 4 data, sehingga hasil training digunakan untuk memverifikasi proses pembelajaran dan bukan untuk menilai kemampuan generalisasi pada dataset yang lebih besar.
- Tidak terdapat train-validation split, sehingga validation loss dan train-validation gap tidak dapat dianalisis.

### 7.3 Tindak lanjut

Jika tersedia tambahan waktu atau komputasi, eksperimen yang paling bernilai adalah mengulangi *training* dengan beberapa nilai seed yang berbeda. Hal ini dapat digunakan untuk melihat apakah hasil loss dan prediksi XOR tetap konsisten atau dipengaruhi oleh inisialisasi parameter, sehingga kestabilan hasil eksperimen dapat dianalisis dengan lebih baik.


## Referensi

1. Modul Praktikum 02, Backpropagation dan Automatic Differentiation, notebook praktikum dengan seed 1004. 
2. Dokumentasi dan library yang digunakan dalam notebook: NumPy, Pandas, Scikit-learn, dan PyTorch. 


## Pernyataan Orisinalitas

Saya menyatakan bahwa kode, eksperimen, analisis, dan laporan ini merupakan
pekerjaan individual. Semua sumber eksternal, termasuk potongan kode, telah
dicantumkan. Saya memahami bahwa kemiripan hasil akibat seed atau data yang
sama tidak membenarkan penyalinan notebook maupun analisis.

**Nama:** [ISI NAMA]  
**Tanggal:** [YYYY-MM-DD]

---

## Checklist Sebelum Mengumpulkan

> Hapus bagian petunjuk dan contoh yang tidak diperlukan, tetapi checklist ini
> boleh dipertahankan pada berkas Markdown. Checklist tidak perlu muncul pada
> PDF jika batas halaman ketat.

- [ ] Nama berkas adalah `MXX_NIM.md` dan `MXX_NIM.pdf`.
- [ ] Identitas, seed, device, dan versi library telah diisi.
- [ ] Isi laporan tidak melebihi batas halaman modul.
- [ ] Protokol sama dengan modul atau setiap perubahan telah dijelaskan.
- [ ] Tabel hasil konsisten dengan `MXX_NIM_metrics.csv`.
- [ ] Seluruh run dicantumkan, termasuk run yang gagal atau buruk.
- [ ] Grafik memiliki judul, label sumbu, legenda, dan caption.
- [ ] Setiap klaim utama disertai angka atau rujukan gambar/tabel.
- [ ] Test set tidak digunakan untuk memilih model atau hyperparameter.
- [ ] Kesimpulan menjawab tujuan dan menyebutkan trade-off.
- [ ] Sumber eksternal telah dicantumkan.
- [ ] Pernyataan orisinalitas telah diisi.
- [ ] Notebook lolos **Restart Kernel and Run All**.
- [ ] Tiga berkas pengumpulan (`ipynb`, `pdf`, dan `metrics.csv`) tersedia.
