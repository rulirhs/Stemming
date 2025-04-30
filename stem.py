import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import re

# Kamus sederhana untuk validasi bahasa
english_words = set([
    'the', 'is', 'are', 'run', 'jump', 'cat', 'dog', 'garden', 
    'combine', 'clarify', 'create', 'organize', 'implement', 'review', 
    'analyze', 'explain', 'prepare', 'identify', 'complete', 'propose', 
    'test', 'train', 'develop', 'discuss', 'operate', 'observe', 'perform', 
    'improve', 'evaluate', 'arrange', 'monitor', 'finalize', 'adjust', 
    'conduct', 'integrate', 'interpret', 'conclude', 'plan', 'process', 
    'execute', 'validate', 'compare', 'address', 'confirm', 'measure', 
    'define', 'coordinate', 'resolve', 'participate', 'examine', 'explore', 
    'recommend', 'simplify', 'report', 'introduce', 'establish', 'classify', 
    'describe', 'design', 'record', 'investigate', 'generate', 'focus', 
    'verify', 'provide', 'select', 'study', 'formulate', 'recognize', 'engage', 
    'adjust', 'imagine', 'communicate', 'reflect', 'coordinate', 'elaborate'
])
indonesian_words = set([
    'dan', 'atau', 'kucing', 'anjing', 'taman', 'lari', 'lompat', 
    'makan', 'minum', 'bicara', 'ajar', 'tulis', 'baca', 
    'jalan', 'tidur', 'bangun', 'duduk', 'diri', 'lihat', 'dengar', 
    'masak', 'main', 'miliki', 'suka', 'bantu', 'selesai', 
    'ajar', 'hitung', 'tahu', 'menang', 'kalah', 'pikir', 
    'tari', 'nyanyi', 'kenal', 'kunjung', 'daki', 
    'kumpul', 'jaga', 'susun', 'rancang', 'hargai', 'ikut', 
    'henti', 'tunggu', 'putus', 'pakaian', 'rokok', 'rawat', 
    'guna', 'hindar', 'ungkap', 'komunikasi', 'kenal', 
    'tawar', 'baik', 'dekat', 'hasil', 'cerita', 
    'cari', 'peroleh', 'hapus', 'kenal', 'temu', 
    'baik', 'pindah', 'semangat', 'bayar', 'pinjam', 
    'putus', 'ambil', 'atur', 'organisir', 'hibur', 
    'utama', 'atasi', 'perhati', 'terima', 'kontrol'
])

# Fungsi untuk mendeteksi bahasa
def detect_language(text):
    """
    Mendeteksi bahasa teks berdasarkan kata-kata umum.
    Mengembalikan 'english', 'indonesian', atau 'unknown'.
    """
    words = set(re.sub(r'[^\w\s]', '', text.lower()).split())
    eng_count = len(words.intersection(english_words))
    indo_count = len(words.intersection(indonesian_words))
    if eng_count > indo_count * 1.5:
        return "english"
    elif indo_count > eng_count * 1.5:
        return "indonesian"
    return "unknown"

# =============================================
# 1. STEMMING BAHASA INDONESIA (NAZIEF-ADRIANI via SASTRAWI)
# =============================================
def stem_indonesian(input_folder, output_folder):
    """
    Melakukan stemming pada dokumen Bahasa Indonesia menggunakan algoritma Nazief-Adriani (Sastrawi).
    Input: Folder dengan file .txt.
    Output: File stemmed di folder output.
    """
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()  # Menggunakan Nazief-Adriani Stemmer
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"stemmed_{filename}")
            
            encodings = ['utf-8', 'ISO-8859-1', 'windows-1252']
            text = None
            for encoding in encodings:
                try:
                    with open(input_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    print(f"Berhasil membaca {input_path} dengan encoding {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                print(f"Gagal membaca {input_path} dengan semua encoding yang dicoba")
                continue
            
            # Validasi bahasa
            detected_lang = detect_language(text)
            if detected_lang != "indonesian":
                print(f"Peringatan: File {input_path} tampaknya dalam bahasa {detected_lang}, diharapkan indonesian")
            
            stemmed_text = stemmer.stem(text)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(stemmed_text)
            
            print(f"Stemming selesai: {filename} → stemmed_{filename}")

# ============================================
# 2. STEMMING BAHASA INGGRIS (PORTER STEMMER)
# ============================================
def stem_english(input_folder, output_folder):
    stemmer = PorterStemmer() 
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"stemmed_{filename}")
            
            encodings = ['utf-8', 'ISO-8859-1', 'windows-1252']
            text = None
            for encoding in encodings:
                try:
                    with open(input_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    print(f"Berhasil membaca {input_path} dengan encoding {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                print(f"Gagal membaca {input_path} dengan semua encoding yang dicoba")
                continue
            
            # Validasi bahasa
            detected_lang = detect_language(text)
            if detected_lang != "english":
                print(f"Peringatan: File {input_path} tampaknya dalam bahasa {detected_lang}, diharapkan english")
            
            tokens = word_tokenize(text)
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            stemmed_text = ' '.join(stemmed_tokens)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(stemmed_text)
            
            print(f"Stemming selesai: {filename} → stemmed_{filename}")

# ============================================
# 4. PLOT METRIK EVALUASI
# ============================================
def plot_metrics(csv_file, language, output_png):
    """
    Membuat bar plot untuk metrik UI, OI, dan MWC per dokumen.
    Menambahkan garis rata-rata untuk setiap metrik.
    """
    # Baca CSV
    df = pd.read_csv(csv_file)
    
    # Hapus baris AVERAGE untuk plot per dokumen
    df_docs = df[df['Dokumen'] != 'AVERAGE']
    
    # Ambil data
    documents = df_docs['Dokumen'].tolist()
    ui_values = df_docs['UI (%)'].tolist()
    oi_values = df_docs['OI (%)'].tolist()
    mwc_values = df_docs['MWC'].tolist()
    
    # Ambil rata-rata dari baris AVERAGE
    avg_row = df[df['Dokumen'] == 'AVERAGE']
    avg_ui = avg_row['UI (%)'].iloc[0] if not avg_row.empty else 0
    avg_oi = avg_row['OI (%)'].iloc[0] if not avg_row.empty else 0
    avg_mwc = avg_row['MWC'].iloc[0] if not avg_row.empty else 0
    
    # Setup plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(documents))
    width = 0.25
    
    # Plot batang
    plt.bar(x - width, ui_values, width, label='UI (%)', color='skyblue')
    plt.bar(x, oi_values, width, label='OI (%)', color='salmon')
    plt.bar(x + width, mwc_values, width, label='MWC', color='lightgreen')
    
    # Tambahkan garis rata-rata
    plt.axhline(y=avg_ui, color='blue', linestyle='--', label=f'Avg UI ({avg_ui:.2f}%)')
    plt.axhline(y=avg_oi, color='red', linestyle='--', label=f'Avg OI ({avg_oi:.2f}%)')
    plt.axhline(y=avg_mwc, color='green', linestyle='--', label=f'Avg MWC ({avg_mwc:.2f})')
    
    # Kustomisasi plot
    plt.xlabel('Dokumen')
    plt.ylabel('Nilai')
    plt.title(f'Metrik Stemming - {language.capitalize()}')
    plt.xticks(x, documents, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Simpan plot
    plt.savefig(output_png, bbox_inches='tight')
    plt.close()
    print(f"Plot disimpan di: {output_png}")

def plot_average_comparison(indo_csv, eng_csv, output_png):
    """
    Membuat bar plot untuk membandingkan rata-rata UI, OI, dan MWC antara Bahasa Indonesia dan Inggris.
    """
    # Baca CSV
    indo_df = pd.read_csv(indo_csv)
    eng_df = pd.read_csv(eng_csv)
    
    # Ambil baris AVERAGE
    indo_avg = indo_df[indo_df['Dokumen'] == 'AVERAGE']
    eng_avg = eng_df[eng_df['Dokumen'] == 'AVERAGE']
    
    # Ambil nilai rata-rata
    metrics = ['UI (%)', 'OI (%)', 'MWC']
    indo_values = [indo_avg[metric].iloc[0] if not indo_avg.empty else 0 for metric in metrics]
    eng_values = [eng_avg[metric].iloc[0] if not eng_avg.empty else 0 for metric in metrics]
    
    # Setup plot
    plt.figure(figsize=(8, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    # Plot batang
    plt.bar(x - width/2, indo_values, width, label='Indonesia', color='skyblue')
    plt.bar(x + width/2, eng_values, width, label='Inggris', color='salmon')
    
    # Kustomisasi plot
    plt.xlabel('Metrik')
    plt.ylabel('Nilai Rata-rata')
    plt.title('Perbandingan Rata-rata Metrik Stemming')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    
    # Simpan plot
    plt.savefig(output_png, bbox_inches='tight')
    plt.close()
    print(f"Plot perbandingan rata-rata disimpan di: {output_png}")

# ============================================
# 3. ANALISIS PER DOKUMEN (UI, OI, MWC) + CSV + PLOT
# ============================================
def analyze_and_export_to_csv(original_folder, stemmed_folder, language, output_csv):
    """
    Menganalisis stemming dengan metrik UI (Understemming Index), OI (Overstemming Index), dan MWC (Mean Word Count).
    Menghitung rata-rata metrik per bahasa, menyimpan hasil ke CSV, dan membuat plot.
    Menggunakan heuristik tanpa kamus untuk evaluasi.
    """
    data = []
    
    for filename in os.listdir(original_folder):
        if filename.endswith(".txt"):
            original_path = os.path.join(original_folder, filename)
            stemmed_path = os.path.join(stemmed_folder, f"stemmed_{filename}")
            
            encodings = ['utf-8', 'ISO-8859-1', 'windows-1252']
            original_text = None
            stemmed_text = None
            
            # Baca teks asli
            for encoding in encodings:
                try:
                    with open(original_path, 'r', encoding=encoding) as file:
                        original_text = file.read().split()
                    break
                except UnicodeDecodeError:
                    continue
            if original_text is None:
                print(f"Gagal membaca {original_path} dengan semua encoding yang dicoba")
                continue
            
            # Baca teks stemmed
            for encoding in encodings:
                try:
                    with open(stemmed_path, 'r', encoding=encoding) as file:
                        stemmed_text = file.read().split()
                    break
                except UnicodeDecodeError:
                    continue
            if stemmed_text is None:
                print(f"Gagal membaca {stemmed_path} dengan semua encoding yang dicoba")
                continue
            
            # Sesuaikan panjang
            min_len = min(len(original_text), len(stemmed_text))
            original_text = original_text[:min_len]
            stemmed_text = stemmed_text[:min_len]
            
            # Hitung UI dan OI menggunakan heuristik
            ui = 0
            oi = 0
            for orig, stem in zip(original_text, stemmed_text):
                # UI: Kata yang tidak berubah
                if orig == stem:
                    ui += 1
                # OI: Hasil stemming terlalu pendek (< 3 karakter)
                if len(stem) < 3 and stem != orig:
                    oi += 1
            
            mwc = sum(abs(len(orig) - len(stem)) for orig, stem in zip(original_text, stemmed_text))
            
            ui_percent = (ui / min_len) * 100 if min_len > 0 else 0
            oi_percent = (oi / min_len) * 100 if min_len > 0 else 0
            avg_mwc = mwc / min_len if min_len > 0 else 0
            
            data.append({
                "Dokumen": filename,
                "UI (%)": round(ui_percent, 2),
                "OI (%)": round(oi_percent, 2),
                "MWC": round(avg_mwc, 2)
            })
    
    # Hitung rata-rata
    avg_ui = sum(d["UI (%)"] for d in data) / len(data) if data else 0
    avg_oi = sum(d["OI (%)"] for d in data) / len(data) if data else 0
    avg_mwc = sum(d["MWC"] for d in data) / len(data) if data else 0
    
    # Tambahkan baris rata-rata ke data
    data.append({
        "Dokumen": "AVERAGE",
        "UI (%)": round(avg_ui, 2),
        "OI (%)": round(avg_oi, 2),
        "MWC": round(avg_mwc, 2)
    })
    
    # Export ke CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Dokumen", "UI (%)", "OI (%)", "MWC"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    # Tampilkan rata-rata di konsol
    print(f"\nRata-rata untuk {language.upper()}:")
    print(f"Avg UI (%): {avg_ui:.2f}")
    print(f"Avg OI (%): {avg_oi:.2f}")
    print(f"Avg MWC: {avg_mwc:.2f}")
    print(f"Hasil analisis disimpan di: {output_csv}")
    
    # Buat plot metrik per dokumen
    output_png = f"results/{language}_metrics_plot.png"
    plot_metrics(output_csv, language, output_png)

# ============================================
# 5. JALANKAN SEMUA FUNGSI
# ============================================
if __name__ == "__main__":
    # Stemming Bahasa Indonesia (Nazief-Adriani)
    print("Memulai stemming Bahasa Indonesia dengan Nazief-Adriani Stemmer...")
    stem_indonesian(
        input_folder="indonesian_docs",
        output_folder="results/indonesian_stemmed"
    )
    
    # Stemming Bahasa Inggris (Porter Stemmer)
    print("\nMemulai stemming Bahasa Inggris dengan Porter Stemmer...")
    stem_english(
        input_folder="english_docs",
        output_folder="results/english_stemmed"
    )
    
    # Analisis, Export CSV, dan Plot
    print("\nMenganalisis, mengekspor hasil, dan membuat plot...")
    analyze_and_export_to_csv(
        original_folder="indonesian_docs",
        stemmed_folder="results/indonesian_stemmed",
        language="indonesian",
        output_csv="results/analysis_indonesian.csv"
    )
    
    analyze_and_export_to_csv(
        original_folder="english_docs",
        stemmed_folder="results/english_stemmed",
        language="english",
        output_csv="results/analysis_english.csv"
    )
    
    # Plot perbandingan rata-rata
    plot_average_comparison(
        indo_csv="results/analysis_indonesian.csv",
        eng_csv="results/analysis_english.csv",
        output_png="results/average_metrics_comparison.png"
    )