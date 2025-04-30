import os
import csv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# =============================================
# 1. STEMMING BAHASA INDONESIA (Nazief & Adriani via Sastrawi)
# =============================================
def stem_indonesian(input_folder, output_folder):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()  # Menggunakan Nazief & Adriani

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"stemmed_{filename}")

            try:
                with open(input_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            except UnicodeDecodeError:
                with open(input_path, 'r', encoding='ISO-8859-1') as file:
                    text = file.read()

            stemmed_text = stemmer.stem(text)

            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(stemmed_text)

            print(f"Stemming selesai: {filename} → stemmed_{filename}")

# ============================================
# 2. STEMMING BAHASA INGGRIS (Porter only)
# ============================================
def stem_english(input_folder, output_folder):
    stemmer = PorterStemmer()  # Hanya menggunakan Porter

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"stemmed_{filename}")

            with open(input_path, 'r', encoding='utf-8') as file:
                text = file.read()

            tokens = word_tokenize(text)
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            stemmed_text = ' '.join(stemmed_tokens)

            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(stemmed_text)

            print(f"Stemming selesai: {filename} → stemmed_{filename}")

# ============================================
# 3. ANALISIS PER DOKUMEN (UI, OI, MWC) + CSV
# ============================================
def analyze_and_export_to_csv(original_folder, stemmed_folder, language, output_csv):
    data = []
    total_ui = 0
    total_oi = 0
    total_mwc = 0
    doc_count = 0

    for filename in os.listdir(original_folder):
        if filename.endswith(".txt"):
            original_path = os.path.join(original_folder, filename)
            stemmed_path = os.path.join(stemmed_folder, f"stemmed_{filename}")

            try:
                with open(original_path, 'r', encoding='utf-8') as file:
                    original_text = file.read().split()
            except UnicodeDecodeError:
                with open(original_path, 'r', encoding='ISO-8859-1') as file:
                    original_text = file.read().split()

            try:
                with open(stemmed_path, 'r', encoding='utf-8') as file:
                    stemmed_text = file.read().split()
            except UnicodeDecodeError:
                with open(stemmed_path, 'r', encoding='ISO-8859-1') as file:
                    stemmed_text = file.read().split()

            min_len = min(len(original_text), len(stemmed_text))
            if min_len == 0:
                continue

            original_text = original_text[:min_len]
            stemmed_text = stemmed_text[:min_len]

            ui = sum(1 for orig, stem in zip(original_text, stemmed_text) if orig == stem)
            oi = sum(1 for orig, stem in zip(original_text, stemmed_text) if stem not in orig)
            mwc = sum(abs(len(orig) - len(stem)) for orig, stem in zip(original_text, stemmed_text))

            ui_percent = (ui / min_len) * 100
            oi_percent = (oi / min_len) * 100
            avg_mwc = mwc / min_len

            data.append({
                "Dokumen": filename,
                "UI (%)": round(ui_percent, 2),
                "OI (%)": round(oi_percent, 2),
                "MWC": round(avg_mwc, 2)
            })

            total_ui += ui_percent
            total_oi += oi_percent
            total_mwc += avg_mwc
            doc_count += 1

    # Tambahkan baris rata-rata
    if doc_count > 0:
        avg_ui = total_ui / doc_count
        avg_oi = total_oi / doc_count
        avg_mwc = total_mwc / doc_count

        data.append({
            "Dokumen": "RATA-RATA",
            "UI (%)": round(avg_ui, 2),
            "OI (%)": round(avg_oi, 2),
            "MWC": round(avg_mwc, 2)
        })

    # Tulis ke CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Dokumen", "UI (%)", "OI (%)", "MWC"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"\nHasil analisis + rata-rata disimpan di: {output_csv}")


# ============================================
# 4. JALANKAN SEMUA
# ============================================
if __name__ == "__main__":
    # Stemming Bahasa Indonesia
    print("Memulai stemming Bahasa Indonesia...")
    stem_indonesian(
        input_folder="indonesian_docs",
        output_folder="results/indonesian_stemmed"
    )

    # Stemming Bahasa Inggris
    print("\nMemulai stemming Bahasa Inggris...")
    stem_english(
        input_folder="english_docs",
        output_folder="results/english_stemmed"
    )

    # Analisis dan Export CSV
    print("\nMenganalisis dan mengekspor hasil...")
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
