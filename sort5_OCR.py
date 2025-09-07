from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import tempfile
import os
from tqdm import tqdm

# --------------------
# PDF → OpenCV画像
# --------------------
def pdf2img(pdf_path, dpi=150):
    pages = convert_from_path(pdf_path, dpi=dpi)
    opencv_images = []
    for page in pages:
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        opencv_images.append(img)
    return opencv_images

# --------------------
# OCRで文字抽出
# --------------------
def extract_text(img, min_font_size=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, lang="jpn+eng", output_type=pytesseract.Output.DICT)

    words = []
    for i in range(len(data['text'])):
        w, h = data['width'][i], data['height'][i]
        txt = data['text'][i].strip()
        if len(txt) > 1 and h >= min_font_size:
            words.append(txt)
    return words

# --------------------
# TF-IDF + コサイン類似度
# --------------------
def tfidf_cosine_similarity(words1, words2):
    if not words1 or not words2:
        return 0.0
    docs = [" ".join(words1), " ".join(words2)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(docs)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return float(sim[0][0])

# --------------------
# スコアマトリクス構築（全組み合わせ）
# --------------------
def build_score_matrix(words1_cache, words2_cache):
    num1, num2 = len(words1_cache), len(words2_cache)
    matrix = np.zeros((num1, num2))
    for i in tqdm(range(num1), desc="スコア計算(TF-IDF)"):
        for j in range(num2):
            score = tfidf_cosine_similarity(words1_cache[i], words2_cache[j])
            matrix[i, j] = score
    return matrix

# --------------------
# ペアリング処理
# --------------------
def pair_unique(matrix, threshold=0.1):
    pairs = []
    used_img2 = set()
    num_img1, num_img2 = matrix.shape

    for i in range(num_img1):
        candidates = sorted([(j, matrix[i, j]) for j in range(num_img2)], key=lambda x: x[1], reverse=True)
        chosen = None
        for j, score in candidates:
            if j not in used_img2 and score >= threshold:
                chosen = (j, score)
                used_img2.add(j)
                break
        if chosen:
            pairs.append((i, chosen[0], chosen[1]))
        else:
            pairs.append((i, None, 0.0))
    return pairs

# --------------------
# PDF出力
# --------------------
def save_pairs_to_pdf(pairs, imgs1_high, imgs2_high, output_pdf="paired_ocr.pdf"):
    c = canvas.Canvas(output_pdf, pagesize=A4)
    page_width, page_height = A4
    blank = np.ones_like(imgs1_high[0]) * 255

    num_img1 = len(imgs1_high)
    num_img2 = len(imgs2_high)
    used_img2 = {p[1] for p in pairs if p[1] is not None}

    for (p1, p2, score) in tqdm(pairs, desc="PDF作成"):
        img1 = imgs1_high[p1]
        img2 = imgs2_high[p2] if p2 is not None else blank

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
            cv2.imwrite(tmp1.name, img1)
            cv2.imwrite(tmp2.name, img2)
            c.drawImage(ImageReader(tmp1.name), 0, page_height/2, page_width/2, page_height/2)
            c.drawImage(ImageReader(tmp2.name), page_width/2, page_height/2, page_width/2, page_height/2)
            label = f"Pair: img1[{p1}] - img2[{p2 if p2 is not None else 'blank'}], TF-IDF Score={score:.4f}"
            c.drawString(50, page_height/2 - 20, label)
        os.remove(tmp1.name)
        os.remove(tmp2.name)
        c.showPage()

    for j in range(num_img2):
        if j not in used_img2:
            img1 = blank
            img2 = imgs2_high[j]
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1, \
                 tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
                cv2.imwrite(tmp1.name, img1)
                cv2.imwrite(tmp2.name, img2)
                c.drawImage(ImageReader(tmp1.name), 0, page_height/2, page_width/2, page_height/2)
                c.drawImage(ImageReader(tmp2.name), page_width/2, page_height/2, page_width/2, page_height/2)
                label = f"Unpaired: img1[blank] - img2[{j}]"
                c.drawString(50, page_height/2 - 20, label)
            os.remove(tmp1.name)
            os.remove(tmp2.name)
            c.showPage()

    c.save()
    print(f"Saved paired PDF as {output_pdf}")

# --------------------
# メイン処理
# --------------------
def main():
    pdf1_path = "壁量計算書-1(補正前).pdf"
    pdf2_path = "壁量計算書-2（補正後）.pdf"

    print("[1] PDF読み込み中...")
    imgs1_high = pdf2img(pdf1_path, dpi=150)
    imgs2_high = pdf2img(pdf2_path, dpi=150)

    print("[1.5] OCRキャッシュ作成...")
    words1_cache = [extract_text(img) for img in tqdm(imgs1_high, desc="OCR PDF1")]
    words2_cache = [extract_text(img) for img in tqdm(imgs2_high, desc="OCR PDF2")]

    print("[2] スコアマトリクス構築...")
    matrix = build_score_matrix(words1_cache, words2_cache)

    print("[3] ペアリング...")
    pairs = pair_unique(matrix, threshold=0.1)

    print("[4] PDF出力...")
    save_pairs_to_pdf(pairs, imgs1_high, imgs2_high, output_pdf="paired_ocr.pdf")

if __name__ == "__main__":
    main()
