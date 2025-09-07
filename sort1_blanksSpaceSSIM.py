from pdf2image import convert_from_path
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import tempfile
import os
from tqdm import tqdm

def pdf2img(pdf_path, dpi=150):
    pages = convert_from_path(pdf_path, dpi=dpi)
    opencv_images = []
    for page in pages:
        img = np.array(page)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        opencv_images.append(img)
    return opencv_images

def resize_to_dpi(imgs, src_dpi=150, dst_dpi=30):
    resized_images = []
    for img in imgs:
        scale = dst_dpi / src_dpi
        new_w = int(img.shape[1] * scale)
        new_h = int(img.shape[0] * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_images.append(resized)
    return resized_images

def crop_content(img, thresh_val=250):
    """
    余白を除去して中身だけを切り出す
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img  # 真っ白ならそのまま返す
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def img_ssim(img1, img2):
    """
    白くない部分だけを対象にSSIMを計算
    """
    crop1 = crop_content(img1)
    crop2 = crop_content(img2)

    gray1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape
    gray2 = cv2.resize(gray2, (w, h))

    score, _ = ssim(gray1, gray2, full=True)
    return score

def build_score_matrix(imgs1_low, imgs2_low, imgs1_high, imgs2_high, top_k=5):
    matrix = np.zeros((len(imgs1_low), len(imgs2_low)))
    for i, img1_low in enumerate(tqdm(imgs1_low, desc="スコア計算")):
        scores = [(j, img_ssim(img1_low, img2_low)) for j, img2_low in enumerate(imgs2_low)]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scores[:top_k]
        for j, _ in top_candidates:
            score = img_ssim(imgs1_high[i], imgs2_high[j])
            matrix[i, j] = score
    return matrix

def pair_unique(matrix, threshold=0.5):
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

def save_pairs_to_pdf(pairs, imgs1_high, imgs2_high, output_pdf="paired_blanksSpace_SSIM.pdf"):
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
            label = f"Pair: img1[{p1}] - img2[{p2 if p2 is not None else 'blank'}], SSIM={score:.4f}"
            c.drawString(50, page_height/2 - 20, label)
        os.remove(tmp1.name); os.remove(tmp2.name)
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
            os.remove(tmp1.name); os.remove(tmp2.name)
            c.showPage()

    c.save()
    print(f"Saved paired PDF as {output_pdf}")

def main():
    pdf1_path = "壁量計算書-1(補正前).pdf"
    pdf2_path = "壁量計算書-2（補正後）.pdf"

    print("[1] PDF読み込み中...")
    imgs1_high = pdf2img(pdf1_path, dpi=150)
    imgs2_high = pdf2img(pdf2_path, dpi=150)
    imgs1_low = resize_to_dpi(imgs1_high, src_dpi=150, dst_dpi=30)
    imgs2_low = resize_to_dpi(imgs2_high, src_dpi=150, dst_dpi=30)

    print("[2] スコアマトリクス構築...")
    matrix = build_score_matrix(imgs1_low, imgs2_low, imgs1_high, imgs2_high, top_k=5)

    print("[3] ペアリング...")
    pairs = pair_unique(matrix, threshold=0.5)

    print("[4] PDF出力...")
    save_pairs_to_pdf(pairs, imgs1_high, imgs2_high, output_pdf="paired_blanksSpace_ssim.pdf")

if __name__ == "__main__":
    main()
