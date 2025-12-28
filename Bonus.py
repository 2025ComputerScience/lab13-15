import pytesseract
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import requests
from io import BytesIO

#安裝繁體中文字庫
!sudo apt install tesseract-ocr-chi-tra

#取得作業指定的賀卡圖片
img_url = "https://hackmd.io/_uploads/Bkc0Sw8mbg.png"
response = requests.get(img_url)
img = Image.open(BytesIO(response.content)).convert('L')

#影像預處理
img = ImageOps.autocontrast(img)

#進行中英文夾雜辨識
#lang='chi_tra+eng'同時偵測繁體中文與英文
custom_config = r'--oem 3 --psm 6'
recognition_text = pytesseract.image_to_string(img, lang='chi_tra+eng', config=custom_config)

#顯示圖片與辨識結果
plt.figure(figsize=(8, 5))
plt.imshow(img, cmap='gray')
plt.title("AI Sees This (Mixed Language OCR)", fontsize=14)
plt.axis('off')
plt.show()

print("OCR 辨識結果輸出：")
print("-" * 40)
print(recognition_text.strip())