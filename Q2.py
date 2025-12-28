!sudo apt update
!sudo apt install tesseract-ocr
!pip install pytesseract

import pytesseract
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from google.colab import files

#上傳手寫檔
print("請上傳手寫英文單字圖檔：")
uploaded = files.upload()

for fn in uploaded.keys():
    #影像預處理
    img = Image.open(fn).convert('L')
    
    #自動對比增強
    img = ImageOps.autocontrast(img)
    
    #使用Tesseract進行辨識
    # lang='eng'表示英文,--psm 7適合單行文字
    custom_config = r'--oem 3 --psm 7'
    recognition_text = pytesseract.image_to_string(img, lang='eng', config=custom_config)

    #顯示結果
    plt.figure(figsize=(6, 3))
    plt.imshow(img, cmap='gray')
    plt.title(f"AI Sees This -> Pred: {recognition_text.strip()}", fontsize=16, color='blue')
    plt.axis('off')
    plt.show()

    print(f"檔案名稱: {fn}")
    print(f"OCR 辨識結果: {recognition_text.strip()}")
    print("-" * 30)