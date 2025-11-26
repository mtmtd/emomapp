from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import numpy as np
from PIL import Image
import pillow_heif   # HEIC対応ライブラリ
import uuid
import joblib        # ランダムフォレストモデル読み込み用

# PillowにHEIFサポートを登録
pillow_heif.register_heif_opener()

app = FastAPI()

# 静的ファイル（CSSや診断表画像など）
app.mount("/static", StaticFiles(directory="static"), name="static")

# アップロード画像を公開
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Jinja2テンプレート設定
templates = Jinja2Templates(directory="templates")

# モデル読み込み（ランダムフォレスト）
model_x = joblib.load("model_x.pkl")
model_y = joblib.load("model_y.pkl")

# -------------------------------
# 診断用関数群
# -------------------------------

def custom_round(value):
    rounded = round(value, 1)
    if rounded <= -2.5: return -3
    elif -2.5 < rounded <= -1.5: return -2
    elif -1.5 < rounded <= -0.1: return -1
    elif -0.1 < rounded <= 1.4: return 1
    elif 1.4 < rounded <= 2.4: return 2
    elif rounded > 2.4: return 3
    else: return 1

def classify_6region(x, y):
    if x >= 2:
        return "未来ポジティブ" if y > 0 else "未来ネガティブ" if y < 0 else "未来中立"
    elif -1 <= x <= 1:
        return "現在ポジティブ" if y > 0 else "現在ネガティブ" if y < 0 else "現在中立"
    elif x <= -2:
        return "過去ポジティブ" if y > 0 else "過去ネガティブ" if y < 0 else "過去中立"
    else:
        return "分類不能"

# 感性語辞書（完全版）
kansengo_dict = {
    (-3, -3): ["絶望", "恐怖", "悲劇"], (-3, -2): ["後悔", "苦い記憶", "屈辱"], (-3, -1): ["切ない", "物悲しい", "哀愁"],
    (-3, 1): ["思い出", "懐かしさ", "優しさ"], (-3, 2): ["ほっとする", "癒される", "ふるさと"], (-3, 3): ["懐かしい", "微笑ましい", "優しい"],
    (-2, -3): ["喪失", "崩壊", "絶体絶命"], (-2, -2): ["屈折", "恨み", "悔しさ"], (-2, -1): ["寂しさ", "胸苦しさ", "胸が痛む"],
    (-2, 1): ["涙", "静けさ", "柔らかい"], (-2, 2): ["安らぎ", "やさしさ", "包まれる"], (-2, 3): ["思い出深い", "しみじみ", "穏やか"],
    (-1, -3): ["深い悲しみ", "絶望的", "闇"], (-1, -2): ["孤独", "心配", "閉塞感"], (-1, -1): ["感傷的", "涙ぐむ", "やるせない"],
    (-1, 1): ["静か", "控えめな幸福", "静かな喜び"], (-1, 2): ["安心", "和やか", "満たされる"], (-1, 3): ["ノスタルジー", "郷愁", "情緒的"],
    (1, -3): ["虚無", "混乱", "崩れ落ちる"], (1, -2): ["無力感", "疲弊", "空虚"], (1, -1): ["不安", "焦り", "違和感"],
    (1, 1): ["落ち着く", "平穏", "自然体"], (1, 2): ["安定", "信頼", "心地よい"], (1, 3): ["感動", "感激", "胸が熱くなる"],
    (2, -3): ["絶望感", "破滅", "終焉"], (2, -2): ["焦燥", "恐れ", "動揺"], (2, -1): ["迷い", "不確か", "揺らぎ"],
    (2, 1): ["興味", "関心", "好奇心"], (2, 2): ["期待", "楽しみ", "前向き"], (2, 3): ["ワクワク", "高揚", "胸躍る"],
    (3, -3): ["滅亡", "無価値", "未来喪失感"], (3, -2): ["不信", "諦め", "破綻の予感"], (3, -1): ["漠然とした不安", "予測不能", "緊張感"],
    (3, 1): ["予感", "兆し", "可能性の芽"], (3, 2): ["成長", "挑戦", "飛躍", "未来志向"], (3, 3): ["希望", "夢", "可能性", "輝き"]
}

def get_kansengo_for_6region(region):
    words = []
    if region == "過去ポジティブ":
        for x in [-3, -2]:
            for y in [1, 2, 3]:
                words.extend(kansengo_dict.get((x, y), []))
    elif region == "過去ネガティブ":
        for x in [-3, -2]:
            for y in [-3, -2, -1]:
                words.extend(kansengo_dict.get((x, y), []))
    elif region == "現在ポジティブ":
        for x in [-1, 1]:
            for y in [1, 2, 3]:
                words.extend(kansengo_dict.get((x, y), []))
    elif region == "現在ネガティブ":
        for x in [-1, 1]:
            for y in [-3, -2, -1]:
                words.extend(kansengo_dict.get((x, y), []))
    elif region == "未来ポジティブ":
        for x in [2, 3]:
            for y in [1, 2, 3]:
                words.extend(kansengo_dict.get((x, y), []))
    elif region == "未来ネガティブ":
        for x in [2, 3]:
            for y in [-3, -2, -1]:
                words.extend(kansengo_dict.get((x, y), []))
    return words if words else ["該当なし"]

# -------------------------------
# ルーティング
# -------------------------------

@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/diagnose", response_class=HTMLResponse)
async def diagnose(request: Request, image: UploadFile):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # 常にJPEGで保存する安全なファイル名を生成
    safe_filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(upload_dir, safe_filename)

    # Pillowで読み込み（HEICも対応）→ JPEGに変換して保存
    pil_img = Image.open(image.file).convert("RGB")
    pil_img.save(file_path, "JPEG")

    # モデル入力用にリサイズ
    pil_img = pil_img.resize((256, 256))
    img_array = np.array(pil_img) / 255.0

    # flattenして特徴量ベクトル化（学習時と同じ処理に合わせる）
    features = img_array.flatten()

    # RFモデルで予測
    pred_x = model_x.predict([features])[0]
    pred_y = model_y.predict([features])[0]

    # 座標を丸めて分類
    x_label = custom_round(pred_x)
    y_label = custom_round(pred_y)
    region = classify_6region(x_label, y_label)
    kansengo = get_kansengo_for_6region(region)

    # 結果をテンプレートに渡す
    return templates.TemplateResponse("index.html", {
        "request": request,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "x_label": x_label,
        "y_label": y_label,
        "region": region,
        "kansengo": kansengo,
        "image_filename": safe_filename   # JPEG化したファイル名を渡す
    })


# アップロード画像一覧ページ
@app.get("/uploads-list", response_class=HTMLResponse)
def uploads_list(request: Request):
    upload_dir = "uploads"
    files = os.listdir(upload_dir) if os.path.exists(upload_dir) else []
    # 画像ファイルだけに絞る（主要形式＋HEICも含める）
    image_files = [
        f for f in files if f.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif")
        )
    ]

    return templates.TemplateResponse("uploads.html", {
        "request": request,
        "image_files": image_files
    })


# -------------------------------
# アプリ起動
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)