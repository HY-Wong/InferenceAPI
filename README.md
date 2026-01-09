# AI Inference API (CNN-Transformer)

本專案提供一套基於 FastAPI 的高效能 RESTful API，用於部署 CNN-Transformer 影像分類模型。支援多線程並發、Docker 容器化部署，並附帶批次預測工具。

---

## 本地啟動 API
使用 uvicorn 且支援多線程並發請求：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

## Docker 部署 (推薦)
透過容器化技術，確保在不同伺服器環境下均能正常執行：

```bash
# 建立 Image
docker build -t ai-inference-service .

# 啟動並對外開放 8000
docker run -p 8000:8000 ai-inference-service
```

## API 規格與測試

### [POST] /predict
上傳單張圖片獲取模型預測結果。

### 測試工具：

- Swagger UI: 啟動後訪問 http://localhost:8000/docs。

- Postman: 建立 POST 請求，在 Body 選擇 form-data 上傳圖片。

### 成功回應範例
**Response (JSON)**:
```json
{
  "filename": "img_0_label_6.png",
  "label": 6
}
```

## 批次預測
針對 test/ 資料夾內的所有圖片進行推論，並產出結果清單（read.csv）：

```bash
python batch_predict.py
```