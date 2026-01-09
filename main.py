import io
import asyncio
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from inference import ModelInference
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from typing import List


app = FastAPI(title="AI Inference API")

executor = ThreadPoolExecutor(max_workers=4)
model = ModelInference() # 初始化模型


def inference(img: Image.Image):
    result = model.predict(img)
    return result


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """單張圖片推論"""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 使用多線程執行 blocking 推論
    loop = asyncio.get_running_loop()

    result = await loop.run_in_executor(
        executor,
        inference,
        img,
    )

    return {
        "filename": file.filename,
        "label": result,
    }


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """多張圖片推論"""
    images = []
    filenames = []

    # 先把每張圖片讀進 memory
    for file in files:
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert("L")
            images.append(img)
            filenames.append(file.filename)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")

    loop = asyncio.get_running_loop()

    # 多線程平行推論
    tasks = [
        loop.run_in_executor(executor, inference, img)
        for img in images
    ]
    results = await asyncio.gather(*tasks)

    # 對應 filename 回傳
    return [{"filename": fn, "label": res} for fn, res in zip(filenames, results)]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
