import io
import uvicorn
import threading
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from inference import ModelInference
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


app = FastAPI(title="AI Inference API")

executor = ThreadPoolExecutor(max_workers=4)
model = ModelInference() # 初始化模型


def inference(img: Image.Image):
    thread_id = threading.get_ident()
    print(f"[Thread {thread_id}] Start inference")

    time.sleep(5)  # 模擬耗時推論

    print(f"[Thread {thread_id}] End inference")

    result = model.predict(img)
    return result


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    上傳圖片並回傳預測結果
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 使用多線程執行 blocking 推論
    loop = None
    try:
        import asyncio
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    result = await loop.run_in_executor(
        executor,
        inference,
        img,
    )

    return {
        "filename": file.filename,
        "label": result,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
