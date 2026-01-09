import os
import pandas as pd

from inference import ModelInference
from PIL import Image


def run_batch(test_dir):
    model = ModelInference()
    results = []
    
    for filename in os.listdir(test_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(test_dir, filename)
            img = Image.open(img_path).convert("L")
            prediction = model.predict(img)
            results.append({"ImageId": filename, "Label": prediction})
    
    # 輸出 CSV
    df = pd.DataFrame(results)
    df.to_csv("result.csv", index=False)
    print("Batch prediction completed. Saved to result.csv")


if __name__ == "__main__":
    run_batch("test/")