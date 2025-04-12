from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
from io import BytesIO

app = FastAPI()

model_path = "models/laptop_price_model.pkl"
model = joblib.load(model_path)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(BytesIO(contents))
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}