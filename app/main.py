from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import traceback

from app.infer import ModelService


# Path ke model dan class mapping
MODEL_PATH = "artifacts/model_best.pth"
CLASS_MAP = "app/classes.json"

# Load model sekali saat server start
model_service = ModelService(
    model_path=MODEL_PATH,
    class_map_path=CLASS_MAP
)

app = FastAPI(
    title="GERD & Polyp Classification API",
    description="ConvNeXt-Tiny model for endoscopy image classification",
    version="1.0.0"
)


@app.get("/")
def home():
    return {"message": "API is running. Use POST /predict-image to get predictions."}


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read file bytes
        contents = await file.read()

        # Convert to PIL image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Predict
        result = model_service.predict(image)

        return JSONResponse(content={
            "filename": file.filename,
            "predicted_class": result["pred_class"],
            "confidence": result["confidence"]
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
