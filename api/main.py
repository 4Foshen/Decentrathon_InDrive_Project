from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

def check_car_condition(image: Image.Image) -> str:
    return "чистая" if image.size[0] % 2 == 0 else "грязная"

@app.post("/check_car/")
async def check_car(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        status = check_car_condition(image)
        return JSONResponse(content={"status": status})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки файла: {e}")
