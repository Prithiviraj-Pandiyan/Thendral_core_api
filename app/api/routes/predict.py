from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def predict_endpoint():
    return {"Thendral core API Prediction will be coming soon"}
