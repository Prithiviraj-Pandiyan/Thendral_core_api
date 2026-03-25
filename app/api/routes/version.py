from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def version_check():
    return {"Version" : "0.1.1"}