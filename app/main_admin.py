from fastapi import FastAPI

from app.api.routes.datasets import router as datasets_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Thendral Admin API",
        description="Dataset ingestion and model training management endpoints",
        version="1.0.0",
    )
    app.include_router(datasets_router, prefix="/datasets", tags=["Datasets"])
    return app


app = create_app()


@app.on_event("startup")
def startup_event() -> None:
    print("Thendral Admin API is starting...")
    print("Admin API is ready")


@app.on_event("shutdown")
def shutdown_event() -> None:
    print("Thendral Admin API is shutting down...")
