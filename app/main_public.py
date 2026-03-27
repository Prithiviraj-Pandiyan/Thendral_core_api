from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.predict import router as predict_router
from app.api.routes.version import router as version_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Thendral Public API",
        description="Runtime inference and service health endpoints",
        version="1.0.0",
    )
    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(version_router, prefix="/version", tags=["Version"])
    app.include_router(predict_router, prefix="/predict", tags=["Prediction"])
    return app


app = create_app()


@app.on_event("startup")
def startup_event() -> None:
    print("Thendral Public API is starting...")
    print("Public API is ready")


@app.on_event("shutdown")
def shutdown_event() -> None:
    print("Thendral Public API is shutting down...")
