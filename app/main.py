from fastapi import FastAPI

# Importing routers 
from app.api.routes import health, version, predict


def create_app() -> FastAPI:
    app = FastAPI(
        title="Thendral Core API",
        description="Phase 1 - ML Intelligence Service for Thendral",
        version="0.1.0"
    )

    # Register routes
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(version.router, prefix="/version", tags=["Version"])
    app.include_router(predict.router, prefix="/predict", tags=["Prediction"])

    return app


app = create_app()


# Startup event 
@app.on_event("startup")
def startup_event():
    print("Thendral Core API is starting...")
    print("API is ready to use")


# Shutdown event 
@app.on_event("shutdown")
def shutdown_event():
    print("Thendral Core API is shutting down...")
    print("Adios!")