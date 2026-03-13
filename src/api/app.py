"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import admin, health, picks, predictions, performance, teams

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenBet - Football Prediction Engine",
        version="0.1.0",
        description="Professional football match prediction and betting analysis API",
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(picks.router, prefix="/picks", tags=["Picks"])
    app.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
    app.include_router(teams.router, prefix="/teams", tags=["Teams"])
    app.include_router(performance.router, prefix="/performance", tags=["Performance"])
    app.include_router(admin.router, prefix="/admin", tags=["Admin"])

    # Serve static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(str(STATIC_DIR / "index.html"))

    return app


app = create_app()
