"""
FastAPI Application - Model Serving & Management

Production-ready REST API for:
- Model predictions (single & batch)
- Model registry management
- Monitoring & metrics
- Health checks

Author: Amey Talkatkar
Email: ameytalkatkar169@gmail.com
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
from datetime import datetime

sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for caching
model_cache = {}
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting FastAPI application...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Project: {settings.project_name}")
    
    # Setup directories
    settings.create_directories()
    
    # Pre-load production model (optional)
    # model_cache['production'] = load_production_model()
    
    logger.info("✅ Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application...")
    model_cache.clear()
    logger.info("✅ Cleanup complete")


# Initialize FastAPI app
app = FastAPI(
    title="Sales Forecasting MLOps API",
    description="Production ML API for sales forecasting with MLflow integration",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Root & Health Endpoints ====================

@app.get("/", tags=["Root"])
async def root():
    """
    Welcome message and API information.
    """
    return {
        "message": "Sales Forecasting MLOps API",
        "version": "1.0.0",
        "author": "Amey Talkatkar",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predictions": "/predict, /predict/batch",
            "models": "/models, /models/{name}/latest",
            "monitoring": "/metrics/drift, /metrics/performance",
            "experiments": "/experiments",
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status with system information
    """
    from src.utils import get_memory_usage
    
    try:
        memory = get_memory_usage()
        
        # Check database connection
        from sqlalchemy import create_engine, text
        engine = create_engine(settings.get_database_url())
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "healthy"
        engine.dispose()
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        db_status = "unhealthy"
        memory = {}
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "environment": settings.environment,
        "database": db_status,
        "memory": {
            "used_mb": memory.get('process_mb', 0),
            "available_mb": memory.get('available_mb', 0),
            "percent": memory.get('percent', 0),
        }
    }


# Import and register routes
from api.routes import predictions, models, monitoring

app.include_router(predictions.router, prefix="/predict", tags=["Predictions"])
app.include_router(models.router, prefix="/models", tags=["Models"])
app.include_router(monitoring.router, prefix="/metrics", tags=["Monitoring"])


# ==================== Exception Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers,
    )
