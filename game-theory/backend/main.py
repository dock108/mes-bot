from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from dotenv import load_dotenv
from api.auth import router as auth_router
from api.simulation import router as simulation_router
from api.payments import router as payments_router
from api.gto import router as gto_router
from middleware.security import security_middleware

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Create FastAPI app
app = FastAPI(
    title="AI Trading Simulation API",
    description="AI-powered trading simulation and prediction tool",
    version="1.0.0"
)

# Add security middleware
app.middleware("http")(security_middleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(simulation_router)
app.include_router(payments_router)
app.include_router(gto_router)

@app.get("/")
async def root():
    return {"message": "AI Trading Simulation API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)