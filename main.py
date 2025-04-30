from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import logging
from xumm import XummSdk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from the 'static' folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at the root URL
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Configuration
# Load API keys from environment variables
XAMAN_API_KEY = os.getenv("XAMAN_API_KEY")
XAMAN_API_SECRET = os.getenv("XAMAN_API_SECRET")

# Validate environment variables
if not XAMAN_API_KEY or not XAMAN_API_SECRET:
    logger.error("XAMAN_API_KEY or XAMAN_API_SECRET not set as environment 
variables")
    raise ValueError("XAMAN_API_KEY and XAMAN_API_SECRET must be set as 
environment variables")

# Initialize Xumm SDK
xumm = XummSdk(XAMAN_API_KEY, XAMAN_API_SECRET)

# OAuth initiation endpoint (required for "Log in with Xaman")
@app.post("/initiate-oauth")
async def initiate_oauth():
    try:
        payload = {
            "txjson": {
                "TransactionType": "SignIn"
            }
        }
        response = await xumm.payload.create(payload)
        if not response:
            raise HTTPException(status_code=500, detail="Failed to create 
OAuth payload")
        return {
            "payload_uuid": response.uuid,
            "qr_code_url": f"https://xumm.app/sign/{response.uuid}_q.png",
            "authorize_url": response.next.always
        }
    except Exception as e:
        logger.error(f"Error in initiate_oauth: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
