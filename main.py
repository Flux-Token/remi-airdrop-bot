from fastapi import FastAPI, HTTPException, Depends, Query, Header
from fastapi.responses import JSONResponse, FileResponse  # Added for 
serving index.html
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Added for serving static 
files
import os
import logging
from xumm import XummSdk
from xrpl.clients import JsonRpcClient
from xrpl.models.requests import AccountInfo, AccountLines
from xrpl.models.transactions import Payment
from xrpl.utils import xrp_to_drops
from pydantic import BaseModel, Field
from typing import List, Optional

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
# Load API keys and fee wallet address from environment variables
XAMAN_API_KEY = os.getenv("XAMAN_API_KEY")
XAMAN_API_SECRET = os.getenv("XAMAN_API_SECRET")
FEE_WALLET_ADDRESS = os.getenv("FEE_WALLET_ADDRESS")

# Validate environment variables
if not XAMAN_API_KEY or not XAMAN_API_SECRET or not FEE_WALLET_ADDRESS:
    logger.error("XAMAN_API_KEY, XAMAN_API_SECRET, or FEE_WALLET_ADDRESS 
not set as environment variables")
    raise ValueError("XAMAN_API_KEY, XAMAN_API_SECRET, and 
FEE_WALLET_ADDRESS must be set as environment variables")

# Initialize Xumm SDK
xumm = XummSdk(XAMAN_API_KEY, XAMAN_API_SECRET)

# Initialize XRPL client
XRPL_NODE = os.getenv("XRPL_NODE", "wss://s1.ripple.com")
client = JsonRpcClient(XRPL_NODE)

# Pydantic models for request validation
class Wallet(BaseModel):
    address: str
    amount: float = Field(..., ge=0)

class AirdropRequest(BaseModel):
    token_type: str
    total_amount: float = Field(..., ge=0)
    wallets: List[Wallet]
    account: str
    issuer: Optional[str] = None
    currency: Optional[str] = None

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

@app.get("/callback")
async def callback(payload_uuid: str = Query(...)):
    try:
        payload_status = await xumm.payload.get(payload_uuid)
        if not payload_status:
            raise HTTPException(status_code=404, detail="Payload not 
found")

        if payload_status.meta.cancelled:
            return JSONResponse(status_code=400, content={"detail": 
"Payload was cancelled"})
        if payload_status.meta.expired:
            return JSONResponse(status_code=400, content={"detail": 
"Payload has expired"})
        if not payload_status.meta.signed:
            return JSONResponse(status_code=202, content={"status": 
"pending"})

        return payload_status
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-tokens")
async def get_tokens(authorization: str = Header(...), account: str = 
Header(...)):
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid 
authorization header")
        token = authorization.split(" ")[1]
        # Validate token with Xumm (simplified for this example)
        user_token_validation = await 
xumm.payload.validate_user_token(token, account)
        if not user_token_validation:
            raise HTTPException(status_code=401, detail="Invalid or 
expired token")

        # Fetch account lines (trustlines) to get token holdings
        request = AccountLines(account=account)
        response = await client.request(request)
        lines = response.result.get("lines", [])

        tokens = []
        for line in lines:
            tokens.append({
                "currency": line["currency"],
                "issuer": line["account"],
                "name": line["currency"],  # Simplified; in practice, you 
might map currency codes to names
                "balance": float(line["balance"])
            })

        return {"tokens": tokens}
    except Exception as e:
        logger.error(f"Error in get_tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/balance")
async def get_balance(authorization: str = Header(...), account: str = 
Header(...), issuer: Optional[str] = Query(None), currency: Optional[str] 
= Query(None)):
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid 
authorization header")
        token = authorization.split(" ")[1]
        user_token_validation = await 
xumm.payload.validate_user_token(token, account)
        if not user_token_validation:
            raise HTTPException(status_code=401, detail="Invalid or 
expired token")

        # Get XRP balance
        request = AccountInfo(account=account, ledger_index="validated")
        response = await client.request(request)
        if not response.result.get("account_data"):
            raise HTTPException(status_code=404, detail="Account not found 
or not funded")
        xrp_balance = float(response.result["account_data"]["Balance"]) / 
1_000_000

        if issuer and currency:
            # Get token balance
            request = AccountLines(account=account)
            response = await client.request(request)
            lines = response.result.get("lines", [])
            token_balance = None
            for line in lines:
                if line["account"] == issuer and line["currency"] == 
currency:
                    token_balance = float(line["balance"])
                    break
            if token_balance is None:
                raise HTTPException(status_code=404, detail="Token not 
found in account")
            return {"balance_token": token_balance, "currency": currency, 
"account": account}

        return {"balance_xrp": xrp_balance, "account": account}
    except Exception as e:
        logger.error(f"Error in get_balance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate-wallets")
async def validate_wallets(wallets: List[Wallet], token_type: str = 
Query(...), issuer: Optional[str] = Query(None), currency: Optional[str] = 
Query(None), authorization: str = Header(...), account: str = 
Header(...)):
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid 
authorization header")
        token = authorization.split(" ")[1]
        user_token_validation = await 
xumm.payload.validate_user_token(token, account)
        if not user_token_validation:
            raise HTTPException(status_code=401, detail="Invalid or 
expired token")

        results = []
        for wallet in wallets:
            try:
                request = AccountInfo(account=wallet.address, 
ledger_index="validated")
                response = await client.request(request)
                if not response.result.get("account_data"):
                    results.append({"address": wallet.address, "status": 
"invalid", "error": "Account not found or not funded"})
                else:
                    results.append({"address": wallet.address, "status": 
"valid"})
            except Exception as e:
                results.append({"address": wallet.address, "status": 
"invalid", "error": str(e)})
        return results
    except Exception as e:
        logger.error(f"Error in validate_wallets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-trustlines")
async def check_trustlines(wallets: List[Wallet], token_type: str = 
Query(...), issuer: Optional[str] = Query(None), currency: Optional[str] = 
Query(None), authorization: str = Header(...), account: str = 
Header(...)):
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid 
authorization header")
        token = authorization.split(" ")[1]
        user_token_validation = await 
xumm.payload.validate_user_token(token, account)
        if not user_token_validation:
            raise HTTPException(status_code=401, detail="Invalid or 
expired token")

        if token_type != "XRP" and (not issuer or not currency):
            raise HTTPException(status_code=400, detail="Issuer and 
currency required for non-XRP tokens")

        results = []
        for wallet in wallets:
            if token_type == "XRP":
                results.append({"address": wallet.address, 
"has_trustline": True})  # XRP doesn't require trustlines
                continue
            try:
                request = AccountLines(account=wallet.address)
                response = await client.request(request)
                lines = response.result.get("lines", [])
                has_trustline = any(line["account"] == issuer and 
line["currency"] == currency for line in lines)
                results.append({"address": wallet.address, 
"has_trustline": has_trustline})
            except Exception as e:
                logger.error(f"Error checking trustline for 
{wallet.address}: {str(e)}")
                results.append({"address": wallet.address, 
"has_trustline": False})
        return results
    except Exception as e:
        logger.error(f"Error in check_trustlines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/airdrop")
async def airdrop(request: AirdropRequest, authorization: str = 
Header(...)):
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid 
authorization header")
        token = authorization.split(" ")[1]
        user_token_validation = await 
xumm.payload.validate_user_token(token, request.account)
        if not user_token_validation:
            raise HTTPException(status_code=401, detail="Invalid or 
expired token")

        if request.token_type != "XRP" and (not request.issuer or not 
request.currency):
            raise HTTPException(status_code=400, detail="Issuer and 
currency required for non-XRP tokens")

        # Validate total amount matches sum of wallet amounts
        wallet_sum = sum(wallet.amount for wallet in request.wallets)
        if abs(wallet_sum - request.total_amount) > 0.000001:
            raise HTTPException(status_code=422, detail={"error": "Total 
amount does not match sum of wallet amounts", "wallet_sum": wallet_sum, 
"total_amount": request.total_amount})

        # Calculate fees
        network_fee_per_tx = 0.000012  # 12 drops per transaction
        total_transactions = len([w for w in request.wallets if w.amount > 
0])
        total_network_fee = network_fee_per_tx * (total_transactions + 1)  
# +1 for fee transaction
        service_fee_per_wallet = 0.05
        max_service_fee = 5.0
        total_service_fee = min(len(request.wallets) * 
service_fee_per_wallet, max_service_fee)

        # Create fee transaction
        fee_payload = {
            "txjson": {
                "TransactionType": "Payment",
                "Account": request.account,
                "Destination": FEE_WALLET_ADDRESS,
                "Amount": xrp_to_drops(total_service_fee)
            }
        }
        fee_response = await xumm.payload.create(fee_payload)
        if not fee_response:
            raise HTTPException(status_code=500, detail="Failed to create 
fee transaction payload")

        # Create airdrop transactions
        transactions = []
        for wallet in request.wallets:
            if wallet.amount <= 0:
                transactions.append({"status": {"address": wallet.address, 
"status": "skipped (amount <= 0)"}})
                continue
            amount = xrp_to_drops(wallet.amount) if request.token_type == 
"XRP" else {
                "currency": request.currency,
                "value": str(wallet.amount),
                "issuer": request.issuer
            }
            payload = {
                "txjson": {
                    "TransactionType": "Payment",
                    "Account": request.account,
                    "Destination": wallet.address,
                    "Amount": amount
                }
            }
            response = await xumm.payload.create(payload)
            if not response:
                transactions.append({"status": {"address": wallet.address, 
"status": "failed to create payload"}})
                continue
            transactions.append({
                "payload_uuid": response.uuid,
                "sign_url": response.next.always,
                "status": {"address": wallet.address, "status": "pending"}
            })

        return {
            "transactions": transactions,
            "fee_transaction": {
                "payload_uuid": fee_response.uuid,
                "sign_url": fee_response.next.always
            },
            "total_fee": total_network_fee,
            "service_fee": total_service_fee
        }
    except Exception as e:
        logger.error(f"Error in airdrop: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
