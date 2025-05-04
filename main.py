from xumm import XummSdk
import asyncio
import logging
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, Header
from fastapi import FastAPI, HTTPException, status, Header, Request
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
import requests
from xrpl.asyncio.clients import AsyncWebsocketClient
from xrpl.models.requests import AccountInfo, Ledger, AccountLines, ServerInfo
from xrpl.models.requests import GenericRequest
from xrpl.utils import xrp_to_drops
import uvicorn
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Add a root route to serve index.html
@app.get("/")
async def serve_index():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
# Load API keys from .env file (ensure XAMAN_API_KEY and XAMAN_API_SECRET are set in .env)
XAMAN_API_KEY = os.getenv("XAMAN_API_KEY")
XAMAN_API_SECRET = os.getenv("XAMAN_API_SECRET")
if not XAMAN_API_KEY or not XAMAN_API_SECRET:
    logger.error("XAMAN_API_KEY or XAMAN_API_SECRET not set in .env file")
    raise ValueError("XAMAN_API_KEY and XAMAN_API_SECRET must be set in .env file")
FEE_WALLET_ADDRESS = "rNtwcwRkSJE7kAE3pEzt93txNf3WzdxeZy"

# Pydantic models
class Wallet(BaseModel):
    address: str
    amount: Optional[float] = 0

class AirdropRequest(BaseModel):
    token_type: str
    total_amount: float
    issuer: Optional[str] = None
    currency: Optional[str] = None
    wallets: List[Wallet]
    account: str

class Token(BaseModel):
    name: str
    issuer: str
    currency: str

# Dependency for validating access token
async def get_access_token(authorization: str = Header(...), account: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    logger.info(f"Validating access token: {token} for account: {account}")
    # Skip Xumm /user-token validation and trust the issued_user_token
    # Optionally, add token validation logic here in the future if needed
    return {"token": token, "account": account}

# Get XRPL client for Mainnet
async def get_xrpl_client():
    nodes = [
        "wss://s1.ripple.com",
        "wss://s2.ripple.com",
        "wss://xrplcluster.com",
        "wss://xrpl.ws"
    ]
    client = None
    for node in nodes:
        logger.info(f"Connecting to node: {node}")
        try:
            client = AsyncWebsocketClient(node)
            await client.open()
            if client.is_open():
                response = await client.request(ServerInfo())
                if response.is_successful():
                    network_id = response.result.get("info", {}).get("network_id", 0)
                    if network_id != 0:
                        logger.error(f"Connected to non-Mainnet network (network_id: {network_id}) at {node}")
                        await client.close()
                        continue
                    logger.info(f"Successfully connected to Mainnet node: {node}")
                    return client
                else:
                    logger.warning(f"Node {node} response invalid, trying next node...")
                    await client.close()
        except Exception as e:
            logger.warning(f"Failed to connect to node {node}: {str(e)}, trying next node...")
            if client and client.is_open():
                await client.close()
    raise HTTPException(
        status_code=500,
        detail="Failed to connect to any XRP Ledger Mainnet nodes. Please try again later."
    )
    

# Get current network fee
async def get_current_fee(client: AsyncWebsocketClient) -> int:
    try:
        response = await asyncio.wait_for(client.request(Ledger(ledger_index="validated")), timeout=30)
        if not response.is_successful():
            raise Exception("Failed to fetch ledger data")
        base_fee = int(response.result["ledger"]["base_fee"])
        logger.info(f"Fetched base fee: {base_fee} drops")
        return base_fee
    except Exception as e:
        logger.error(f"Fee fetch error: {str(e)}")
        return 12  # Fallback to 12 drops

# Add the decode_hex_currency function here
def decode_hex_currency(hex_currency: str) -> str:
    if not hex_currency or len(hex_currency) != 40 or not hex_currency.isalnum():
        return hex_currency
    try:
        result = ''
        for i in range(0, len(hex_currency), 2):
            byte = int(hex_currency[i:i+2], 16)
            if byte == 0:
                break
            result += chr(byte)
        return result or hex_currency
    except Exception as e:
        logger.warning(f"Failed to decode hex currency {hex_currency}: {str(e)}")
        return hex_currency

@app.post("/check-trustlines")
async def check_trustlines(
    wallets: List[Wallet],
    token_type: str = Query(...),
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    token_data: dict = Depends(get_access_token)
):
    logger.info(f"Checking trustlines for wallets: {wallets}, token_type: {token_type}, issuer: {issuer}, currency: {currency}")
    decoded_token_type = decode_hex_currency(token_type)  # Decode token_type
    decoded_currency = decode_hex_currency(currency) if currency else None  # Decode currency
    if decoded_token_type != "XRP" and (not issuer or not decoded_currency):
        raise HTTPException(
            status_code=400,
            detail="Issuer and currency are required for token trustline checks"
        )
    client = None
    try:
        client = await get_xrpl_client()
        results = []
        for wallet in wallets:
            wallet.address = wallet.address.strip()
            try:
                if decoded_token_type == "XRP":
                    results.append({
                        "address": wallet.address,
                        "has_trustline": True
                    })
                else:
                    # Check if the account exists first
                    account_info_request = AccountInfo(account=wallet.address, ledger_index="validated")
                    account_response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
                    if not account_response.is_successful():
                        logger.warning(f"Account {wallet.address} does not exist or is not funded: {account_response.result}")
                        results.append({
                            "address": wallet.address,
                            "has_trustline": False
                        })
                        continue

                    trustline_request = GenericRequest(
                        command="account_lines",
                        account=wallet.address,
                        ledger_index="validated"
                    )
                    response = await asyncio.wait_for(client.request(trustline_request), timeout=30)
                    if not response.is_successful():
                        logger.error(f"Trustline request failed for {wallet.address}: {response.result}")
                        results.append({
                            "address": wallet.address,
                            "has_trustline": False
                        })
                        continue

                    trustlines = response.result.get("lines", [])
                    logger.info(f"Trustlines for wallet {wallet.address}: {trustlines}")
                    logger.debug(f"Checking trustline: issuer={issuer}, decoded_currency={decoded_currency}")
                    trustline_exists = any(
                        line["account"] == issuer and decode_hex_currency(line["currency"]) == decoded_currency
                        for line in trustlines
                    )
                    results.append({
                        "address": wallet.address,
                        "has_trustline": trustline_exists
                    })
            except Exception as e:
                logger.error(f"Error checking trustline for {wallet.address}: {str(e)}")
                results.append({
                    "address": wallet.address,
                    "has_trustline": False
                })
        logger.info("Trustline check completed.")
        return results
    except Exception as e:
        logger.error(f"Trustline check error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to check trustlines: Ensure wallets are valid and have "
                "trustlines set up on the XRP Ledger Mainnet. Acquire XRP from "
                "an exchange like Coinbase or Bitstamp."
            )
        )
    finally:
        if client and client.is_open():
            await client.close()

# Initiate OAuth with Xumm
XAMAN_API_KEY = os.getenv("XAMAN_API_KEY")
XAMAN_API_SECRET = os.getenv("XAMAN_API_SECRET")
xumm = XummSdk(XAMAN_API_KEY, XAMAN_API_SECRET)

@app.post("/initiate-oauth")
async def initiate_oauth():
    logger.info("Received request to /initiate-oauth")
    try:
        payload = xumm.payload.create({"TransactionType": "SignIn", "options": {"push": True}})
        logger.info(f"Payload response: {payload.__dict__}")
        response = {
            "payload_uuid": payload.uuid,
            "qr_code_url": f"https://xumm.app/sign/{payload.uuid}_q.png",
            "authorize_url": payload.next.always,
            "websocket_url": payload.refs.websocket_status,
            "mobile_url": payload.refs.deeplink if hasattr(payload.refs, "deeplink") else payload.next.always,
            "pushed": payload.pushed
        }
        logger.info(f"Returning response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in initiate_oauth: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate OAuth: {str(e)}")
# Callback for OAuth polling

@app.get("/callback")
async def callback(payload_uuid: str):
    headers = {
        "X-API-Key": XAMAN_API_KEY,
        "X-API-Secret": XAMAN_API_SECRET
    }
    response = requests.get(f"https://xumm.app/api/v1/platform/payload/{payload_uuid}", headers=headers)
    data = response.json()
    logger.info(f"Callback response for payload {payload_uuid}: {data}")
    
    # Check if the payload is signed and has an account
    if (
        response.status_code == 200 
        and data.get("meta", {}).get("exists", False)
        and data.get("meta", {}).get("signed", False)
        and data.get("response", {}).get("account")
        and data.get("application", {}).get("issued_user_token")
    ):
        account = data["response"]["account"]
        issued_user_token = data["application"]["issued_user_token"]
        # Attempt to generate user token for verification
        token_response = requests.post(
            "https://xumm.app/api/v1/platform/user-token",
            headers=headers,
            json={"user_token": data["response"]["hex"]}
        )
        if token_response.status_code == 200:
            token_data = token_response.json()
            logger.info(f"Xumm user-token response: {token_data}")
            user_token = token_data.get("token")
            if not user_token:
                logger.warning(f"No token in user-token response: {token_data}, using issued_user_token")
                user_token = issued_user_token
        else:
            logger.warning(f"Xumm user-token failed: {token_response.status_code} - {token_response.text}, using issued_user_token")
            user_token = issued_user_token
        return {
            "meta": {"signed": True},
            "application": {"issued_user_token": user_token},
            "response": {"account": account}
        }
    elif data.get("meta", {}).get("cancelled", False):
        raise HTTPException(status_code=400, detail="Payload was cancelled")
    elif data.get("meta", {}).get("expired", False):
        raise HTTPException(status_code=400, detail="Payload has expired")
    else:
        return JSONResponse(status_code=202, content={"status": "pending"})

# Get token holdings
@app.get("/get-tokens")
async def get_tokens(token_data: dict = Depends(get_access_token)):
    client = None
    try:
        client = await get_xrpl_client()
        account_lines_request = AccountLines(account=token_data["account"], ledger_index="validated")
        response = await asyncio.wait_for(client.request(account_lines_request), timeout=30)
        if not response.is_successful():
            raise Exception(response.result.get("error_message", "Failed to fetch account lines"))
        tokens = []
        for line in response.result.get("lines", []):
            if float(line["limit"]) > 0:
                tokens.append({
                    "name": line["currency"],
                    "issuer": line["account"],
                    "currency": line["currency"]
                })
        logger.info(f"Token holdings for {token_data['account']}: {tokens}")
        return {"tokens": tokens}
    except Exception as e:
        logger.error(f"Token fetch error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to fetch token holdings: Ensure account is funded on "
                "the XRP Ledger Mainnet. Acquire XRP from an exchange like "
                "Coinbase or Bitstamp."
            )
        )
    finally:
        if client and client.is_open():
            await client.close()

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# Check balance
@app.get("/balance")
async def balance(
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    token_data: dict = Depends(get_access_token)
):
    logger.info(f"Received request to /balance for account: {token_data['account']}, issuer: {issuer}, currency: {currency}")
    client = None
    try:
        client = await get_xrpl_client()
        account_info_request = AccountInfo(account=token_data["account"], ledger_index="validated")
        response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
        if not response.is_successful():
            raise Exception(response.result.get("error_message", "Account not found"))
        balance_xrp = float(response.result["account_data"]["Balance"]) / 1_000_000
        logger.info(f"XRP balance for {token_data['account']}: {balance_xrp}")
        
        if issuer and currency:
            account_lines_request = AccountLines(account=token_data["account"], ledger_index="validated")
            lines_response = await asyncio.wait_for(client.request(account_lines_request), timeout=30)
            if not lines_response.is_successful():
                raise Exception(lines_response.result.get("error_message", "Failed to fetch account lines"))
            balance_token = 0
            for line in lines_response.result.get("lines", []):
                if line["account"] == issuer and line["currency"] == currency:
                    balance_token = float(line["balance"])
                    break
            logger.info(f"Token balance for {token_data['account']} ({currency}): {balance_token}")
            return {
                "account": token_data["account"],
                "balance_token": balance_token,
                "currency": currency
            }
        return {
            "account": token_data["account"],
            "balance_xrp": balance_xrp
        }
    except Exception as e:
        logger.error(f"Balance error: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=(
                "Failed to fetch balance: Account not found or not funded. "
                "Please acquire XRP from an exchange like Coinbase or Bitstamp "
                "and fund your Mainnet account."
            )
        )
    finally:
        if client and client.is_open():
            await client.close()

# Validate wallets
@app.post("/validate-wallets")
async def validate_wallets(
    wallets: List[Wallet],
    token_type: str = Query(...),
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    token_data: dict = Depends(get_access_token)
):
    logger.info(f"Validating wallets: {wallets}, token_type: {token_type}, issuer: {issuer}, currency: {currency}")
    client = None
    try:
        client = await get_xrpl_client()
        results = []
        for wallet in wallets:
            wallet.address = wallet.address.strip()
            try:
                account_info_request = AccountInfo(account=wallet.address, ledger_index="validated")
                response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
                if response.is_successful():
                    results.append({
                        "address": wallet.address,
                        "status": "Valid",
                        "error": None
                    })
                else:
                    error_message = response.result.get("error_message", "Unknown error")
                    results.append({
                        "address": wallet.address,
                        "status": "Invalid",
                        "error": error_message
                    })
            except Exception as e:
                results.append({
                    "address": wallet.address,
                    "status": "Invalid",
                    "error": str(e)
                })
        return results
    except Exception as e:
        logger.error(f"Wallet validation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to validate wallets: Ensure addresses are valid and "
                "funded on the XRP Ledger Mainnet. Acquire XRP from an exchange "
                "like Coinbase or Bitstamp."
            )
        )
    finally:
        if client and client.is_open():
            await client.close()

# Check trustlines
@app.post("/check-trustlines")
async def check_trustlines(
    wallets: List[Wallet],
    token_type: str = Query(...),
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    token_data: dict = Depends(get_access_token)
):
    logger.info(f"Checking trustlines for wallets: {wallets}, token_type: {token_type}, issuer: {issuer}, currency: {currency}")
    if token_type != "XRP" and (not issuer or not currency):
        raise HTTPException(
            status_code=400,
            detail="Issuer and currency are required for token trustline checks"
        )
    client = None
    try:
        client = await get_xrpl_client()
        results = []
        for wallet in wallets:
            wallet.address = wallet.address.strip()
            logger.info(f"Processing wallet: {wallet.address}")
            try:
                # Step 1: Check if the account exists and is funded
                account_info_request = AccountInfo(account=wallet.address, ledger_index="validated")
                account_response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
                if not account_response.is_successful():
                    logger.warning(f"Account {wallet.address} does not exist or is not funded: {account_response.result}")
                    results.append({
                        "address": wallet.address,
                        "has_trustline": False,
                        "error": "Account does not exist or is not funded",
                        "suggestion": "Ensure the wallet is funded with at least 10 XRP on the XRP Ledger Mainnet."
                    })
                    continue

                # Step 2: Check XRP balance for funding requirements
                xrp_balance = float(account_response.result["account_data"]["Balance"]) / 1_000_000
                # Fetch all trustlines with pagination
                all_trustlines = []
                marker = None
                retry_attempts = 3
                for attempt in range(retry_attempts):
                    try:
                        while True:
                            logger.info(f"Fetching trustlines for {wallet.address}, marker: {marker}")
                            trustline_request = GenericRequest(
                                command="account_lines",
                                account=wallet.address,
                                ledger_index="validated",
                                marker=marker
                            )
                            trustline_response = await asyncio.wait_for(client.request(trustline_request), timeout=30)
                            if not trustline_response.is_successful():
                                logger.error(f"Trustline request failed for {wallet.address}: {trustline_response.result}")
                                break

                            trustlines = trustline_response.result.get("lines", [])
                            all_trustlines.extend(trustlines)
                            marker = trustline_response.result.get("marker")
                            if not marker:
                                break
                        break  # Exit retry loop if successful
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1}/{retry_attempts} failed for {wallet.address}: {str(e)}")
                        if attempt == retry_attempts - 1:
                            results.append({
                                "address": wallet.address,
                                "has_trustline": False,
                                "error": "Failed to fetch trustlines after retries",
                                "suggestion": "Try again later or check the wallet on xrpscan.com."
                            })
                            continue
                        await asyncio.sleep(1)  # Wait before retrying

                if not all_trustlines and "error" in results[-1] if results else False:
                    continue  # Skip if trustline fetch failed

                logger.info(f"Total trustlines for wallet {wallet.address}: {len(all_trustlines)}")
                logger.info(f"Trustlines for wallet {wallet.address}: {all_trustlines}")
                min_reserve = 10  # XRP for account activation
                trustline_reserve = 2  # XRP per trustline
                required_xrp = min_reserve + (len(all_trustlines) * trustline_reserve)
                if xrp_balance < required_xrp:
                    logger.warning(f"Account {wallet.address} has insufficient XRP: {xrp_balance} < {required_xrp}")
                    results.append({
                        "address": wallet.address,
                        "has_trustline": False,
                        "error": f"Insufficient XRP balance ({xrp_balance} < {required_xrp}) to support trustlines",
                        "suggestion": "Fund the wallet with more XRP to meet the reserve requirement."
                    })
                    continue

                # Step 3: Check issuer's RequireAuth flag
                issuer_info_request = AccountInfo(account=issuer, ledger_index="validated")
                issuer_response = await asyncio.wait_for(client.request(issuer_info_request), timeout=30)
                require_auth = False
                if issuer_response.is_successful():
                    flags = issuer_response.result["account_data"].get("Flags", 0)
                    # lsfRequireAuth flag is 0x00040000
                    require_auth = (flags & 0x00040000) != 0
                    logger.info(f"Issuer {issuer} RequireAuth flag: {require_auth}")

                # Step 4: Check for trustline in the fetched data
                if token_type == "XRP":
                    logger.info(f"Wallet {wallet.address} supports XRP (no trustline needed)")
                    results.append({
                        "address": wallet.address,
                        "has_trustline": True,
                        "error": None
                    })
                else:
                    trustline_exists = False
                    matching_trustline = None
                    for line in all_trustlines:
                        # Compare currency and issuer directly in hex format
                        if (line["account"] == issuer and 
                            line["currency"] == currency):
                            matching_trustline = line
                            # Check if the trustline is usable (non-zero limit or balance)
                            if float(line["limit"]) > 0 or float(line["balance"]) != 0:
                                trustline_exists = True
                            # If RequireAuth is enabled, check if the trustline is authorized
                            if require_auth and not line.get("authorized", False):
                                trustline_exists = False
                            break

                    if not trustline_exists:
                        decoded_currency = decode_hex_currency(currency)
                        error_msg = f"No trustline for {decoded_currency} with issuer {issuer}"
                        suggestion = "Verify the trustline exists on the XRP Ledger Mainnet using xrpscan.com."
                        # Fallback: Check with an external API (e.g., xrpscan.com)
                        try:
                            logger.info(f"Fallback: Checking trustline for {wallet.address} via external API")
                            # Example: Query xrpscan.com API (you'd need to sign up for an API key)
                            xrpscan_url = f"https://api.xrpscan.com/api/v1/account/{wallet.address}/trustlines"
                            response = requests.get(xrpscan_url)
                            if response.status_code == 200:
                                xrpscan_trustlines = response.json()
                                for trustline in xrpscan_trustlines:
                                    if (trustline["counterparty"] == issuer and 
                                        trustline["currency"] == decoded_currency):
                                        trustline_exists = True
                                        matching_trustline = trustline
                                        error_msg = None
                                        suggestion = "Trustline found via external API, but not in node response. Consider using a different XRP Ledger node."
                                        break
                        except Exception as e:
                            logger.warning(f"External API check failed for {wallet.address}: {str(e)}")

                        if not trustline_exists:
                            if matching_trustline:
                                if require_auth and not matching_trustline.get("authorized", False):
                                    error_msg = f"Trustline for {decoded_currency} with issuer {issuer} exists but is not authorized (RequireAuth enabled)"
                                    suggestion = "The issuer must authorize the trustline. Contact the issuer or check xrpl.services for authorization steps."
                                elif float(matching_trustline["limit"]) == 0 and float(matching_trustline["balance"]) == 0:
                                    error_msg = f"Trustline for {decoded_currency} with issuer {issuer} exists but has zero limit and balance"
                                    suggestion = "Set a non-zero limit on the trustline using xrpl.services or another wallet tool."
                                logger.info(f"Trustline details for {wallet.address}: {matching_trustline}")
                            logger.info(f"No usable trustline found for {wallet.address}: issuer={issuer}, currency={currency} ({decoded_currency})")
                        results.append({
                            "address": wallet.address,
                            "has_trustline": trustline_exists,
                            "error": error_msg,
                            "suggestion": suggestion
                        })
                    else:
                        logger.info(f"Trustline found for {wallet.address}: issuer={issuer}, currency={currency}")
                        results.append({
                            "address": wallet.address,
                            "has_trustline": True,
                            "error": None,
                            "trustline_details": matching_trustline
                        })
            except Exception as e:
                logger.error(f"Error checking trustline for {wallet.address}: {str(e)}")
                results.append({
                    "address": wallet.address,
                    "has_trustline": False,
                    "error": str(e),
                    "suggestion": "Try again later or check the wallet on xrpscan.com."
                })
        logger.info("Trustline check completed.")
        return results
    except Exception as e:
        logger.error(f"Trustline check error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to check trustlines: Ensure wallets are valid and have "
                "trustlines set up on the XRP Ledger Mainnet. Acquire XRP from "
                "an exchange like Coinbase or Bitstamp."
            )
        )
    finally:
        if client and client.is_open():
            await client.close()

# Airdrop endpoint
@app.post("/airdrop")
async def airdrop(request: AirdropRequest, token_data: dict = Depends(get_access_token)):
    logger.info(f"Initiating airdrop with payload: {request.model_dump()}")
    account = request.account
    if not account:
        logger.error("Account is required in request body but missing")
        raise HTTPException(status_code=400, detail="Account is required in request body")
    logger.info(f"Processing airdrop for account: {account}")
    try:
        total_wallet_amount = round(sum(float(wallet.amount or 0)
                                       for wallet in request.wallets), 6)
        request_total_amount = round(float(request.total_amount or 0), 6)
        logger.info(
            f"Total wallet amount: {total_wallet_amount}, "
            f"Request total amount: {request_total_amount}, "
            f"Difference: {abs(total_wallet_amount - request_total_amount)}"
        )
        if abs(total_wallet_amount - request_total_amount) > 0.000001:
            wallet_amounts = [float(wallet.amount or 0) for wallet in request.wallets]
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Total amount does not match sum of wallet amounts",
                    "request_total_amount": request_total_amount,
                    "calculated_wallet_amount": total_wallet_amount,
                    "difference": abs(total_wallet_amount - request_total_amount),
                    "wallet_amounts": wallet_amounts,
                    "raw_payload": request.model_dump()
                }
            )
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid amount data: {str(e)}, Payload: {request.model_dump()}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"Invalid amount data: {str(e)}",
                "request_total_amount": str(request.total_amount),
                "calculated_wallet_amount": "N/A",
                "wallet_amounts": [str(wallet.amount) for wallet in request.wallets],
                "raw_payload": request.model_dump()
            }
        )

    if not request.wallets:
        raise HTTPException(status_code=422, detail="At least one wallet is required")

    if request.token_type != "XRP":
        if not request.issuer:
            raise HTTPException(status_code=400, detail="Issuer is required for token airdrops")
        if not request.currency:
            raise HTTPException(
                status_code=400,
                detail="Currency code is required for token airdrops"
            )

    client = None
    try:
        client = await get_xrpl_client()
        sequence_response = await asyncio.wait_for(
            client.request(AccountInfo(account=account)),
            timeout=30
        )
        if not sequence_response.is_successful():
            raise Exception(
                f"Failed to fetch account info: "
                f"{sequence_response.result.get('error_message', 'Unknown error')}"
            )
        sequence_int = int(sequence_response.result["account_data"]["Sequence"])
        logger.info(f"Fetched sequence for account {account}: {sequence_int}")

        last_ledger_response = await asyncio.wait_for(
            client.request(Ledger(ledger_index="validated")),
            timeout=30
        )
        if not last_ledger_response.is_successful():
            raise Exception(
                f"Failed to fetch ledger: "
                f"{last_ledger_response.result.get('error_message', 'Unknown error')}"
            )
        last_ledger = int(last_ledger_response.result["ledger"]["ledger_index"])
        last_ledger_sequence = last_ledger + 100
        logger.info(
            f"Fetched last ledger index: {last_ledger}, "
            f"using last_ledger_sequence: {last_ledger_sequence}"
        )

        fee = await get_current_fee(client)
        logger.info(f"Using fee: {fee} drops")

        # Calculate REMI service fee: 0.05 XRP per wallet, capped at 5 XRP
        service_fee = min(len(request.wallets) * 0.05, 5.0)
        logger.info(f"Calculated REMI service fee: {service_fee} XRP for {len(request.wallets)} wallets")

        total_network_fee = 0
        transactions = []
        fee_transaction = None
        headers = {
            "X-API-Key": XAMAN_API_KEY,
            "X-API-Secret": XAMAN_API_SECRET,
            "Authorization": f"Bearer {token_data['token']}"
        }

        # Create fee transaction
        if service_fee > 0:
            fee_amount = xrp_to_drops(service_fee)
            logger.info(f"Creating fee transaction: {service_fee} XRP ({fee_amount} drops) to {FEE_WALLET_ADDRESS}")
            fee_tx = {
                "TransactionType": "Payment",
                "Account": account,
                "Destination": FEE_WALLET_ADDRESS,
                "Amount": str(fee_amount),
                "Fee": str(fee),
                "Sequence": int(sequence_int),
                "LastLedgerSequence": int(last_ledger_sequence)
            }
            logger.info(f"Fee transaction: {fee_tx}")
            payload = {"txjson": fee_tx}
            response = requests.post(
                "https://xumm.app/api/v1/platform/payload",
                headers=headers,
                json=payload
            )
            if response.status_code != 200:
                logger.error(
                    f"Xumm payload error for fee payment: "
                    f"{response.status_code} - {response.text}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create fee payment payload: {response.text}"
                )
            payload_data = response.json()
            fee_transaction = {
                "payload_uuid": payload_data["uuid"],
                "sign_url": payload_data["next"]["always"]
            }
            total_network_fee += float(fee) / 1_000_000
            sequence_int += 1

        # Create airdrop transactions
        for i, wallet in enumerate(request.wallets):
            wallet.address = wallet.address.strip()
            if float(wallet.amount or 0) <= 0:
                transactions.append({
                    "status": {
                        "address": wallet.address,
                        "status": "Skipped",
                        "error": "Amount is zero"
                    }
                })
                continue
            if request.token_type != "XRP":
                trustline_request = GenericRequest(
                    command="account_lines",
                    account=wallet.address,
                    ledger_index="validated"
                )
                trustline_response = await asyncio.wait_for(client.request(trustline_request), timeout=30)
                trustlines = trustline_response.result.get("lines", [])
                logger.info(f"Trustlines for wallet {wallet.address}: {trustlines}")
                trustline_exists = any(
                    line["account"] == request.issuer and line["currency"] == request.currency
                    for line in trustlines
                )
                if not trustline_exists:
                    transactions.append({
                        "status": {
                            "address": wallet.address,
                            "status": "Failed",
                            "error": "No trustline exists for this token"
                        }
                    })
                    continue
                payment_tx = {
                    "TransactionType": "Payment",
                    "Account": account,
                    "Destination": wallet.address,
                    "Amount": {
                        "currency": request.currency,
                        "value": str(float(wallet.amount)),
                        "issuer": request.issuer
                    },
                    "Fee": str(fee),
                    "Sequence": int(sequence_int),
                    "LastLedgerSequence": int(last_ledger_sequence)
                }
                logger.info(f"Token payment transaction: {payment_tx}")
                payload = {"txjson": payment_tx}
                response = requests.post(
                    "https://xumm.app/api/v1/platform/payload",
                    headers=headers,
                    json=payload
                )
                if response.status_code != 200:
                    logger.error(
                        f"Xumm payload error for token payment: "
                        f"{response.status_code} - {response.text}"
                    )
                    transactions.append({
                        "status": {
                            "address": wallet.address,
                            "status": "Failed",
                            "error": f"Failed to create payment payload: {response.text}"
                        }
                    })
                    continue
                payload_data = response.json()
                transactions.append({
                    "payload_uuid": payload_data["uuid"],
                    "sign_url": payload_data["next"]["always"],
                    "status": {
                        "address": wallet.address,
                        "status": "Pending Payment"
                    }
                })
                total_network_fee += float(fee) / 1_000_000
                sequence_int += 1
            else:
                amount = xrp_to_drops(float(wallet.amount or 0))
                logger.info(f"XRP payment amount: {amount} drops")
                payment_tx = {
                    "TransactionType": "Payment",
                    "Account": account,
                    "Destination": wallet.address,
                    "Amount": str(amount),
                    "Fee": str(fee),
                    "Sequence": int(sequence_int),
                    "LastLedgerSequence": int(last_ledger_sequence)
                }
                logger.info(f"Payment transaction (XRP): {payment_tx}")
                payload = {"txjson": payment_tx}
                logger.info(
                    f"Sending Xumm payload for XRP payment "
                    f"{i + 1}/{len(request.wallets)}: {payload}"
                )
                response = requests.post(
                    "https://xumm.app/api/v1/platform/payload",
                    headers=headers,
                    json=payload
                )
                if response.status_code != 200:
                    logger.error(
                        f"Xumm payload error for XRP payment: "
                        f"{response.status_code} - {response.text}"
                    )
                    transactions.append({
                        "status": {
                            "address": wallet.address,
                            "status": "Failed",
                            "error": (
                                f"Failed to create payment payload: "
                                f"{response.text}"
                            )
                        }
                    })
                    continue
                payload_data = response.json()
                transactions.append({
                    "payload_uuid": payload_data["uuid"],
                    "sign_url": payload_data["next"]["always"],
                    "status": {
                        "address": wallet.address,
                        "status": "Pending Payment"
                    }
                })
                total_network_fee += float(fee) / 1_000_000
                sequence_int += 1
    except Exception as e:
        logger.error(f"Airdrop error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process airdrop: Ensure the account is funded with "
                "XRP and tokens on the XRP Ledger Mainnet. Acquire XRP from an "
                "exchange like Coinbase or Bitstamp."
            )
        )
    finally:
        if client and client.is_open():
            await client.close()
    logger.info("Airdrop initiated, awaiting user authorization in Xaman.")
    response_content = {
        "transactions": transactions,
        "total_fee": total_network_fee,
        "service_fee": service_fee
    }
    if fee_transaction:
        response_content["fee_transaction"] = fee_transaction
    return JSONResponse(content=response_content)

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

