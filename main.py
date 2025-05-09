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
from fastapi.responses import Response, HTMLResponse
from fastapi import Request
from pydantic import BaseModel
import requests
from xrpl.asyncio.clients import AsyncWebsocketClient
from xrpl.models.requests import AccountInfo, Ledger, AccountLines, ServerInfo, GatewayBalances
from xrpl.models.requests import GenericRequest
from xrpl.utils import xrp_to_drops
import uvicorn
from dotenv import load_dotenv
import httpx 

# Load environment variables from .env file
load_dotenv()

# Configure logging
# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Changed from INFO to DEBUG
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
XAMAN_API_KEY = os.getenv("XAMAN_API_KEY")
XAMAN_API_SECRET = os.getenv("XAMAN_API_SECRET")
if not XAMAN_API_KEY or not XAMAN_API_SECRET:
    logger.error("XAMAN_API_KEY or XAMAN_API_SECRET not set in .env file")
    raise ValueError("XAMAN_API_KEY and XAMAN_API_SECRET must be set in .env file")
FEE_WALLET_ADDRESS = "rNtwcwRkSJE7kAE3pEzt93txNf3WzdxeZy"
FLUX_ISSUER = "rhbmVVzvDme96hHsb2DxKKKfxqnMexB2mz"
FLUX_CURRENCY = "464C555800000000000000000000000000000000"
TRUSTLINE_RESERVE_XRP = 2.0  # Minimum XRP for trustline creation

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
    return {"token": token, "account": account}

# Get XRPL client for Mainnet
async def get_xrpl_client():
    primary_node = "wss://s1.ripple.com"
    fallback_node = "wss://s2.ripple.com"
    logger.info(f"Connecting to primary Mainnet node: {primary_node}")
    client = None
    try:
        client = AsyncWebsocketClient(primary_node)
        await client.open()
        if client.is_open():
            response = await client.request(ServerInfo())
            if response.is_successful():
                network_id = response.result.get("info", {}).get("network_id", 0)
                if network_id != 0:
                    logger.error(f"Connected to non-Mainnet network (network_id: {network_id})")
                    await client.close()
                    raise HTTPException(
                        status_code=500,
                        detail="Connected to incorrect network. Expected XRP Ledger Mainnet."
                    )
                logger.info(
                    f"Connected to primary Mainnet node ({primary_node}): "
                    f"{response.result}"
                )
                return client
            else:
                logger.warning(
                    f"Primary node ({primary_node}) response invalid, "
                    "trying fallback..."
                )
                await client.close()
    except Exception as e:
        logger.warning(
            f"Failed to connect to primary Mainnet node ({primary_node}): "
            f"{str(e)}, trying fallback..."
        )
        if client and client.is_open():
            await client.close()

    logger.info(f"Connecting to fallback Mainnet node: {fallback_node}")
    try:
        client = AsyncWebsocketClient(fallback_node)
        await client.open()
        if client.is_open():
            response = await client.request(ServerInfo())
            if response.is_successful():
                network_id = response.result.get("info", {}).get("network_id", 0)
                if network_id != 0:
                    logger.error(f"Connected to non-Mainnet network (network_id: {network_id})")
                    await client.close()
                    raise HTTPException(
                        status_code=500,
                        detail="Connected to incorrect network. Expected XRP Ledger Mainnet."
                    )
                logger.info(
                    f"Connected to fallback Mainnet node ({fallback_node}): "
                    f"{response.result}"
                )
                return client
            else:
                logger.error(
                    f"Fallback node ({fallback_node}) response invalid."
                )
                await client.close()
    except Exception as e:
        logger.error(
            f"Failed to connect to fallback Mainnet node ({fallback_node}): "
            f"{str(e)}"
        )
        if client and client.is_open():
            await client.close()
    raise HTTPException(
        status_code=500,
        detail=(
            "Failed to connect to XRP Ledger Mainnet nodes. Please try again "
            "later."
        )
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

# Decode hex currency
# Updated decode_hex_currency for case-insensitive handling
from fastapi import Request  # Add this import

# Update decode_hex_currency to handle edge cases
from fastapi import Request

# Ensure decode_hex_currency is robust
from fastapi import Request
import httpx  # Add for external API calls

# Ensure decode_hex_currency is robust
def decode_hex_currency(hex_currency: str) -> str:
    if not hex_currency or not hex_currency.strip():
        return ""
    hex_currency = hex_currency.strip().upper()
    # Standard currency codes (e.g., "XRP", "USD") are 3-4 characters and not hex
    if len(hex_currency) < 40 and hex_currency.isalnum():
        return hex_currency
    # XRPL token currency codes are 40-character hex strings
    if len(hex_currency) != 40 or not hex_currency.isalnum():
        return hex_currency
    try:
        result = ''
        for i in range(0, len(hex_currency), 2):
            byte = int(hex_currency[i:i+2], 16)
            if byte == 0:
                break
            result += chr(byte)
        decoded = result or hex_currency
        logger.debug(f"Decoded hex currency {hex_currency} to {decoded}")
        return decoded
    except Exception as e:
        logger.warning(f"Failed to decode hex currency {hex_currency}: {str(e)}")
        return hex_currency

# In the /check-trustlines endpoint, update the trustline comparison logic
@app.post("/check-trustlines")
async def check_trustlines(
    wallets: List[Wallet],
    token_type: str = Query(...),
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None, alias="currency"),
    fetch_balances: bool = Query(False),  # New parameter to fetch balances
    token_data: dict = Depends(get_access_token),
    request: Request = None
):
    logger.info(f"Received request to /check-trustlines with wallets: {wallets}, token_type: {token_type}, issuer: {issuer}, currency: {currency}, fetch_balances: {fetch_balances}")
    logger.debug(f"Raw query string: {request.query_params}")
    
    effective_currency = currency if currency is not None else token_type
    decoded_token_type = decode_hex_currency(token_type)
    decoded_currency = decode_hex_currency(effective_currency)
    logger.debug(f"Decoded parameters: token_type={decoded_token_type}, currency={decoded_currency}, effective_currency={effective_currency}, issuer: {issuer}")
    
    if decoded_token_type != "XRP" and (not issuer or not effective_currency):
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
            logger.debug(f"Checking trustline for wallet: {wallet.address}")
            try:
                result = {
                    "address": wallet.address,
                    "has_trustline": False,
                    "balance": None,  # Will be populated if fetch_balances is true
                    "error": None
                }

                # Verify account exists
                account_info_request = AccountInfo(account=wallet.address, ledger_index="validated")
                account_response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
                if not account_response.is_successful():
                    logger.warning(f"Account {wallet.address} does not exist or is not funded: {account_response.result}")
                    result["error"] = "Account not found or not funded"
                    results.append(result)
                    continue

                # Fetch balance if requested
                if fetch_balances:
                    balance = 0
                    if decoded_token_type == "XRP":
                        balance = float(account_response.result["account_data"]["Balance"]) / 1_000_000
                        logger.debug(f"XRP balance for {wallet.address}: {balance}")
                    else:
                        account_lines_request = AccountLines(account=wallet.address, ledger_index="validated")
                        lines_response = await asyncio.wait_for(client.request(account_lines_request), timeout=30)
                        if lines_response.is_successful():
                            for line in lines_response.result.get("lines", []):
                                if (line["account"] == issuer and 
                                    line["currency"] == effective_currency):
                                    balance = float(line["balance"])
                                    logger.debug(f"Token balance for {wallet.address}: {balance}")
                                    break
                    result["balance"] = balance

                # Check trustlines (only if token_type is not XRP)
                if decoded_token_type == "XRP":
                    logger.debug(f"Token type is XRP for {wallet.address}, trustline not required")
                    result["has_trustline"] = True
                else:
                    trustline_request = GenericRequest(
                        command="account_lines",
                        account=wallet.address,
                        ledger_index="validated"
                    )
                    response = await asyncio.wait_for(client.request(trustline_request), timeout=30)
                    if not response.is_successful():
                        logger.error(f"Trustline request failed for {wallet.address}: {response.result}")
                        result["error"] = "Failed to fetch trustlines"
                        results.append(result)
                        continue

                    trustlines = response.result.get("lines", [])
                    logger.info(f"Trustlines for wallet {wallet.address}: {trustlines}")
                    trustline_exists = False
                    for line in trustlines:
                        line_issuer = line["account"].strip()
                        line_currency = line["currency"].strip()
                        target_issuer = issuer.strip() if issuer else ""
                        target_currency = effective_currency.strip()

                        logger.debug(f"Comparing trustline for {wallet.address}: "
                                    f"line_issuer={line_issuer}, target_issuer={target_issuer}, "
                                    f"line_currency={line_currency}, target_currency={target_currency}")

                        if line_issuer == target_issuer and line_currency == target_currency:
                            trustline_exists = True
                            logger.info(f"Trustline match found for {wallet.address}: {line}")
                            break
                    result["has_trustline"] = trustline_exists
                    if not trustline_exists:
                        result["error"] = "No trustline found"

                results.append(result)
            except Exception as e:
                logger.error(f"Error processing wallet {wallet.address}: {str(e)}")
                results.append({
                    "address": wallet.address,
                    "has_trustline": False,
                    "balance": None,
                    "error": str(e)
                })
        logger.info("Trustline check completed successfully.")
        return results
    except Exception as e:
        logger.error(f"Trustline check error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to check trustlines: Ensure wallets are valid on the XRP Ledger Mainnet."
        )
    finally:
        if client and client.is_open():
            await client.close()

# Updated /check-balances endpoint with additional logging
# Replace the existing /check-balances endpoint with this updated version
@app.post("/check-balances")
async def check_balances(
    wallets: List[Wallet],
    token_type: str = Query(...),
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    token_data: dict = Depends(get_access_token),
    request: Request = None
):
    logger.debug(f"Raw query string: {request.query_params}")
    logger.info(f"Checking balances for wallets: {wallets}, token_type: {token_type}, issuer: {issuer}, currency: {currency}")
    decoded_token_type = decode_hex_currency(token_type)
    decoded_currency = decode_hex_currency(currency) if currency else decode_hex_currency(token_type)
    logger.debug(f"Decoded parameters: token_type={decoded_token_type}, currency={decoded_currency}, issuer={issuer}")
    
    if decoded_token_type != "XRP" and (not issuer or not decoded_currency):
        raise HTTPException(
            status_code=400,
            detail="Issuer and currency are required for token balance checks"
        )
    client = None
    try:
        client = await get_xrpl_client()
        async with httpx.AsyncClient() as http_client:
            results = []
            for wallet in wallets:
                wallet.address = wallet.address.strip()
                try:
                    # Verify account exists and has sufficient XRP
                    account_info_request = AccountInfo(account=wallet.address, ledger_index="validated")
                    account_response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
                    if not account_response.is_successful():
                        logger.warning(f"Account {wallet.address} does not exist or is not funded: {account_response.result}")
                        results.append({
                            "address": wallet.address,
                            "has_balance": False,
                            "balance": 0.0,
                            "error": "Account not found or not funded"
                        })
                        continue
                    xrp_balance = float(account_response.result["account_data"]["Balance"]) / 1_000_000
                    if xrp_balance < TRUSTLINE_RESERVE_XRP:
                        logger.warning(f"Insufficient XRP balance for {wallet.address}: {xrp_balance} XRP")
                        results.append({
                            "address": wallet.address,
                            "has_balance": False,
                            "balance": 0.0,
                            "error": f"Insufficient XRP balance ({xrp_balance} XRP) for trustline creation"
                        })
                        continue

                    # Initialize balance check
                    has_balance = False
                    balance_value = 0.0
                    balance_source = None

                    # Try account_lines first for balance
                    trustline_request = GenericRequest(
                        command="account_lines",
                        account=wallet.address,
                        ledger_index="validated",
                        peer=issuer if issuer else None,
                        limit=1000
                    )
                    trustline_response = await asyncio.wait_for(client.request(trustline_request), timeout=30)
                    if trustline_response.is_successful():
                        trustlines = trustline_response.result.get("lines", [])
                        logger.debug(f"Trustlines for {wallet.address} via account_lines: {trustlines}")
                        for line in trustlines:
                            line_issuer = line.get("account", "").strip()
                            line_currency = decode_hex_currency(line.get("currency", "")).strip().upper()
                            line_balance = float(line.get("balance", "0.0"))
                            logger.debug(f"Checking account_lines for {wallet.address}: issuer={line_issuer}, currency={line_currency}, balance={line_balance}, expected_issuer={issuer}, expected_currency={decoded_currency}")
                            if (line_issuer == issuer and 
                                line_currency == decoded_currency and 
                                line_balance > 0):
                                has_balance = True
                                balance_value = line_balance
                                balance_source = "account_lines"
                                logger.info(f"Balance found via account_lines for {wallet.address}: {balance_value}")
                                break
                    else:
                        logger.warning(f"account_lines request failed for {wallet.address}: {trustline_response.result}")

                    # Try GatewayBalances if account_lines finds no balance
                    if not has_balance:
                        gateway_request = GatewayBalances(account=issuer, hotwallet=[wallet.address], ledger_index="validated")
                        gateway_response = await asyncio.wait_for(client.request(gateway_request), timeout=30)
                        if gateway_response.is_successful():
                            balances = gateway_response.result.get("assets", {}).get(wallet.address, [])
                            logger.debug(f"Gateway balances for {wallet.address}: {balances}")
                            for asset in balances:
                                asset_currency = asset.get("currency", "").strip().upper()
                                asset_value = float(asset.get("value", "0"))
                                logger.debug(f"Checking GatewayBalances for {wallet.address}: currency={asset_currency}, value={asset_value}, expected_currency={decoded_currency}")
                                if (asset_currency == decoded_currency and 
                                    asset_value > 0):
                                    has_balance = True
                                    balance_value = asset_value
                                    balance_source = "GatewayBalances"
                                    logger.info(f"Balance found via GatewayBalances for {wallet.address}: {balance_value}")
                                    break
                        else:
                            logger.warning(f"GatewayBalances request failed for {wallet.address}: {gateway_response.result}")

                    # Fallback to XRPScan API if both XRPL methods fail
                    if not has_balance:
                        try:
                            xrpscan_url = f"https://api.xrpscan.com/api/v1/account/{wallet.address}/trustlines"
                            response = await http_client.get(xrpscan_url, timeout=10)
                            if response.status_code == 200:
                                trustlines = response.json()
                                logger.debug(f"XRPScan raw response for {wallet.address}: {trustlines}")
                                if not isinstance(trustlines, list):
                                    logger.warning(f"XRPScan response for {wallet.address} is not a list: {trustlines}")
                                    continue
                                expected_currency_decoded = decoded_currency
                                if expected_currency_decoded.startswith("46"):
                                    try:
                                        expected_currency_decoded = bytes.fromhex(decoded_currency.rstrip("0")).decode("utf-8")
                                    except Exception as e:
                                        logger.warning(f"Failed to decode expected currency {decoded_currency} for {wallet.address}: {str(e)}")
                                        expected_currency_decoded = decoded_currency  # Fallback to raw form
                                for line in trustlines:
                                    counterparty = line.get("counterparty", "")
                                    token_currency = line.get("currency", "")
                                    balance_str = line.get("balance", "0")
                                    logger.debug(f"XRPScan trustline for {wallet.address}: counterparty={counterparty}, currency={token_currency}, balance={balance_str}, expected_issuer={issuer}, expected_currency_decoded={expected_currency_decoded}, expected_currency_raw={decoded_currency}")
                                    if (counterparty == issuer and 
                                        (token_currency == expected_currency_decoded or 
                                         token_currency == decoded_currency)):
                                        try:
                                            balance = float(balance_str)
                                            if balance > 0:
                                                has_balance = True
                                                balance_value = balance
                                                balance_source = "XRPScan"
                                                logger.info(f"Balance found via XRPScan for {wallet.address}: {balance_value}")
                                                break
                                        except ValueError:
                                            logger.warning(f"Invalid balance format in XRPScan response for {wallet.address}: {balance_str}")
                            else:
                                logger.warning(f"XRPScan API failed for {wallet.address}: {response.status_code} - {response.text}")
                        except Exception as e:
                            logger.error(f"XRPScan API error for {wallet.address}: {str(e)}")

                    logger.info(f"Balance result for {wallet.address}: has_balance={has_balance}, balance={balance_value}, source={balance_source}")
                    results.append({
                        "address": wallet.address,
                        "has_balance": has_balance,
                        "balance": balance_value,
                        "balance_source": balance_source,
                        "error": None if has_balance else "No balance found"
                    })
                except Exception as e:
                    logger.error(f"Error checking balance for {wallet.address}: {str(e)}")
                    results.append({
                        "address": wallet.address,
                        "has_balance": False,
                        "balance": 0.0,
                        "balance_source": None,
                        "error": str(e)
                    })
            logger.info("Balance check completed.")
            return results
    except Exception as e:
        logger.error(f"Balance check error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to check balances: Ensure wallets are valid on the XRP Ledger Mainnet."
        )
    finally:
        if client and client.is_open():
            await client.close()



# Initiate OAuth with Xumm
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
    
    if (
        response.status_code == 200 
        and data.get("meta", {}).get("exists", False)
        and data.get("meta", {}).get("signed", False)
        and data.get("response", {}).get("account")
        and data.get("application", {}).get("issued_user_token")
    ):
        account = data["response"]["account"]
        issued_user_token = data["application"]["issued_user_token"]
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

# Modified airdrop endpoint
@app.post("/airdrop")
async def airdrop(
    request: AirdropRequest,
    token_data: dict = Depends(get_access_token),
):
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
                    "payload_uuid": None,
                    "sign_url": None,
                    "status": {
                        "address": wallet.address,
                        "status": "Skipped",
                        "error": "Amount is zero"
                    }
                })
                continue
            if request.token_type != "XRP":
                # Check trustline
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
                        "payload_uuid": None,
                        "sign_url": None,
                        "status": {
                            "address": wallet.address,
                            "status": "Failed",
                            "error": "No trustline exists for this token"
                        }
                    })
                    continue

                # Create token payment transaction
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
                        "payload_uuid": None,
                        "sign_url": None,
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
                # XRP payment
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
                        "payload_uuid": None,
                        "sign_url": None,
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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
