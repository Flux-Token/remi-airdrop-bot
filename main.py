from xumm import XummSdk
import asyncio
import logging
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, HTMLResponse
from fastapi import Request
from pydantic import BaseModel
import requests
from xrpl.asyncio.clients import AsyncWebsocketClient
from xrpl.models.requests import AccountInfo, Ledger, AccountLines
from xrpl.utils import xrp_to_drops
import uvicorn
from dotenv import load_dotenv

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
XAMAN_API_KEY = os.getenv("XAMAN_API_KEY")
XAMAN_API_SECRET = os.getenv("XAMAN_API_SECRET")
if not XAMAN_API_KEY or not XAMAN_API_SECRET:
    logger.error("XAMAN_API_KEY or XAMAN_API_SECRET not set in .env file")
    raise ValueError("XAMAN_API_KEY and XAMAN_API_SECRET must be set in .env file")
FEE_WALLET_ADDRESS = "rNtwcwRkSJE7kAE3pEzt93txNf3WzdxeZy"
TRUSTLINE_RESERVE_XRP = 2.0  # Minimum XRP for trustline creation

# Initialize Xumm SDK
xumm = XummSdk(XAMAN_API_KEY, XAMAN_API_SECRET)

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
            return client
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
            return client
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

# Helper function to submit transaction to Xaman
async def submit_to_xaman(tx: dict, headers: dict) -> dict:
    payload = {"txjson": tx}
    response = requests.post(
        "https://xumm.app/api/v1/platform/payload",
        headers=headers,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(f"Failed to create payload: {response.text}")
    return response.json()

# New endpoint for initiating OAuth
@app.post("/initiate-oauth")
async def initiate_oauth():
    logger.info("Initiating OAuth with Xaman")
    try:
        # Create a sign-in payload
        payload = {
            "txjson": {
                "TransactionType": "SignIn"
            }
        }
        response = xumm.payload.create(payload)
        if not response or not response.uuid:
            raise Exception("Failed to create OAuth payload")
        
        logger.info(f"OAuth payload created: {response.uuid}")
        return {
            "payload_uuid": response.uuid,
            "qr_code_url": f"https://xumm.app/sign/{response.uuid}_q.png",
            "authorize_url": response.next.always
        }
    except Exception as e:
        logger.error(f"Error initiating OAuth: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate OAuth: {str(e)}"
        )

# New endpoint for polling OAuth callback
@app.get("/callback")
async def callback(payload_uuid: str = Query(...)):
    logger.info(f"Polling callback for payload_uuid: {payload_uuid}")
    try:
        payload_status = xumm.payload.get(payload_uuid)
        if not payload_status:
            raise Exception("Payload not found")

        if payload_status.meta.cancelled:
            return JSONResponse(status_code=400, content={"detail": "Payload was cancelled"})
        if payload_status.meta.expired:
            return JSONResponse(status_code=400, content={"detail": "Payload has expired"})
        if not payload_status.meta.signed:
            return JSONResponse(status_code=202, content={"status": "pending"})

        # Payload is signed, retrieve account and token
        account = payload_status.response.account
        issued_user_token = payload_status.application.issued_user_token
        if not account or not issued_user_token:
            raise Exception("Invalid payload response: missing account or token")

        logger.info(f"Payload signed successfully for account: {account}")
        return {
            "meta": {"signed": True},
            "response": {"account": account},
            "application": {"issued_user_token": issued_user_token}
        }
    except Exception as e:
        logger.error(f"Error polling callback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error polling callback: {str(e)}"
        )

# New endpoint for checking balance
@app.get("/balance")
async def get_balance(
    token_data: dict = Depends(get_access_token),
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None)
):
    logger.info(f"Fetching balance for account: {token_data['account']}")
    client = None
    try:
        client = await get_xrpl_client()
        account = token_data['account']

        # Fetch account info for XRP balance
        account_info_request = AccountInfo(account=account, ledger_index="validated")
        account_response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
        if not account_response.is_successful():
            logger.error(f"Account not found or not funded: {account_response.result.get('error_message', 'Unknown error')}")
            raise HTTPException(
                status_code=404,
                detail=(
                    "No funded XRPL account linked in Xaman. Please create and fund an account on Mainnet at https://xrpl.org. "
                    "Acquire XRP from an exchange like Coinbase or Bitstamp."
                )
            )

        xrp_balance = float(account_response.result["account_data"]["Balance"]) / 1_000_000
        response_data = {
            "account": account,
            "balance_xrp": xrp_balance
        }

        # Fetch token balance if issuer and currency are provided
        if issuer and currency:
            account_lines_request = AccountLines(account=account, ledger_index="validated")
            account_lines_response = await asyncio.wait_for(client.request(account_lines_request), timeout=30)
            if not account_lines_response.is_successful():
                raise Exception("Failed to fetch account lines")

            lines = account_lines_response.result.get("lines", [])
            token_balance = None
            for line in lines:
                if line["account"] == issuer and line["currency"] == currency:
                    token_balance = float(line["balance"])
                    break

            if token_balance is not None:
                response_data["balance_token"] = token_balance
                response_data["currency"] = currency
            else:
                logger.warning(f"No trustline found for {currency} issued by {issuer}")
                response_data["balance_token"] = 0
                response_data["currency"] = currency

        logger.info(f"Balance fetched: {response_data}")
        return response_data
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching balance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to fetch balance: Ensure the account is funded with XRP and tokens on the XRP Ledger Mainnet. "
                "Acquire XRP from an exchange like Coinbase or Bitstamp."
            )
        )
    finally:
        if client and client.is_open():
            await client.close()

# New endpoint for fetching tokens
@app.get("/get-tokens")
async def get_tokens(token_data: dict = Depends(get_access_token)):
    logger.info(f"Fetching tokens for account: {token_data['account']}")
    client = None
    try:
        client = await get_xrpl_client()
        account = token_data['account']

        # Fetch account lines to get tokens
        account_lines_request = AccountLines(account=account, ledger_index="validated")
        account_lines_response = await asyncio.wait_for(client.request(account_lines_request), timeout=30)
        if not account_lines_response.is_successful():
            raise Exception("Failed to fetch account lines")

        lines = account_lines_response.result.get("lines", [])
        tokens = [
            {
                "currency": line["currency"],
                "issuer": line["account"],
                "balance": float(line["balance"])
            }
            for line in lines if float(line["balance"]) > 0
        ]

        logger.info(f"Tokens fetched: {tokens}")
        return {"tokens": tokens}
    except Exception as e:
        logger.error(f"Error fetching tokens: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to fetch tokens: Ensure the account is funded with XRP and has trustlines on the XRP Ledger Mainnet. "
                "Acquire XRP from an exchange like Coinbase or Bitstamp."
            )
        )
    finally:
        if client and client.is_open():
            await client.close()

# Modified airdrop endpoint (unchanged from your original)
@app.post("/airdrop")
async def airdrop(
    request: AirdropRequest,
    token_data: dict = Depends(get_access_token),
    xpmarket_mode: bool = Query(False)
):
    logger.info(f"Initiating airdrop with payload: {request.model_dump()}, xpmarket_mode: {xpmarket_mode}")
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
        # Fetch sequence number
        sequence_response = await asyncio.wait_for(
            client.request(AccountInfo(account=account)),
            timeout=30
        )
        if not sequence_response.is_successful():
            raise Exception(
                f"Failed to fetch account info: "
                f"{sequence_response.result.get('error_message', 'Unknown error')}"
            )
        sequence = int(sequence_response.result["account_data"]["Sequence"])
        logger.info(f"Fetched sequence for account {account}: {sequence}")

        # Fetch last ledger
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
                "Sequence": sequence,
                "LastLedgerSequence": last_ledger_sequence
            }
            logger.info(f"Fee transaction: {fee_tx}")
            payload_data = await submit_to_xaman(fee_tx, headers)
            fee_transaction = {
                "payload_uuid": payload_data["uuid"],
                "sign_url": payload_data["next"]["always"]
            }
            total_network_fee += float(fee) / 1_000_000
            sequence += 1

        # Process airdrop transactions for each wallet
        for index, wallet in enumerate(request.wallets):
            wallet.address = wallet.address.strip()
            if float(wallet.amount or 0) <= 0:
                transactions.append({
                    "index": index,
                    "payload_uuid": None,
                    "sign_url": None,
                    "status": {
                        "address": wallet.address,
                        "status": "Skipped",
                        "error": "Amount is zero"
                    }
                })
                continue

            # Verify account exists and has sufficient XRP in xpmarket mode
            if xpmarket_mode:
                account_info_request = AccountInfo(account=wallet.address, ledger_index="validated")
                account_response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
                if not account_response.is_successful():
                    transactions.append({
                        "index": index,
                        "payload_uuid": None,
                        "sign_url": None,
                        "status": {
                            "address": wallet.address,
                            "status": "Failed",
                            "error": "Account not found or not funded"
                        }
                    })
                    continue
                xrp_balance = float(account_response.result["account_data"]["Balance"]) / 1_000_000
                if xrp_balance < TRUSTLINE_RESERVE_XRP:
                    transactions.append({
                        "index": index,
                        "payload_uuid": None,
                        "sign_url": None,
                        "status": {
                            "address": wallet.address,
                            "status": "Failed",
                            "error": f"Insufficient XRP balance ({xrp_balance} XRP) for trustline creation (requires {TRUSTLINE_RESERVE_XRP} XRP)"
                        }
                    })
                    continue

            # Create payment transaction
            try:
                if request.token_type == "XRP":
                    amount = xrp_to_drops(float(wallet.amount))
                    payment_tx = {
                        "TransactionType": "Payment",
                        "Account": account,
                        "Destination": wallet.address,
                        "Amount": str(amount),
                        "Fee": str(fee),
                        "Sequence": sequence,
                        "LastLedgerSequence": last_ledger_sequence
                    }
                    logger.info(f"Creating XRP payment transaction for {wallet.address}: {payment_tx}")
                else:
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
                        "Sequence": sequence,
                        "LastLedgerSequence": last_ledger_sequence
                    }
        logger.info(f"Creating token payment transaction for {wallet.address}: {payment_tx}")

                # Submit transaction to Xaman
                logger.info(f"Submitting transaction for {wallet.address} to Xaman with sequence {sequence}")
                payload_data = await submit_to_xaman(payment_tx, headers)
                transactions.append({
                    "index": index,
                    "payload_uuid": payload_data["uuid"],
                    "sign_url": payload_data["next"]["always"],
                    "status": {
                        "address": wallet.address,
                        "status": "Pending Payment"
                    }
                })
                total_network_fee += float(fee) / 1_000_000
                sequence += 1

            except Exception as e:
                logger.error(f"Transaction to {wallet.address} failed: {str(e)}")
                # Check if the error is related to a trustline issue (based on Xaman's response)
                error_str = str(e).lower()
                if "trustline" in error_str or "no path" in error_str:
                    logger.warning(f"Transaction to {wallet.address} cancelled due to trustline issue: {str(e)}")
                    transactions.append({
                        "index": index,
                        "payload_uuid": None,
                        "sign_url": None,
                        "status": {
                            "address": wallet.address,
                            "status": "Cancelled",
                            "error": "No trustline for token"
                        }
                    })
                else:
                    transactions.append({
                        "index": index,
                        "payload_uuid": None,
                        "sign_url": None,
                        "status": {
                            "address": wallet.address,
                            "status": "Failed",
                            "error": str(e)
                        }
                    })
                continue  # Move to the next wallet

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

# New endpoint for checking trustlines
@app.post("/check-trustlines")
async def check_trustlines(
    wallets: List[Wallet],
    token_type: str = Query(...),
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    token_data: dict = Depends(get_access_token)
):
    logger.info(f"Checking trustlines for {len(wallets)} wallets, token_type: {token_type}")
    client = None
    try:
        client = await get_xrpl_client()
        results = []

        for wallet in wallets:
            wallet.address = wallet.address.strip()
            if not wallet.address:
                results.append({
                    "address": wallet.address,
                    "has_trustline": False,
                    "error": "Empty wallet address"
                })
                continue

            try:
                # Fetch account lines to check trustlines
                account_lines_request = AccountLines(account=wallet.address, ledger_index="validated")
                account_lines_response = await asyncio.wait_for(client.request(account_lines_request), timeout=30)
                if not account_lines_response.is_successful():
                    results.append({
                        "address": wallet.address,
                        "has_trustline": False,
                        "error": "Account not found or not funded"
                    })
                    continue

                if token_type == "XRP":
                    # XRP does not require trustlines
                    results.append({
                        "address": wallet.address,
                        "has_trustline": True
                    })
                else:
                    # Check for trustline to the specified issuer and currency
                    lines = account_lines_response.result.get("lines", [])
                    has_trustline = any(
                        line["account"] == issuer and line["currency"] == currency
                        for line in lines
                    )
                    results.append({
                        "address": wallet.address,
                        "has_trustline": has_trustline,
                        "error": "No trustline found" if not has_trustline else None
                    })

            except Exception as e:
                logger.error(f"Error checking trustline for {wallet.address}: {str(e)}")
                results.append({
                    "address": wallet.address,
                    "has_trustline": False,
                    "error": str(e)
                })

        logger.info(f"Trustline check completed: {results}")
        return results
    except Exception as e:
        logger.error(f"Error checking trustlines: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to check trustlines: Ensure the wallets are valid and funded on the XRP Ledger Mainnet."
            )
        )
    finally:
        if client and client.is_open():
            await client.close()

# New endpoint for validating wallets
@app.post("/validate-wallets")
async def validate_wallets(
    wallets: List[Wallet],
    token_type: str = Query(...),
    issuer: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    token_data: dict = Depends(get_access_token)
):
    logger.info(f"Validating {len(wallets)} wallets, token_type: {token_type}")
    client = None
    try:
        client = await get_xrpl_client()
        results = []

        for wallet in wallets:
            wallet.address = wallet.address.strip()
            if not wallet.address:
                results.append({
                    "address": wallet.address,
                    "status": "Invalid",
                    "error": "Empty wallet address"
                })
                continue

            try:
                # Check if account exists
                account_info_request = AccountInfo(account=wallet.address, ledger_index="validated")
                account_response = await asyncio.wait_for(client.request(account_info_request), timeout=30)
                if not account_response.is_successful():
                    results.append({
                        "address": wallet.address,
                        "status": "Invalid",
                        "error": "Account not found or not funded"
                    })
                    continue

                # Validate amount
                if wallet.amount is not None and float(wallet.amount) < 0:
                    results.append({
                        "address": wallet.address,
                        "status": "Invalid",
                        "error": "Amount cannot be negative"
                    })
                    continue

                # For tokens, check trustline
                if token_type != "XRP":
                    account_lines_request = AccountLines(account=wallet.address, ledger_index="validated")
                    account_lines_response = await asyncio.wait_for(client.request(account_lines_request), timeout=30)
                    if not account_lines_response.is_successful():
                        results.append({
                            "address": wallet.address,
                            "status": "Invalid",
                            "error": "Failed to fetch account lines"
                        })
                        continue

                    lines = account_lines_response.result.get("lines", [])
                    has_trustline = any(
                        line["account"] == issuer and line["currency"] == currency
                        for line in lines
                    )
                    if not has_trustline:
                        results.append({
                            "address": wallet.address,
                            "status": "Invalid",
                            "error": "No trustline for token"
                        })
                        continue

                results.append({
                    "address": wallet.address,
                    "status": "Valid"
                })

            except Exception as e:
                logger.error(f"Error validating wallet {wallet.address}: {str(e)}")
                results.append({
                    "address": wallet.address,
                    "status": "Invalid",
                    "error": str(e)
                })

        logger.info(f"Wallet validation completed: {results}")
        return results
    except Exception as e:
        logger.error(f"Error validating wallets: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to validate wallets: Ensure the wallets are valid and funded on the XRP Ledger Mainnet."
            )
        )
    finally:
        if client and client.is_open():
            await client.close()

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)