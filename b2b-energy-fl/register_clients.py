import json
from web3 import Web3

# Load deployed contract addresses + RPC
with open("deployed.json") as f:
    config = json.load(f)

COORD_ADDRESS = config["coordinator"]
RPC_URL = config["rpc"]

# Hardhat account #0 (owner/deployer)
OWNER_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# Hardhat accounts to register as participants
CLIENTS = [
    ("client1", "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"),
    ("client2", "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"),
    ("client3", "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"),
]

# Connect to local Hardhat node
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# Load Coordinator ABI
with open("artifacts/contracts/Coordinator.sol/Coordinator.json") as f:
    abi = json.load(f)["abi"]

coordinator = w3.eth.contract(address=COORD_ADDRESS, abi=abi)

# Load owner (deployer) account
owner = w3.eth.account.from_key(OWNER_PK)

# Register clients
for name, pk in CLIENTS:
    acct = w3.eth.account.from_key(pk)
    tx = coordinator.functions.registerParticipant(acct.address).build_transaction({
        "from": owner.address,
        "nonce": w3.eth.get_transaction_count(owner.address),
        "gas": 500000,
        "gasPrice": w3.to_wei("1", "gwei"),
    })
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=OWNER_PK)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"✅ Registered {name} at {acct.address}, tx: {receipt.transactionHash.hex()}")
