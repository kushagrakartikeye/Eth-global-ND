from web3 import Web3
import json

# Replace with your Coordinator contract address
COORDINATOR_ADDRESS = "0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9"

# Connect to local node
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

# Load ABI
with open("artifacts/contracts/Coordinator.sol/Coordinator.json", "r") as f:
    abi = json.load(f)["abi"]

coord = w3.eth.contract(address=COORDINATOR_ADDRESS, abi=abi)

owner = coord.functions.owner().call()

print(f"Coordinator contract owner address: {owner}")
