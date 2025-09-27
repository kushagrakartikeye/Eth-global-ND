// scripts/deploy.cjs
const fs = require("fs");
const path = require("path");
const hre = require("hardhat");
const { ethers } = hre;

async function main() {
  const signers = await ethers.getSigners();
  const [deployer, client1, client2, client3] = signers;
  console.log("Deploying contracts with account:", deployer.address);

  const Reward = await ethers.getContractFactory("RewardToken");
  const reward = await Reward.deploy();
  await reward.deployed();
  console.log("RewardToken deployed to:", reward.address);

  const Coordinator = await ethers.getContractFactory("Coordinator");
  const coordinator = await Coordinator.deploy(reward.address);
  await coordinator.deployed();
  console.log("Coordinator deployed to:", coordinator.address);

  await reward.transferOwnership(coordinator.address);
  console.log("✅ Transferred RewardToken ownership to Coordinator");

  const rpc = process.env.RPC_URL || `http://127.0.0.1:${process.env.PORT || 8546}`;

  const localNetworks = ["localhost", "hardhat"];
  let ownerPrivateKey = null;
  let clientKeys = {};
  let clientAddrs = {};

  if (localNetworks.includes(hre.network.name)) {
    let mnemonic;
    if (ethers.provider._mnemonic) {
      mnemonic = ethers.provider._mnemonic().phrase;
    } else {
      mnemonic = "test test test test test test test test test test test junk";
    }
    const { Mnemonic, HDNodeWallet } = ethers;
    const mnemonicObj = Mnemonic.fromPhrase(mnemonic);
    const masterNode = HDNodeWallet.fromMnemonic(mnemonicObj, "m/44'/60'/0'/0/0");
    ownerPrivateKey = masterNode.privateKey.replace("0x", "");
    for (let i = 1; i <= 3; i++) {
      const childNode = HDNodeWallet.fromMnemonic(mnemonicObj, `m/44'/60'/0'/0/${i}`);
      clientKeys[`client${i}`] = childNode.privateKey.replace("0x", "");
      clientAddrs[`client${i}`] = signers[i].address;
    }
    console.warn(
      "⚠️ writing local owner/client private keys to deployed.json for convenience. Do NOT commit this file."
    );
  } else if (process.env.OWNER_PK) {
    ownerPrivateKey = process.env.OWNER_PK;
  }

  const data = {
    reward: reward.address,
    coordinator: coordinator.address,
    rpc: rpc,
    owner: deployer.address,
    owner_pk: ownerPrivateKey,
    client_keys: clientKeys,
    client_addrs: clientAddrs,
  };

  fs.writeFileSync(path.join(__dirname, "../deployed.json"), JSON.stringify(data, null, 2));
  console.log("✅ Addresses + RPC + keys written to deployed.json");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
