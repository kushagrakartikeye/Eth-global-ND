const hre = require("hardhat");

const COORDINATOR_ADDRESS = process.env.COORDINATOR;

async function main() {
  const [owner, participant1, participant2, participant3] = await hre.ethers.getSigners();

  const coordinator = await hre.ethers.getContractAt("Coordinator", COORDINATOR_ADDRESS);

  console.log("Registering participants...");

  await (await coordinator.registerParticipant(participant1.address)).wait();
  console.log("Registered participant:", participant1.address);

  await (await coordinator.registerParticipant(participant2.address)).wait();
  console.log("Registered participant:", participant2.address);

  await (await coordinator.registerParticipant(participant3.address)).wait();
  console.log("Registered participant:", participant3.address);
}

main()
  .then(() => process.exit(0))
  .catch(error => {
    console.error(error);
    process.exit(1);
  });
