<<<<<<< HEAD
=======
<<<<<<< HEAD
const hre = require("hardhat");

const COORDINATOR_ADDRESS = process.env.COORDINATOR;

async function main() {
  const coordinator = await hre.ethers.getContractAt("Coordinator", COORDINATOR_ADDRESS);
  await (await coordinator.startRound()).wait();
  const round = await coordinator.currentRound();
  console.log("Started round number:", round.toString());
}

main()
  .then(() => process.exit(0))
  .catch(error => {
    console.error(error);
    process.exit(1);
  });
=======
>>>>>>> 8d64af2 (replaced troublesome uagents with simulation)
const hre = require("hardhat");

const COORDINATOR_ADDRESS = process.env.COORDINATOR;

async function main() {
  const coordinator = await hre.ethers.getContractAt("Coordinator", COORDINATOR_ADDRESS);
  await (await coordinator.startRound()).wait();
  const round = await coordinator.currentRound();
  console.log("Started round number:", round.toString());
}

main()
  .then(() => process.exit(0))
  .catch(error => {
    console.error(error);
    process.exit(1);
  });
<<<<<<< HEAD
=======
>>>>>>> 5b9db2d (agent networking sample)
>>>>>>> 8d64af2 (replaced troublesome uagents with simulation)
