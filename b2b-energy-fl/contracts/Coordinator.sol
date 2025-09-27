// SPDX-License-Identifier: MIT
pragma solidity ^0.8.21;

import "@openzeppelin/contracts/access/Ownable.sol";

interface IRewardToken {
    function mint(address to, uint256 amount) external;
}

contract Coordinator is Ownable {
    IRewardToken public reward;

    uint256 public currentRound;

    mapping(address => bool) public registered;
    mapping(uint256 => mapping(address => bytes32)) public updateHash;
    mapping(uint256 => mapping(address => bool)) public submitted;
    address[] public participantList;

    event Registered(address indexed participant);
    event RoundStarted(uint256 indexed round);
    event UpdateSubmitted(uint256 indexed round, address indexed participant, bytes32 updateHash);
    event RoundFinalized(uint256 indexed round, bytes32 globalModelHash);

    constructor(address rewardToken) {
        reward = IRewardToken(rewardToken);
    }

    function registerParticipant(address p) external onlyOwner {
        require(!registered[p], "Already registered");
        registered[p] = true;
        participantList.push(p);
        emit Registered(p);
    }

    function getParticipants() external view returns (address[] memory) {
        return participantList;
    }

    function startRound() external onlyOwner {
        currentRound += 1;
        emit RoundStarted(currentRound);
    }

    function submitUpdate(bytes32 h) external {
        require(registered[msg.sender], "Not registered");
        require(updateHash[currentRound][msg.sender] == bytes32(0), "Already submitted");
        updateHash[currentRound][msg.sender] = h;
        submitted[currentRound][msg.sender] = true;
        emit UpdateSubmitted(currentRound, msg.sender, h);
    }

    function finalizeRound(bytes32 globalHash, address[] calldata participants, uint256 rewardPerParticipant) external onlyOwner {
        for (uint256 i = 0; i < participants.length; i++) {
            address p = participants[i];
            if (submitted[currentRound][p]) {
                reward.mint(p, rewardPerParticipant);
            }
        }
        emit RoundFinalized(currentRound, globalHash);
    }
}
