@echo off
REM Navigate to project root (replace path with your actual location)
cd /d C:\Users\KIIT0001\b2b-energy-fl

REM Activate venv once and keep it for all child processes
call .\venv\Scripts\activate

REM Start client1 on port 8001
start "client1" cmd /k "set AGENT_NAME=client1&& set AGENT_PORT=8001&& python agents\client_agent.py"

REM Start client2 on port 8002
start "client2" cmd /k "set AGENT_NAME=client2&& set AGENT_PORT=8002&& python agents\client_agent.py"

REM Start client3 on port 8003
start "client3" cmd /k "set AGENT_NAME=client3&& set AGENT_PORT=8003&& python agents\client_agent.py"