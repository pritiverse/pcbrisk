@echo off
title PCBRisk Live AOI Platform
echo ======================================================================
echo             PCBRisk: Risk-Calibrated Topological AOI Engine
echo ======================================================================
echo Starting FastAPI Backend Server on http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
python code/server.py
pause
