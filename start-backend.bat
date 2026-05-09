@echo off
cd /d "%~dp0backend"
uv run python server.py
