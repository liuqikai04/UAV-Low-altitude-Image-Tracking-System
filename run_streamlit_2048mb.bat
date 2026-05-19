@echo off
cd /d "%~dp0"
python -m streamlit run uav_detector\streamlit_app\app.py --server.maxUploadSize=2048
pause
