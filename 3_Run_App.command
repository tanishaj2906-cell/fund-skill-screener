#!/bin/bash
# Right-click > Open to start the Fund Skill Screener (Mac). Keep this window open while using the app.
cd "$(dirname "$0")"
python3 -m pip install --quiet -r requirements.txt || python3 -m pip install --quiet --break-system-packages -r requirements.txt
echo "Starting the app. It will open in Google Chrome in a few seconds."
echo "If it doesn't, open Chrome and go to:  http://localhost:8501"
echo "Close this window to stop the app."
( sleep 6; open -a "Google Chrome" "http://localhost:8501" 2>/dev/null || open "http://127.0.0.1:8501" ) &
python3 -m streamlit run app.py --server.headless true
