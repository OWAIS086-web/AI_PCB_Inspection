#!/bin/bash

# # Source conda
source ~/anaconda3/etc/profile.d/conda.sh
# source "$HOME/anaconda3/etc/profile.d/conda.sh"

# Step 0: Check if "--debug" flag is passed
DEBUG_MODE=false
if [ "$1" == "--debug" ]; then
    DEBUG_MODE=true
fi

# Step 1: Overwrite config.json
cp /home/osama/PCB_Inspection/App/configs/config_template.json /home/osama/PCB_Inspection/App/configs/config.json
echo "Config file has been reset."

# Step 2: Activate pcb environment
conda activate pcb
# Always use absolute paths; $HOME expands even inside .desktop Exec lines
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$HOME/anaconda3/lib:$HOME/anaconda3/envs/pcb/lib:$LD_LIBRARY_PATH


# Step 3: Run app.py in background
#python3 /home/osama/PCB_Inspection/App/app.py &
#APP_PID=$!
#echo "app.py started with PID $APP_PID."

LOG_DIR="/home/osama/PCB_Inspection/App/logs"
mkdir -p "$LOG_DIR"
APP_LOG="$LOG_DIR/app_$(date +'%Y-%m-%d_%H-%M-%S').log"

python3 /home/osama/PCB_Inspection/App/app.py > "$APP_LOG" 2>&1 &
APP_PID=$!
echo "app.py started with PID $APP_PID. Logs: $APP_LOG"


# Step 5: Run working_main.py
if [ "$DEBUG_MODE" = true ]; then
    echo "🛠️ DEBUG MODE: Opening new terminal for working_main.py..."
    gnome-terminal -- bash -c "source ~/anaconda3/etc/profile.d/conda.sh; conda activate pcb; python3 /home/osama/PCB_Inspection/App/working_main.py; exec bash"
else
    echo "Running working_main.py silently in background..."
    python3 /home/osama/PCB_Inspection/App/working_main.py &
    WORKING_MAIN_PID=$!
    echo "working_main.py started with PID $WORKING_MAIN_PID."
fi

# Step 6: Launch Firefox and open the URL
firefox http://127.0.0.1:5000/login &

# Get the Firefox PID
FIREFOX_PID=$!
echo "Firefox started with PID $FIREFOX_PID."

# Step 7: Monitor Firefox process using pgrep
echo "Monitoring Firefox. When Firefox closes, scripts will be terminated."

while pgrep -x "firefox" > /dev/null; do
    sleep 1
done

echo "Firefox has closed. Exiting script."


echo "Firefox closed. Killing processes..."

# Step 8: Kill the running processes
# kill $APP_PID
# echo "Killed app.py (PID $APP_PID)."

pkill -f app.py
echo "Killed app.py via pkill."

pkill -f working_main.py
echo "Killed working_main.py via pkill."

echo "All processes terminated. Exiting."
