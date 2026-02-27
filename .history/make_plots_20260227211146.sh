#!/bin/bash 
source venv/bin/active

# test if script is working
echo "Generating plots..."

# run python script
python3 -u plot_results.py

echo "Done..."