#!/bin/sh
# Start script for Railway deployment
# Sets CHAINLIT_PORT from Railway's PORT variable

export CHAINLIT_PORT="${PORT:-8000}"
echo "Starting Chainlit on port $CHAINLIT_PORT"
exec chainlit run chat_ui.py --host 0.0.0.0 --port "$CHAINLIT_PORT"
