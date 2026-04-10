#!/bin/bash
# Start vidseq backend + frontend + dev tunnel for remote access
# Usage: ./scripts/start-dev.sh

set -e
cd "$(dirname "$0")/.."

echo "Starting backend..."
uv run vidseq &>/tmp/vidseq_server.log &
BACKEND_PID=$!

echo "Starting frontend..."
cd frontend
npm run dev &>/tmp/vidseq_frontend.log &
FRONTEND_PID=$!
cd ..

# Wait for frontend to bind
echo "Waiting for servers..."
for i in $(seq 1 30); do
    if grep -q 'http://localhost:5173' /tmp/vidseq_frontend.log 2>/dev/null; then
        break
    fi
    sleep 1
done

echo "Backend PID: $BACKEND_PID (port 8000)"
echo "Frontend PID: $FRONTEND_PID (port 5173)"
echo ""
echo "Opening dev tunnel..."
if ! devtunnel user show &>/dev/null; then
    echo "Login required. Logging in..."
    devtunnel user login
fi
devtunnel host -p 5173 --allow-anonymous
