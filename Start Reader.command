#!/bin/bash

# EPUB Reader Launcher
# Double-click this file to start the reader

cd "$(dirname "$0")"

echo "========================================"
echo "       EPUB Reader with AI Audio        "
echo "========================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running!"
    echo ""
    echo "Please start Docker Desktop first:"
    echo "  1. Open Docker Desktop from Applications"
    echo "  2. Wait for it to start (whale icon in menu bar)"
    echo "  3. Double-click this file again"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Starting EPUB Reader..."
echo "(First run may take a few minutes to download)"
echo ""

# Start the container
docker-compose up -d --build

# Wait for server to be ready
echo "Waiting for server to start..."
sleep 3

# Check if container is running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "Server is running!"
    echo ""

    # Open browser
    open "http://localhost:5001"

    echo "Browser opened to http://localhost:5001"
    echo ""
    echo "To stop the reader, run: docker-compose down"
    echo "Or close this window and run 'Stop Reader.command'"
else
    echo ""
    echo "Failed to start. Check Docker logs:"
    docker-compose logs
fi

echo ""
read -p "Press Enter to close this window..."
