#!/bin/bash

# EPUB Reader Stop Script
# Double-click this file to stop the reader

cd "$(dirname "$0")"

echo "Stopping EPUB Reader..."
docker-compose down

echo ""
echo "Reader stopped."
echo "Your books and progress are saved."
echo ""
read -p "Press Enter to close..."
