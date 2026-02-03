# EPUB Reader - Installation Guide

## One-Time Setup (5 minutes)

### Step 1: Install Docker Desktop

1. Download Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Open the downloaded file and drag Docker to Applications
3. Open Docker Desktop from Applications
4. Wait for it to start (you'll see a whale icon in your menu bar)

That's it! Docker only needs to be installed once.

---

## Using the Reader

### To Start
1. Make sure Docker Desktop is running (whale icon in menu bar)
2. Double-click **"Start Reader.command"**
3. Wait for browser to open (first time takes 2-3 minutes)

### To Stop
- Double-click **"Stop Reader.command"**
- Or just quit Docker Desktop

---

## Troubleshooting

### "Docker is not running"
→ Open Docker Desktop from Applications and wait for it to start

### Browser shows "can't connect"
→ Wait a minute and refresh - the server is still starting

### First time is slow
→ Normal! It's downloading the AI voice model (~500MB). Only happens once.

---

## Your Data

All your books, audio, and reading progress are saved in the `data/` folder.
To backup: copy the `data/` folder somewhere safe.
