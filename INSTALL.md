# 📚 EPUB Reader

### Read any book with AI-powered voice narration

---

## 🚀 Getting Started

### First Time Only (5 minutes)

**Download Docker Desktop:**

👉 [Click here to download](https://www.docker.com/products/docker-desktop/)

Then:
1. Open the downloaded file
2. Drag the Docker icon to your Applications folder
3. Open Docker from Applications
4. Wait until you see a **whale icon** 🐳 in your menu bar

✅ **Done! You only need to do this once.**

---

## 📖 How to Use

### ▶️ Start Reading

1. Look for the **whale icon** 🐳 in your menu bar (top of screen)
   - If you don't see it, open Docker Desktop from Applications

2. Double-click **`Start Reader`**

3. Your browser will open automatically

4. Upload any `.epub` book and enjoy!

---

### ⏹️ Stop Reading

Double-click **`Stop Reader`**

That's it!

---

## ✨ Features

| | Feature |
|---|---------|
| 🎧 | **AI Voice** - Listen to any book read aloud |
| 🌍 | **Translation** - Hover over words to translate |
| 🎯 | **Focus Mode** - One sentence at a time |
| 📚 | **Library** - Your books saved automatically |
| 🌙 | **Dark Mode** - Easy on the eyes |
| 🔥 | **Streaks** - Track your reading habits |

---

## ❓ Help

| Problem | Solution |
|---------|----------|
| "Docker is not running" | Open Docker Desktop and wait for the whale 🐳 |
| Browser won't open | Go to **localhost:5001** manually |
| Very slow first time | Normal! Downloading voice (only once) |

---

## 💾 Your Data

All your books and progress are saved in the **`data`** folder.

**To backup:** Copy the `data` folder to a safe place.

---

## ☁️ Optional: EC2 Coordinator + S3 Sync

If you want local Kokoro generation to sync with EC2/S3, create a `.env` file next to `docker-compose.yml`:

```bash
KOKORO_BACKEND=onnx
COORDINATOR_API_URL=https://api.reader.psybytes.com
COORDINATOR_TIMEOUT=20
COORDINATOR_IDLE_POLL_INTERVAL=2.0
AUDIO_UPLOAD_FORMAT=m4b
WORKER_BOOK_PARALLELISM=8
WORKER_UPLOAD_URL_BATCH_SIZE=80
WORKER_COMPLETE_BATCH_SIZE=50
WORKER_MP3_BITRATE=64k
WORKER_MP3_SAMPLE_RATE=24000
```

If user is logged in via Cognito in the web app, token is forwarded to local backend automatically.
You only need `COORDINATOR_BEARER_TOKEN` when running without web-login token forwarding.

`COORDINATOR_API_URL` defaults to `https://api.reader.psybytes.com`, so no env is required for standard setup.

### Start Local Audio Worker

```bash
docker compose up -d --build
```

Check health:

```bash
curl http://127.0.0.1:5001/worker/health
```

Then use `https://reader.psybytes.com` while logged in.

Under each book on homepage:
- `Generate Audio`: runs local worker in parallel on your machine, uploads compressed MP3 to S3, and updates cloud progress.
- `Load Audio Locally`: downloads ready cloud audio files into local cache for faster playback.

---

<p align="center">
  <i>Made with ❤️ for book lovers</i>
</p>
