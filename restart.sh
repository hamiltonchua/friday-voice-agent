#!/bin/bash
kill -9 $(/usr/sbin/lsof -ti :8765) 2>/dev/null
cd "$(dirname "$0")"
set -a && source .env && set +a
export MLX_TTS_MODEL="mlx-community/Kokoro-82M-bf16"
nohup /opt/homebrew/anaconda3/envs/voice-agent/bin/python3 server.py > /tmp/voice-chat.log 2>&1 &
sleep 8
/usr/sbin/lsof -ti :8765 && echo RUNNING || echo FAILED
