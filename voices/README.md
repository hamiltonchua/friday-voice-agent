# Voice References for Chatterbox TTS

This directory contains reference audio clips for Chatterbox voice cloning.

## Current Voice
- `rosamund_pike.wav` — Soft British female voice (from Pride and Prejudice audiobook)

## Adding Custom Voices

1. Find a clean audio sample (10-30 seconds of clear speech works best)
2. Convert to mono WAV at 24kHz:
   ```bash
   ffmpeg -i source.mp3 -ar 24000 -ac 1 voice_name.wav
   ```
3. Update `start-chatterbox.sh` to point to your new file:
   ```bash
   export CHATTERBOX_REF="$SCRIPT_DIR/voices/voice_name.wav"
   ```
4. Restart the server

## Tips
- Audiobook narrations work great (calm, consistent tone)
- Avoid clips with background music or noise
- Interview clips can sound too energetic/lively
