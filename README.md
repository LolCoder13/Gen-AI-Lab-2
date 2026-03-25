# ChatGPT UI Clone (Gemini + Hugging Face + OpenAI Speech)

A full-stack chat app with:
- Gemini text responses
- Gemini and Hugging Face image generation
- File/image attachments
- Speech-to-text (STT)
- Text-to-speech (TTS)
- Chat history and UI state persistence in the browser

Backend entry point: server.js  
Frontend files: index.html, script.js, styles.css  
Project scripts and dependencies: package.json  
Environment template: .env.example

## Prerequisites

1. Node.js 18 or newer (recommended: latest LTS)
2. npm
3. At least one API key:
- Google key for Gemini (required to start server)
- Optional OpenAI key for OpenAI STT/TTS
- Optional Hugging Face token for HF text/image/STT/TTS routes

## Setup

1. Open a terminal in the project folder.
2. Install dependencies:

```bash
npm install
```

3. Create a local env file from .env.example:
- Create a file named `.env` in the project root
- Copy values from `.env.example`
- Fill in your real keys

Minimum required:
- GOOGLE_API_KEY

Optional:
- OPENAI_API_KEY
- HF_TOKEN

## Run the project

1. Start the server:

```bash
npm start
```

2. Open your browser at:

```text
http://localhost:3000
```

Default port comes from .env.example as `PORT=3000`.

## Environment Variables

From .env.example:

- OPENAI_API_KEY: enables OpenAI STT/TTS endpoints
- PORT: server port
- GOOGLE_API_KEY: required for Gemini features
- GEMINI_TEMPERATURE, GEMINI_TOP_P, GEMINI_TOP_K: generation tuning
- GEMINI_MAX_OUTPUT_TOKENS: response length cap
- GEMINI_MAX_HISTORY_TURNS: chat context window size
- GEMINI_HARD_CHAR_LIMIT: hard cap on returned text length
- HF_TOKEN: enables Hugging Face text/image/STT/TTS
- HF_PROVIDER: HF router provider
- HF_IMAGE_MODEL: default HF image model
- HF_STT_MODEL: default HF speech-to-text model
- HF_TTS_MODEL: default HF text-to-speech model

## Notes

- The server exits on startup if GOOGLE_API_KEY is missing (see server.js).
- Do not commit .env.
- Commit .env.example safely with placeholder or empty values only.

## Troubleshooting

1. Error: Missing GOOGLE_API_KEY
- Add GOOGLE_API_KEY to .env and restart.

2. STT/TTS not working
- For OpenAI STT/TTS, set OPENAI_API_KEY.
- For HF STT/TTS models, set HF_TOKEN and choose `hf-stt:` or `hf-tts:` model paths in the UI.

3. Image generation errors
- Check quota and model availability for your provider/key.
- Verify HF_TOKEN if using Hugging Face image models.
