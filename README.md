# DND-Video-Pipeline

AI pipeline that converts D&D audio sessions into cinematic videos using speech-to-text, LLM orchestration, and text-to-video models.

## Overview

This pipeline automates the synthesis of long-form audio into a cinematic video representation:

1. **Audio**: Extracts or takes input audio.
2. **Transcribe**: Uses state-of-the-art ASR models (AssemblyAI, Deepgram, Rev AI, Whisper, WhisperX, Google Cloud STT, Amazon Transcribe, NVIDIA NeMo) to transcribe and diarize the session.
3. **LLM Scenography**: Parses transcripts via LLMs (OpenAI, Anthropic, Google Gemini, Deepseek, Llama, Qwen, Gemma, Mistral, Dolphin) to extract character actions, settings, and visual prompts.
4. **Video**: Generates distinct video scenes using text-to-video APIs (Luma Dream Machine, Minimax Hailuo, Replicate Pixverse, Kling AI, Runway, Pika Labs, HunyuanVideo, LTX-Video, CogVideoX, Mochi 1, Runware).
5. **Final Stitch**: Combines the generated videos, transitions, and original audio into the final deliverable.

## Prerequisites

- **Python >= 3.10**
- **FFmpeg**: Must be installed globally and accessible via your system's PATH.

## Installation

You can easily set up the project on your local machine using our automated scripts:

### Windows

Run the setup script from PowerShell:

```cmd
.\scripts\setup.bat
```

### macOS / Linux

Make the script executable and run it:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

The setup script will:

- Verify Python version.
- Create and activate a `.venv`.
- Install all dependencies from `requirements.txt`.
- Copy the `.env.example` template to a new `.env` file.

## Supported Models

### Stage 1: Transcription (Audio -> Text)

| Provider                | Class                    | Notes                                                              |
| ----------------------- | ------------------------ | ------------------------------------------------------------------ |
| **AssemblyAI**          | `AssemblyAITranscriber`  | Recommended API , WER 2.3, speaker diarization                     |
| **Deepgram**            | `DeepgramTranscriber`    | Fast API, Nova-2 model                                             |
| **Rev AI**              | `RevAiTranscriber`       | Budget-friendly at $0.20/hr                                        |
| **Whisper (Local)**     | `WhisperTranscriber`     | OpenAI Whisper Large v2, no API cost                               |
| **WhisperX (Local)**    | `WhisperXTranscriber`    | Enhanced Whisper with word-level timestamps + Pyannote diarization |
| **Google Cloud STT**    | `GoogleCloudTranscriber` | Chirp 2 model, 2B parameters                                       |
| **Amazon Transcribe**   | `AmazonTranscriber`      | AWS-native, S3 upload required                                     |
| **NVIDIA NeMo (Local)** | `NvidiaNemoTranscriber`  | Parakeet TDT 1.1B, best-in-class WER                               |

### Stage 2: LLM Processing (Text -> Storyboard)

| Provider             | Class                   | Notes                                      |
| -------------------- | ----------------------- | ------------------------------------------ |
| **OpenAI GPT**       | `OpenAIGPTProcessor`       | GPT-4o-mini default, tool calling          |
| **Anthropic Claude** | `ClaudeProcessor`       | Claude 3.5 Sonnet default, tool calling    |
| **Google Gemini**    | `GeminiProcessor`       | Gemini 2.0 Flash default, function calling |
| **Deepseek**         | `DeepseekProcessor`     | deepseek-chat, OpenAI-compatible API       |
| **Llama (Local)**    | `LocalLlamaProcessor`   | Any Ollama model, llama3.1 default         |
| **Qwen (Local)**     | `LocalQwenProcessor`    | Qwen 2.5 via Ollama                        |
| **Gemma (Local)**    | `LocalGemmaProcessor`   | Google Gemma 3 via Ollama                  |
| **Mistral (Local)**  | `LocalMistralProcessor` | Mistral via Ollama                         |
| **Dolphin (Local)**  | `LocalDolphinProcessor` | Dolphin-Mistral via Ollama                 |

### Stage 3: Video Generation (Text -> Video)

| Provider               | Class                     | API                               | Notes                             |
| ---------------------- | ------------------------- | --------------------------------- | --------------------------------- |
| **Luma Dream Machine** | `LumaVideoGenerator`      | Direct API                        | Ray-2, async polling              |
| **Minimax Hailuo**     | `HailuoVideoGenerator`    | fal.ai                            | Hailuo-02 standard                |
| **Replicate Pixverse** | `ReplicateVideoGenerator` | Replicate                         | Pixverse v4                       |
| **Kling AI**           | `KlingVideoGenerator`     | fal.ai                            | v1.6 standard, 5-10s clips        |
| **Runway Gen-4**       | `RunwayVideoGenerator`    | Direct API                        | Gen-4 Turbo / Gen-3 Alpha         |
| **Pika Labs 2.5**      | `PikaVideoGenerator`      | Replicate                         | High motion quality               |
| **HunyuanVideo 1.5**   | `HunyuanVideoGenerator`   | Replicate                         | Tencent, 720p                     |
| **LTX-Video**          | `LTXVideoGenerator`       | Replicate                         | Lightricks, real-time generation  |
| **CogVideoX-5B 1.5**   | `CogVideoXGenerator`      | Replicate                         | ZhipuAI / THUDM                   |
| **Mochi 1**            | `MochiGenerator`          | Replicate                         | Genmo, high fidelity              |
| **Runware**            | `RunwareVideoGenerator`   | Direct API                        | Fast GPU inference                |
| **NVIDIA Cosmos**      | `NvidiaCosmosGenerator`   | Not yet publicly available        |
| **OpenAI Sora**        | `OpenAISoraGenerator`     | No public API (ChatGPT only)      |
| **Google Veo 2**       | `GoogleVeoGenerator`      | Allowlisted Vertex AI access only |
| **Open-Sora**          | `OpenSoraGenerator`       | Local                             | Requires 8x A100 80GB local setup |
| **Pyramid Flow**       | `PyramidFlowGenerator`    | Local                             | Requires 1x A100 80GB local setup |

### Stage 4: Assembly

| Provider   | Class             | Notes                                    |
| ---------- | ----------------- | ---------------------------------------- |
| **FFmpeg** | `FFmpegAssembler` | Video stitching, audio overlay, captions |

## Configuration

Edit the `.env` file in the root directory to include your necessary API keys. At minimum, define keys for the services you intend to use.

```env
# Stage 1: Transcription API Keys
ASSEMBLYAI_API_KEY="your-assembly-ai-key"
DEEPGRAM_API_KEY="your-deepgram-key"
REV_AI_API_KEY="your-rev-ai-key"

# Stage 1: Google Cloud STT (use either API key or service account credentials)
GOOGLE_CLOUD_API_KEY="your-google-cloud-api-key"
# OR: GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Stage 1: Amazon Transcribe
AWS_ACCESS_KEY_ID="your-aws-access-key"
AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
AWS_DEFAULT_REGION="us-east-1"
AWS_S3_BUCKET="your-s3-bucket-name"

# Stage 1: WhisperX (for Pyannote speaker diarization)
HUGGING_FACE_TOKEN="your-huggingface-token"

# Stage 2: LLM API Keys
OPENAI_API_KEY="your-openai-key"
ANTHROPIC_API_KEY="your-anthropic-key"
GOOGLE_API_KEY="your-google-ai-key"
DEEPSEEK_API_KEY="your-deepseek-key"

# Stage 3: Video Generation API Keys
REPLICATE_API_TOKEN="your-replicate-key"
LUMA_API_KEY="your-luma-key"
LUMA_MODEL="ray-2"
FAL_KEY="your-fal-ai-key"
RUNWAY_API_KEY="your-runway-key"
RUNWARE_API_KEY="your-runware-key"

# Server Configuration
local_Server_Port=8500
DEFAULT_LLM='openai'
DEFAULT_TRANSCRIBER='assembly'
```

## Execution

### Web UI

To run the Web UI, use the provided startup script (this automatically handles virtual environments and ports):

#### Windows
```cmd
.\scripts\StartWebserver.bat
```

#### macOS / Linux
```bash
uvicorn Web.app:app --reload
```

Navigate to `http://localhost:8500` in your web browser. (Or whatever port it tells you set in the .env file)

### CLI Orchestrator

To orchestrate from the command line, run the pipeline script (with your `venv` activated):

```bash
python src/orchestrator/pipeline.py --help
```
