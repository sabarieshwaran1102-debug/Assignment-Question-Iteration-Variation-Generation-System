# AMIGO Models Directory

This directory contains metadata, local weight configurations, and notes for running local open-weight Large Language Models (LLMs) within the **AMIGO** (*Assignment Question Iteration & Variation Generation System*) platform.

---

## 🤖 Supported Local Open-Weight Models

AMIGO operates **100% locally** with zero reliance on paid external cloud APIs. The model layer is abstracted through `LocalLLMProvider` (`packages/common/providers/local_provider.py`), which connects to local model runtimes (such as [Ollama](https://ollama.com/)).

### Primary Model Specification
- **Default Model**: `qwen3:4b` (Qwen3 4-Billion Parameter Model)
- **Runtime Server**: Ollama REST API (`http://localhost:11434`)
- **Execution Mode**: Provider-level batch generation via `generate_text_batch()`

---

## 🛠️ How to Download & Setup Local Models

### 1. Install Ollama
Download and install Ollama from [https://ollama.com/download](https://ollama.com/download).

### 2. Pull the Qwen3 Model
Run the following command in your terminal:

```bash
ollama pull qwen3:4b
```

### 3. Verify Model Availability
Verify that Ollama is serving the model locally:

```bash
curl http://localhost:11434/api/tags
```

---

## ⚙️ Environment Configuration

To configure AMIGO to connect to your local model:

**Windows PowerShell:**
```powershell
$env:AMIGO_LLM_PROVIDER="local"
$env:AMIGO_LOCAL_MODEL="qwen3:4b"
$env:AMIGO_LOCAL_BASE_URL="http://localhost:11434"
```

**Linux/macOS Bash:**
```bash
export AMIGO_LLM_PROVIDER="local"
export AMIGO_LOCAL_MODEL="qwen3:4b"
export AMIGO_LOCAL_BASE_URL="http://localhost:11434"
```

---

## 🔒 Offline & Air-Gapped Deployment

If deploying in a fully air-gapped environment:
1. Place GGUF model files or Ollama blob storage archives into this `models/` folder.
2. Direct your local runtime server to load model weights directly from `models/`.
