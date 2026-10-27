# ZentroMart VoiceOps 2026 — Capstone Project

**Public GitHub Repository**: [https://github.com/codeitwithaayush/Phantom-VoiceOPS](https://github.com/codeitwithaayush/Phantom-VoiceOPS)

Voice-first, policy-grounded enterprise AI customer support system for **ZentroMart**, featuring speech recognition and synthesis via Gnani Voice SDK, dual-layer knowledge retrieval with policy precedence, caller identity security gates, relational SQL-style order joins, and multi-turn state tracking.

---

## Architectural Overview

The system is organized into four modular layers:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Gnani Voice Pipeline                            │
│    GnaniSTTClient (en-IN)               GnaniTTSClient (Kaveri, v2.5)  │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ Text Audio / Transcript
┌────────────────────────────────────▼───────────────────────────────────┐
│                      VoiceOps Conversational Brain                     │
│  • Multi-turn Session State            • Adaptive Multilingual (EN/HI) │
│  • Caller Verification Security Gate   • Policy Precedence Engine      │
│  • Escalation Handler (create_ticket)  • Strict Grounding & Anti-Halluc│
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
┌───────────────────▼──────────────┐   ┌─────────────▼───────────────────┐
│        RAG Policy Engine         │   │       Relational Data Engine    │
│  • 21 Markdown Policy Docs (kb/) │   │  • 7 Core CSV Tables (data/)    │
│  • 84 Unique [SECTION-ID] Chunks │   │  • Fast Pandas In-Memory Joins  │
│  • Policy > Guide > Help > FAQ   │   │  • Business Calendar Math       │
│  • Superseded Version Filtering  │   │  • Ref Date: 20 October 2026    │
└──────────────────────────────────┘   └─────────────────────────────────┘
```

1. **Data & RAG Layer (`src/db_loader.py` & `src/kb_loader.py`)**:
   - Ingests 7 relational CSV tables (`customers`, `orders`, `order_items`, `products`, `pincodes`, `holidays`, `support_logs`) using Pandas with `dtype=str` and `keep_default_na=False`.
   - Parses and chunks 21 Markdown policy files in `kb/`, indexing 84 unique `[SECTION-ID]` chunks.
   - Strictly enforces policy precedence:
     $$\text{Policy} > \text{Agent Guide} > \text{Help Article} > \text{FAQ} > \text{Marketing}$$
   - Filters out superseded files (e.g., `pol_returns_v3_superseded.md`) while maintaining historical fallback references for pre-August 2026 orders.

2. **Tool Integration & Security (`src/agent.py`)**:
   - Integrates `lookup_order` and `create_ticket` from `tools.tools`.
   - **Identity Verification Gate**: Requires matching the caller's registered phone last 4 digits (`phone_last4`) or delivery pincode (`delivery_pincode`) before disclosing private order details. Allows at most one retry before permanently refusing order access.
   - **Business Day Calendar**: All delay and timeline calculations operate from the frozen reference date **20 October 2026**, counting Monday–Saturday and skipping listed warehouse holidays.

3. **Conversational Brain & Grounding (`src/agent.py`)**:
   - Tracks multi-turn conversational context (`ConversationState`), maintaining order context across turns and resolving follow-up coreferences (e.g. asking for return deadlines followed by "How long will that take?").
   - Strict Grounding: Queries unsupported by documents output *"Sorry, I don't have information about that. I can connect you to a colleague who can help."* Never invents policy.
   - Automatic escalation via `create_ticket()` for high-value damages (> ₹10,000), unauthorized orders, repeat contacts ($\ge 2$), and overdue refunds (> 3 business days late).

4. **Gnani Voice Layer (`src/voice.py`)**:
   - Speech-to-Text: `GnaniSTTClient` with `language_code="en-IN"`.
   - Text-to-Speech: `GnaniTTSClient` configured with `model="timbre-v2.5"` and `voice="Kaveri"`.

---

## Installation & Setup

### 1. Clone & Set Up Virtual Environment

```bash
cd Phantom-VoiceOPS

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Credentials

Copy the `.env.example` file to `.env` and insert your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```env
GNANI_API_KEY=your_gnani_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
ENVIRONMENT=production
LOG_LEVEL=INFO
```

---

## Running the Application

### 1. Offline Dataset Validation

Verify the synthetic relational database and knowledge base:

```bash
python validate_dataset.py
```

### 2. Batch Evaluation on Practice Datasets

Run batch evaluation across single questions and multi-turn conversations:

```bash
# Evaluate questions dataset (with TTS audio synthesis for first 5 samples)
python run.py --input-file practice/questions.csv --tts-limit 5

# Evaluate multi-turn conversations dataset
python run.py --input-file practice/conversations.csv

# Run complete evaluation on all practice datasets without TTS
python run.py --no-tts
```

### 3. Processing a Directory of Audio Queries

Process audio files (`.wav` or `.mp3`) through Gnani STT, agent reasoning, and Gnani TTS:

```bash
python run.py --input-dir path/to/audio_folder/ --output-dir sample_outputs/
```

### 4. Outputs & Artifacts

All results are automatically generated into `sample_outputs/`:
- `sample_outputs/results.json`: Complete execution traces with decision metrics and tool calls.
- `sample_outputs/questions_evaluation_summary.csv`: Single-turn test evaluations.
- `sample_outputs/conversations_evaluation_summary.csv`: Multi-turn conversational turn evaluations.
- `sample_outputs/audio/`: Synthesized `.wav` audio replies from Gnani TTS.

---

## Repository Structure

```text
Phantom-VoiceOPS/
├── .env.example              # Template environment variables (no real secrets)
├── .gitignore                # Strictly excludes .env, virtual environments, and cache
├── README.md                 # Project architecture, setup, and usage guide
├── requirements.txt          # Production dependencies
├── run.py                    # Batch runner CLI script
├── validate_dataset.py       # Offline dataset validation suite
├── report.pdf                # Formal 4-page capstone report
├── data/                     # 7 core CSV database tables
├── kb/                       # 21 Markdown policy knowledge base files
├── practice/                 # Evaluation benchmark questions and conversations
├── sample_outputs/           # Evaluation logs, metrics, and synthesized audio
│   ├── audio/                # Generated .wav audio replies
│   ├── conversations_evaluation_summary.csv
│   ├── questions_evaluation_summary.csv
│   └── results.json
├── tools/                    # Tool modules (lookup_order, create_ticket)
│   ├── __init__.py
│   └── tools.py
└── src/                      # Source code
    ├── __init__.py
    ├── agent.py              # Conversational brain, security gates, and tool routing
    ├── db_loader.py          # Pandas CSV joins and business day calendar engine
    ├── kb_loader.py          # Precedence-aware RAG chunking and vector indexing
    └── voice.py              # Gnani STT and TTS API wrappers
```
