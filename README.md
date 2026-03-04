# Syrian ASR & Keyword Extraction Engine (`genehub_model.py`)

A comprehensive, production-ready speech-to-text and semantic keyword extraction engine specifically optimized for the **Syrian Arabic dialect**. 

`genehub_model.py` is designed to power voice-search applications by bridging the gap between colloquial Syrian speech and standardized service directories. It seamlessly transcribes audio using `faster-whisper`, normalizes dialectal variations, identifies user intent, and extracts structured entities (Areas, Products, Services, Devices) to build highly accurate search queries.

## ✨ Key Features

- **Blazing Fast Transcription**: Leverages CTranslate2 and `faster-whisper` for optimized, low-latency Arabic speech recognition on CPU or GPU.
- **Dialect Normalization & Synonyms**: Automatically handles common Syrian terms (e.g., "بدي" → "أريد", "بنشرجي" → "بنشر", "شلون" → "كيف") and phonetic misspellings for robust text processing.
- **Semantic Intent Detection**: Smart regex-based category mapping. Translates conversational queries like *"بدي صلح سيارتي"* (I want to fix my car) into canonical service categories like `"ورشة تصليح سيارات"`.
- **Fuzzy Lexicon Matching**: Categorizes keywords into strictly curated semantic buckets:
  - 📍 `area`: Syrian neighborhoods and cities (e.g., المزة, الشعلان, الحميدية)
  - 🛠️ `service`: Service providers (e.g., مطاعم, أطباء, سباكة وصرف صحي)
  - 🛍️ `product`: Entities and items (e.g., غسالة, لابتوب, بنزين)
  - 🩺 `device`: Medical and technical hardware.
- **YAKE Fallback**: Extracts relevant, generic keywords for out-of-lexicon terms.
- **Built-in Fine-Tuning Pipeline**: End-to-end HuggingFace `Seq2SeqTrainer` script to fine-tune Whisper on custom Arabic CSV datasets and effortlessly export the best checkpoint to a `faster-whisper` compatible format.
- **Webhook / API Integration**: Ready-to-use methods for processing audio and safely POST-ing heavily structured JSON data to external search endpoints.

---

## 🚀 Installation & Prerequisites

Make sure you have Python 3.8+ installed. 

### Dependencies

```bash
pip install torch transformers datasets evaluate
pip install faster-whisper ctranslate2 pandas yake requests
```
*(CUDA is highly recommended for real-time transcription and rapid model training.)*

---

## 💻 Usage: Command Line Interface (CLI)

The script provides an out-of-the-box CLI for both running inferences and training models.

### 1. Run Inference
Extract intents and keywords from an audio file using the default `large-v3` model:
```bash
python genehub_model.py run path/to/audio/file.wav --model_size large-v3
```

### 2. Fine-Tune Whisper Model
Train a custom iteration of the whisper model using labeled `.csv` files:
```bash
python genehub_model.py train \
    --train_csv train_data.csv \
    --eval_csv eval_data.csv \
    --out ./my_custom_model \
    --epochs 3 \
    --lr 1e-5 \
    --model openai/whisper-large-v3
```
This routine trains the model, saves the best Hugging Face checkpoint, and automatically converts it to CTranslate2 format for production use.

---

## 🧩 Usage: Python API

The `SyrianASRKeywordEngine` is highly modular and easy to integrate directly into FastAPI, Flask, or backend job queues.

### Basic Extraction
```python
from genehub_model import SyrianASRKeywordEngine

# Initialize the engine (automatically detects CUDA)
engine = SyrianASRKeywordEngine(model_size="large-v3")

# Transcribe and extract keywords
results = engine.speech_to_keywords("voice_note.ogg")

print(results["text"])            # "بدي روح على الشعلان مشان صلح موبايلي"
print(results["tags"]["intent"])  # ['متاجر إلكترونيات وهواتف', 'صيانة أجهزة عامة']
print(results["tags"]["area"])    # ['الشعلان']
print(results["flat_keywords"])   # Ranked search terms consolidated
```

### Direct Webhook Integration
Ideal for distributed microservices. Transcribe and push the JSON structure immediately to your backend search engine:
```python
response = engine.speech_to_keywords_and_send(
    audio_path="voice_note.ogg", 
    url="https://api.your-app.com/v1/search/voice"
)
```

---

## 🏗️ How the Pipeline Works

1. **VAD & Whisper Transcription**: Filters silences dynamically to prevent hallucinations and uses a specialized Arabic prompt to guide the Whisper decoder.
2. **Text Normalization**: Strips Arabic diacritics, unifies Yeh/Alif variants, strips stopwords, and light-stems words.
3. **Intent Detection**: Analyzes the raw + normalized string using high-confidence Regex triggers to instantly route the query.
4. **Lexicon Tagging**: Scans the text using string-matching and stemming to extract known Neighborhoods, Product Types, and Service Classifications.
5. **YAKE Ranking**: Any trailing critical keywords not found in the lexicons are ranked and clustered into the `"generic"` bucket.
6. **Payload Construction**: Generates a flattened, heavily deduced `search_query` string for optimized elastic-search / SQL behavior.

---

## 🛠 Extensibility

To improve accuracy for your specific use cases, open `genehub_model.py` and modify the following dictionaries:
- `PHONETIC_CORRECTION`: For specific ASR hallucination overrides.
- `SYNONYM_MAP`: To map highly-regional slang back to standard Arabic.
- `INTENT_PATTERNS`: Regex patterns mapping phrases to parent directories.
- `AREAS_LEX`, `PRODUCT_LEX`, `SERVICE_LEX`: Hardcoded sets classifying exact entity hits.
