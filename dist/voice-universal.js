// index.ts
import fs5 from "node:fs";
import os4 from "node:os";

// lib/stt.ts
import fs2 from "node:fs";
import { execSync as execSync2 } from "node:child_process";

// lib/sherpa.ts
import { OfflineRecognizer, OfflineTts, Vad } from "sherpa-onnx";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
var SherpaEngine = class {
  sttRecognizer = null;
  ttsEngine = null;
  vad = null;
  currentSTTModel = null;
  currentTTSModel = null;
  modelsDir;
  // Built-in model registry
  STT_MODELS = {
    "shenava-koochik-int8": {
      id: "shenava-koochik-int8",
      name: "Shenava Koochik (Persian, INT8)",
      type: "transducer",
      encoder: "encoder.int8.onnx",
      decoder: "decoder.int8.onnx",
      joiner: "joiner.int8.onnx",
      tokens: "tokens.txt",
      numThreads: 2,
      sampleRate: 16e3,
      featureDim: 80,
      decodingMethod: "greedy_search",
      modelType: "nemo_transducer",
      languages: ["fa-IR"],
      description: "Best Persian STT, WER 7.5%, INT8 quantized (~131MB encoder)"
    },
    "shenava-koochik-fp32": {
      id: "shenava-koochik-fp32",
      name: "Shenava Koochik (Persian, FP32)",
      type: "transducer",
      encoder: "encoder.onnx",
      decoder: "decoder.onnx",
      joiner: "joiner.onnx",
      tokens: "tokens.txt",
      numThreads: 2,
      sampleRate: 16e3,
      featureDim: 80,
      decodingMethod: "greedy_search",
      modelType: "nemo_transducer",
      languages: ["fa-IR"],
      description: "Full precision Persian STT"
    },
    "shenava-rizeh-int8": {
      id: "shenava-rizeh-int8",
      name: "Shenava Rizeh (Persian, INT8, Fast)",
      type: "nemo_ctc",
      encoder: "model.int8.onnx",
      tokens: "tokens.txt",
      numThreads: 2,
      sampleRate: 16e3,
      languages: ["fa-IR"],
      description: "Lightweight Persian STT, 6.9M params, ~24.5% WER"
    },
    "whisper-tiny": {
      id: "whisper-tiny",
      name: "Whisper Tiny (Multilingual)",
      type: "whisper",
      encoder: "tiny-encoder.onnx",
      decoder: "tiny-decoder.onnx",
      tokens: "tokens.txt",
      numThreads: 2,
      sampleRate: 16e3,
      languages: ["en-US", "fa-IR", "auto", "99+"],
      description: "Fastest Whisper, ~39M params, good for English"
    },
    "whisper-base": {
      id: "whisper-base",
      name: "Whisper Base (Multilingual)",
      type: "whisper",
      encoder: "base-encoder.onnx",
      decoder: "base-decoder.onnx",
      tokens: "tokens.txt",
      numThreads: 2,
      sampleRate: 16e3,
      languages: ["en-US", "fa-IR", "auto", "99+"],
      description: "Balanced Whisper, ~74M params"
    },
    "whisper-small": {
      id: "whisper-small",
      name: "Whisper Small (Multilingual)",
      type: "whisper",
      encoder: "small-encoder.onnx",
      decoder: "small-decoder.onnx",
      tokens: "tokens.txt",
      numThreads: 2,
      sampleRate: 16e3,
      languages: ["en-US", "fa-IR", "auto", "99+"],
      description: "Better accuracy, ~244M params"
    },
    "nemo-parakeet": {
      id: "nemo-parakeet",
      name: "NeMo Parakeet (English, Fast)",
      type: "nemo_ctc",
      encoder: "encoder.onnx",
      tokens: "tokens.txt",
      numThreads: 2,
      sampleRate: 16e3,
      languages: ["en-US"],
      description: "Fast English-only STT"
    }
  };
  TTS_MODELS = {
    "piper-fa-gyro-medium": {
      id: "piper-fa-gyro-medium",
      name: "Piper Persian Gyro (Male)",
      type: "piper",
      model: "vits-piper-fa_IR-gyro-medium.onnx",
      tokens: "tokens.txt",
      dataDir: "espeak-ng-data",
      numThreads: 2,
      voices: [
        { id: "fa_IR-gyro-medium", name: "Gyro (Male)", language: "fa-IR", gender: "male", ageGroup: "adult", style: ["neutral"], sampleRate: 22050, description: "Persian male from ManaTTS" }
      ],
      languages: ["fa-IR"],
      description: "Piper Persian male voice, medium quality"
    },
    "piper-fa-amir-medium": {
      id: "piper-fa-amir-medium",
      name: "Piper Persian Amir (Male, Young)",
      type: "piper",
      model: "vits-piper-fa_IR-amir-medium.onnx",
      tokens: "tokens.txt",
      dataDir: "espeak-ng-data",
      numThreads: 2,
      voices: [
        { id: "fa_IR-amir-medium", name: "Amir (Male, Young)", language: "fa-IR", gender: "male", ageGroup: "young", style: ["neutral", "friendly"], sampleRate: 22050, description: "Younger Persian male voice" }
      ],
      languages: ["fa-IR"],
      description: "Piper Persian younger male voice"
    },
    "piper-en-ryan-high": {
      id: "piper-en-ryan-high",
      name: "Piper English Ryan (Male, High)",
      type: "piper",
      model: "en_US-ryan-high.onnx",
      tokens: "tokens.txt",
      dataDir: "espeak-ng-data",
      numThreads: 2,
      voices: [
        { id: "en_US-ryan-high", name: "Ryan (High Quality)", language: "en-US", gender: "male", ageGroup: "adult", style: ["professional"], sampleRate: 22050 }
      ],
      languages: ["en-US"],
      description: "High quality English male voice"
    },
    "piper-en-lessac-medium": {
      id: "piper-en-lessac-medium",
      name: "Piper English Lessac (Male, Medium)",
      type: "piper",
      model: "en_US-lessac-medium.onnx",
      tokens: "tokens.txt",
      dataDir: "espeak-ng-data",
      numThreads: 2,
      voices: [
        { id: "en_US-lessac-medium", name: "Lessac (Medium)", language: "en-US", gender: "male", ageGroup: "senior", style: ["calm", "neutral"], sampleRate: 22050 }
      ],
      languages: ["en-US"],
      description: "Calm older male English voice"
    },
    "vits-kokoro": {
      id: "vits-kokoro",
      name: "Kokoro 82M (Multi-speaker)",
      type: "vits",
      model: "kokoro-v1.0.onnx",
      tokens: "tokens.txt",
      dataDir: "espeak-ng-data",
      numThreads: 2,
      voices: [
        { id: "af_heart", name: "Heart (Female US)", language: "en-US", gender: "female", ageGroup: "young", style: ["premium", "warm"], sampleRate: 24e3 },
        { id: "af_bella", name: "Bella (Female US)", language: "en-US", gender: "female", ageGroup: "young", style: ["warm", "friendly"], sampleRate: 24e3 },
        { id: "af_nicole", name: "Nicole (Female US)", language: "en-US", gender: "female", ageGroup: "adult", style: ["professional"], sampleRate: 24e3 },
        { id: "am_adam", name: "Adam (Male US)", language: "en-US", gender: "male", ageGroup: "adult", style: ["professional"], sampleRate: 24e3 },
        { id: "am_michael", name: "Michael (Male US)", language: "en-US", gender: "male", ageGroup: "adult", style: ["deep", "calm"], sampleRate: 24e3 },
        { id: "bf_emma", name: "Emma (Female UK)", language: "en-GB", gender: "female", ageGroup: "young", style: ["british"], sampleRate: 24e3 },
        { id: "bm_george", name: "George (Male UK)", language: "en-GB", gender: "male", ageGroup: "adult", style: ["british"], sampleRate: 24e3 }
      ],
      languages: ["en-US", "en-GB", "ja", "zh", "es", "fr", "hi", "it", "pt"],
      description: "82M param multi-lingual TTS, 54 voices"
    }
  };
  VAD_MODELS = {
    "silero-vad": {
      id: "silero-vad",
      name: "Silero VAD",
      model: "silero_vad.onnx",
      threshold: 0.5,
      minSpeechDuration: 250,
      minSilenceDuration: 100,
      windowSize: 512
    }
  };
  constructor(modelsDir) {
    this.modelsDir = modelsDir || path.join(os.homedir(), ".local", "share", "opencode-voice", "models");
    this.ensureModelsDir();
  }
  ensureModelsDir() {
    if (!fs.existsSync(this.modelsDir)) {
      fs.mkdirSync(this.modelsDir, { recursive: true });
    }
  }
  getModelPath(modelFile) {
    const fullPath = path.join(this.modelsDir, modelFile);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`Model file not found: ${fullPath}. Run download:models or place manually in ${this.modelsDir}`);
    }
    return fullPath;
  }
  // ==================== STT ====================
  async loadSTT(modelId, options = {}) {
    if (this.currentSTTModel === modelId && this.sttRecognizer) return;
    const config = { ...this.STT_MODELS[modelId], ...options };
    if (!config) throw new Error(`Unknown STT model: ${modelId}`);
    const modelPath = this.getModelPath(config.encoder);
    const tokensPath = this.getModelPath(config.tokens);
    let recognizer;
    switch (config.type) {
      case "transducer":
        if (!config.decoder || !config.joiner) throw new Error("Transducer needs decoder and joiner");
        recognizer = OfflineRecognizer.fromTransducer({
          encoder: this.getModelPath(config.decoder),
          // Note: sherpa-onnx expects decoder path here for transducer
          decoder: this.getModelPath(config.joiner),
          joiner: this.getModelPath(config.encoder),
          // encoder is actually the main model
          tokens: tokensPath,
          numThreads: config.numThreads || 2,
          sampleRate: config.sampleRate || 16e3,
          featureDim: config.featureDim || 80,
          decodingMethod: config.decodingMethod || "greedy_search",
          modelType: config.modelType || "nemo_transducer"
        });
        break;
      case "nemo_ctc":
        recognizer = OfflineRecognizer.fromNemoCTC({
          model: modelPath,
          tokens: tokensPath,
          numThreads: config.numThreads || 2,
          sampleRate: config.sampleRate || 16e3
        });
        break;
      case "whisper":
        if (!config.decoder) throw new Error("Whisper needs decoder");
        recognizer = OfflineRecognizer.fromWhisper({
          encoder: modelPath,
          decoder: this.getModelPath(config.decoder),
          tokens: tokensPath,
          numThreads: config.numThreads || 2,
          sampleRate: config.sampleRate || 16e3,
          language: "auto",
          task: "transcribe",
          tailPadding: -1
        });
        break;
      default:
        throw new Error(`Unsupported STT type: ${config.type}`);
    }
    this.sttRecognizer = recognizer;
    this.currentSTTModel = modelId;
  }
  async transcribe(audio, sampleRate = 16e3, language) {
    if (!this.sttRecognizer) throw new Error("STT model not loaded. Call loadSTT() first.");
    let processedAudio = audio;
    if (sampleRate !== 16e3) {
      processedAudio = this.resample(audio, sampleRate, 16e3);
    }
    const stream = this.sttRecognizer.createStream();
    stream.acceptWaveform(16e3, processedAudio);
    this.sttRecognizer.decodeStream(stream);
    const result = stream.result;
    return {
      text: result.text.trim(),
      language: language || "auto",
      confidence: 1
      // sherpa-onnx doesn't expose confidence directly
    };
  }
  async *transcribeStreaming(audioChunks2, sampleRate = 16e3) {
    if (!this.sttRecognizer) throw new Error("STT model not loaded");
    const stream = this.sttRecognizer.createStream();
    for await (const chunk of audioChunks2) {
      stream.acceptWaveform(16e3, chunk);
      this.sttRecognizer.decodeStream(stream);
      const result = stream.result;
      if (result.text.trim()) {
        yield { text: result.text.trim(), language: "auto" };
      }
    }
  }
  // ==================== TTS ====================
  async loadTTS(modelId, options = {}) {
    if (this.currentTTSModel === modelId && this.ttsEngine) return;
    const config = { ...this.TTS_MODELS[modelId], ...options };
    if (!config) throw new Error(`Unknown TTS model: ${modelId}`);
    const modelPath = this.getModelPath(config.model);
    const tokensPath = this.getModelPath(config.tokens);
    let tts;
    switch (config.type) {
      case "piper":
      case "vits":
        if (!config.dataDir) throw new Error("Piper/VITS needs dataDir (espeak-ng-data)");
        tts = new OfflineTts({
          model: modelPath,
          tokens: tokensPath,
          dataDir: this.getModelPath(config.dataDir),
          numThreads: config.numThreads || 2
        });
        break;
      case "matcha":
        tts = new OfflineTts({
          model: modelPath,
          tokens: tokensPath,
          numThreads: config.numThreads || 2
        });
        break;
      default:
        throw new Error(`Unsupported TTS type: ${config.type}`);
    }
    this.ttsEngine = tts;
    this.currentTTSModel = modelId;
  }
  async synthesize(text, options) {
    if (!this.ttsEngine) throw new Error("TTS model not loaded. Call loadTTS() first.");
    const audio = this.ttsEngine.generate(text, options.voice, {
      speed: options.speed || 1
    });
    return {
      data: audio.samples,
      sampleRate: audio.sampleRate,
      channels: 1
    };
  }
  async *synthesizeStreaming(text, options) {
    if (!this.ttsEngine) throw new Error("TTS model not loaded");
    const sentences = text.split(/(?<=[.!?])\s+/);
    for (const sentence of sentences) {
      if (sentence.trim()) {
        yield this.synthesize(sentence.trim(), options);
      }
    }
  }
  getAvailableVoices(modelId) {
    if (modelId) {
      return this.TTS_MODELS[modelId]?.voices || [];
    }
    return Object.values(this.TTS_MODELS).flatMap((m) => m.voices);
  }
  // ==================== VAD ====================
  async loadVAD(modelId = "silero-vad") {
    if (this.vad) return;
    const config = this.VAD_MODELS[modelId];
    if (!config) throw new Error(`Unknown VAD model: ${modelId}`);
    this.vad = new Vad({
      model: this.getModelPath(config.model),
      threshold: config.threshold || 0.5,
      minSpeechDuration: config.minSpeechDuration || 250,
      minSilenceDuration: config.minSilenceDuration || 100,
      windowSize: config.windowSize || 512
    });
  }
  async detectVoiceActivity(audio) {
    if (!this.vad) await this.loadVAD();
    const result = this.vad.process(audio);
    return {
      hasSpeech: result.hasSpeech,
      probability: result.probability
    };
  }
  // ==================== Utility ====================
  resample(audio, fromRate, toRate) {
    if (fromRate === toRate) return audio;
    const ratio = fromRate / toRate;
    const newLength = Math.round(audio.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const srcIndex = i * ratio;
      const idx = Math.floor(srcIndex);
      const frac = srcIndex - idx;
      if (idx + 1 < audio.length) {
        result[i] = audio[idx] * (1 - frac) + audio[idx + 1] * frac;
      } else {
        result[i] = audio[idx];
      }
    }
    return result;
  }
  // ==================== Model Management ====================
  listSTTModels() {
    return Object.values(this.STT_MODELS);
  }
  listTTSModels() {
    return Object.values(this.TTS_MODELS);
  }
  listVADModels() {
    return Object.values(this.VAD_MODELS);
  }
  getCurrentSTTModel() {
    return this.currentSTTModel;
  }
  getCurrentTTSModel() {
    return this.currentTTSModel;
  }
  unloadSTT() {
    this.sttRecognizer = null;
    this.currentSTTModel = null;
  }
  unloadTTS() {
    this.ttsEngine = null;
    this.currentTTSModel = null;
  }
  unloadAll() {
    this.unloadSTT();
    this.unloadTTS();
    this.vad = null;
  }
};
var sherpaInstance = null;
function getSherpaEngine(modelsDir) {
  if (!sherpaInstance) {
    sherpaInstance = new SherpaEngine(modelsDir);
  }
  return sherpaInstance;
}

// lib/session.ts
async function getSessionTitle(client, sessionID) {
  if (!sessionID) return "";
  try {
    const result = await client.session.list();
    const session = result.data?.find((s) => s.id === sessionID);
    return session?.title || "";
  } catch {
    return "";
  }
}

// lib/stt.ts
var sttApiEndpoint = null;
var sttApiModel = null;
var sttApiKeyEnv = null;
var tmpDir = "/tmp";
var SHERPA_STT_MODELS = {
  "shenava-koochik-int8": {
    label: "Shenava Koochik (Persian, INT8) \u2605 Best",
    description: "Best Persian STT, WER 7.5%, INT8 quantized (~131MB encoder)",
    languages: ["fa-IR"]
  },
  "shenava-koochik-fp32": {
    label: "Shenava Koochik (Persian, FP32)",
    description: "Full precision Persian STT",
    languages: ["fa-IR"]
  },
  "shenava-rizeh-int8": {
    label: "Shenava Rizeh (Persian, Fast)",
    description: "Lightweight Persian STT, 6.9M params, ~24.5% WER",
    languages: ["fa-IR"]
  },
  "whisper-tiny": {
    label: "Whisper Tiny (Multilingual, Fast)",
    description: "Fastest Whisper, ~39M params, good for English",
    languages: ["en-US", "fa-IR", "auto", "99+"]
  },
  "whisper-base": {
    label: "Whisper Base (Multilingual)",
    description: "Balanced Whisper, ~74M params",
    languages: ["en-US", "fa-IR", "auto", "99+"]
  },
  "whisper-small": {
    label: "Whisper Small (Multilingual)",
    description: "Better accuracy, ~244M params",
    languages: ["en-US", "fa-IR", "auto", "99+"]
  },
  "nemo-parakeet": {
    label: "NeMo Parakeet (English, Fast)",
    description: "Fast English-only STT",
    languages: ["en-US"]
  }
};
var DEFAULT_STT_MODEL = "shenava-koochik-int8";
var DEFAULT_LANGUAGE = "auto";
var LANGUAGES = {
  auto: { label: "Auto-detect" },
  "fa-IR": { label: "Persian (Farsi)" },
  "en-US": { label: "English (US)" },
  "en-GB": { label: "English (UK)" },
  "zh": { label: "Chinese" },
  "ja": { label: "Japanese" },
  "ko": { label: "Korean" },
  "de": { label: "German" },
  "fr": { label: "French" },
  "es": { label: "Spanish" },
  "ru": { label: "Russian" },
  "ar": { label: "Arabic" }
};
var sttDefaultLanguage = DEFAULT_LANGUAGE;
var sherpaEngine = null;
var currentSTTModel = DEFAULT_STT_MODEL;
var recording = false;
var processing = false;
var audioChunks = [];
function getSherpa() {
  if (!sherpaEngine) {
    sherpaEngine = getSherpaEngine();
  }
  return sherpaEngine;
}
async function loadSTTModel(modelId = currentSTTModel) {
  const engine = getSherpa();
  await engine.loadSTT(modelId);
  currentSTTModel = modelId;
}
var STT_SYSTEM_PROMPT = `You are a speech-to-text normalizer for a coding assistant CLI.

Clean up raw whisper transcription into a clear, well-punctuated prompt. Rules:
- Fix punctuation, capitalization, and grammar
- Remove filler words (um, uh, like, you know, etc.)
- Keep technical terms, file names, and code references exact
- If the user is dictating code, format it appropriately
- Use the session context above to resolve ambiguous references (e.g. "that function", "the file", "it")
- Output ONLY the cleaned text, nothing else
- Do not add any commentary or explanation
- Keep the user's intent and meaning intact

CRITICAL DOMAIN CORRECTIONS - Fix common STT homophone errors in software engineering contexts:
- "locks" -> "logs" (unless explicitly talking about mutexes/concurrency)
- "note" / "no" -> "node"
- "app and" -> "append"
- "sink" -> "sync"
- "a sink" -> "async"
- "doc" / "talker" -> "docker"
- "cash" -> "cache"
- "rap" -> "wrap"
- "Jason" -> "JSON"
- "get" -> "Git"
- "react" -> "React"
- "types creep" / "type script" -> "TypeScript"
- "bite" -> "byte"
- "string" -> "String"
- "int" -> "Int"
- "bullion" -> "boolean"

Rely heavily on context to fix words that sound similar to programming terminology.`;
function registerSTT(api, kv, complete, prompts, opts, logger) {
  const client = api.client;
  const systemPrompt = prompts?.stt || STT_SYSTEM_PROMPT;
  function toast(message, variant = "info") {
    api.ui.toast({ message, variant, duration: 3e3 });
  }
  if (opts?.sttEndpoint) {
    sttApiEndpoint = opts.sttEndpoint;
    sttApiModel = opts.sttModel || "whisper-large-v3-turbo";
    sttApiKeyEnv = opts.sttApiKeyEnv || null;
    logger?.log("STT", `Configured STT API endpoint=${sttApiEndpoint} model=${sttApiModel}`, "debug");
  }
  if (opts?.sttLanguage) {
    sttDefaultLanguage = opts.sttLanguage;
  }
  logger?.log("STT", `Default language=${sttDefaultLanguage}`, "debug");
  tmpDir = opts?.tmpDir || "/tmp";
  try {
    fs2.mkdirSync(tmpDir, { recursive: true });
  } catch (err) {
    logger?.log("STT", `Failed to create tmpDir ${tmpDir}: ${err.message}`, "warn");
  }
  loadSTTModel(currentSTTModel).catch((err) => {
    logger?.log("STT", `Failed to pre-load model: ${err.message}`, "warn");
  });
  return [
    {
      title: "STT: Record/Transcribe",
      value: "stt.record",
      description: "Toggle recording; press again to stop and transcribe",
      keybind: "ctrl+r",
      slash: { name: "stt-record" },
      onSelect() {
        if (processing) {
          toast("STT busy, please wait...");
          return;
        }
        if (recording) {
          toast("Stopping, transcribing...");
          api.event.emit("stt:stopRecording", { submit: false });
        } else {
          api.event.emit("stt:startRecording");
          toast("Recording... press again to transcribe");
        }
      }
    },
    {
      title: "STT: Submit Recording",
      value: "stt.submit",
      description: "Stop recording, transcribe, and submit prompt",
      keybind: "<leader>r",
      slash: { name: "stt-submit" },
      onSelect() {
        if (processing) {
          toast("STT busy, please wait...");
          return;
        }
        if (!recording) {
          toast("No recording in progress", "warning");
          return;
        }
        toast("Stopping, transcribing...");
        api.event.emit("stt:stopRecording", { submit: true });
      }
    },
    {
      title: "STT: Cancel Recording",
      value: "stt.stop",
      description: "Cancel current recording",
      slash: { name: "stt-stop" },
      onSelect() {
        if (recording) {
          recording = false;
          audioChunks = [];
          logger?.log("STT", "Recording cancelled", "debug");
          toast("Recording cancelled");
        }
      }
    },
    {
      title: "STT: Select Model",
      value: "stt.model",
      description: "Choose STT model (Shenava, Whisper, NeMo)",
      slash: { name: "stt-model" },
      async onSelect() {
        const current = currentSTTModel;
        api.ui.dialog.replace(
          () => api.ui.DialogSelect({
            title: "Select STT Model",
            current,
            options: Object.entries(SHERPA_STT_MODELS).map(([key, v]) => ({
              title: v.label,
              value: key,
              description: v.description,
              onSelect() {
                currentSTTModel = key;
                kv.set("stt.model", key);
                loadSTTModel(key).then(() => {
                  toast(`STT model: ${v.label}`);
                }).catch((err) => {
                  toast(`Failed to load model: ${err.message}`, "error");
                });
                api.ui.dialog.clear();
              }
            }))
          })
        );
      }
    },
    {
      title: "STT: Select Language",
      value: "stt.language",
      description: "Choose transcription language",
      slash: { name: "stt-language" },
      onSelect() {
        const current = kv.get("stt.language") || sttDefaultLanguage;
        api.ui.dialog.replace(
          () => api.ui.DialogSelect({
            title: "Select transcription language",
            current,
            options: Object.entries(LANGUAGES).map(([key, v]) => ({
              title: v.label,
              value: key,
              onSelect() {
                kv.set("stt.language", key);
                sttDefaultLanguage = key;
                toast(`Language: ${v.label}`);
                api.ui.dialog.clear();
              }
            }))
          })
        );
      }
    },
    {
      title: "STT: Configure API",
      value: "stt.api",
      description: "Configure STT API endpoint (OpenAI, Groq, etc.)",
      slash: { name: "stt-api" },
      async onSelect() {
        const endpoint = await api.ui.dialog.input({
          title: "STT API Endpoint",
          placeholder: "https://api.openai.com/v1 or https://api.groq.com/openai/v1",
          value: sttApiEndpoint || ""
        });
        if (!endpoint) return;
        const model = await api.ui.dialog.input({
          title: "STT Model",
          placeholder: "whisper-large-v3-turbo",
          value: sttApiModel || ""
        });
        if (!model) return;
        const keyEnv = await api.ui.dialog.input({
          title: "API Key Environment Variable",
          placeholder: "OPENAI_API_KEY or GROQ_API_KEY",
          value: sttApiKeyEnv || ""
        });
        sttApiEndpoint = endpoint;
        sttApiModel = model;
        sttApiKeyEnv = keyEnv || null;
        kv.set("stt.endpoint", endpoint);
        kv.set("stt.apiModel", model);
        if (sttApiKeyEnv) kv.set("stt.apiKeyEnv", sttApiKeyEnv);
        toast(`STT API configured: ${endpoint}`);
      }
    }
  ];
}

// lib/tts.ts
import fs4 from "node:fs";
import path3 from "node:path";
import os3 from "node:os";

// lib/mixed-lang.ts
import fs3 from "node:fs";
import path2 from "node:path";
import os2 from "node:os";
var DEFAULT_OPTIONS = {
  enabled: true,
  primaryLang: "en-US",
  secondaryLangs: ["fa-IR"],
  defaultVoicePerLang: {
    "fa-IR": "fa_IR-gyro-medium",
    "en-US": "af_heart",
    "en-GB": "bf_emma",
    "auto": "af_heart"
  },
  enInFaStrategy: "pronounce",
  faInEnStrategy: "transliterate",
  customRules: {},
  segmentStrategy: "per-sentence"
};
var PERSIAN_DIGITS = ["\u06F0", "\u06F1", "\u06F2", "\u06F3", "\u06F4", "\u06F5", "\u06F6", "\u06F7", "\u06F8", "\u06F9"];
var ARABIC_DIGITS = ["\u0660", "\u0661", "\u0662", "\u0663", "\u0664", "\u0665", "\u0666", "\u0667", "\u0668", "\u0669"];
var TECH_TERMS_PERSIAN_PRONUNCIATION = {
  "API": "\u0627\u06D2 \u067E\u06CC \u0622\u0626\u06CC",
  "GitHub": "\u06AF\u06CC\u062A\u200C\u0647\u0627\u0628",
  "Git": "\u06AF\u06CC\u062A",
  "TypeScript": "\u062A\u0627\u06CC\u067E\u200C\u0627\u0633\u06A9\u0631\u06CC\u067E\u062A",
  "JavaScript": "\u062C\u0627\u0648\u0627\u0627\u0633\u06A9\u0631\u06CC\u067E\u062A",
  "Python": "\u067E\u0627\u06CC\u062A\u0648\u0646",
  "React": "\u0631\u06CC\u200C\u0627\u06A9\u062A",
  "Node": "\u0646\u0648\u062F",
  "npm": "\u0627\u0646\u200C\u067E\u06CC\u200C\u0627\u0645",
  "Docker": "\u062F\u0627\u06A9\u0631",
  "Kubernetes": "\u06A9\u0648\u0628\u0650\u0631\u0646\u062A\u06CC\u0632",
  "AI": "\u0627\u06D2\u200C\u0622\u06CC",
  "ML": "\u0627\u0645\u200C\u0627\u0644",
  "LLM": "\u0627\u0644\u200C\u0627\u0644\u200C\u0622\u0645",
  "GPU": "\u062C\u06CC\u200C\u067E\u06CC\u200C\u06CC\u0648",
  "CPU": "\u0633\u06CC\u200C\u067E\u06CC\u200C\u06CC\u0648",
  "RAM": "\u0631\u0645",
  "SSD": "\u0627\u0633\u200C\u0627\u0633\u200C\u062F\u06CC",
  "JSON": "\u062C\u0650\u06CC\u200C\u0633\u0627\u0646",
  "HTTP": "\u0627\u0686\u200C\u062A\u06CC\u200C\u062A\u06CC\u200C\u067E\u06CC",
  "HTTPS": "\u0627\u0686\u200C\u062A\u06CC\u200C\u062A\u06CC\u200C\u067E\u06CC\u200C\u0627\u0633",
  "URL": "\u06CC\u0648\u200C\u0627\u0631\u200C\u0627\u0644",
  "SQL": "\u0627\u0633\u200C\u06A9\u06CC\u0648\u200C\u0627\u0644",
  "NoSQL": "\u0646\u0627\u0633\u200C\u06A9\u06CC\u0648\u200C\u0627\u0644",
  "REST": "\u0440\u0435\u0441\u0442",
  "GraphQL": "\u06AF\u0631\u0627\u0641\u200C\u06A9\u06CC\u0648\u200C\u0627\u0644",
  "WebSocket": "\u0648\u0628\u200C\u0633\u0648\u06A9\u062A",
  "OAuth": "\u0627\u064F\u0627\u062A",
  "JWT": "\u062C\u0650\u06CC\u200C\u0648\u0650\u06CC\u200C\u062A\u06CC",
  "CSS": "\u0633\u06CC\u200C\u0627\u0633\u200C\u0627\u0633",
  "HTML": "\u0627\u0686\u200C\u062A\u06CC\u200C\u0627\u0645\u200C\u0627\u0644",
  "VS Code": "\u0648\u06CC\u200C\u0627\u0633 \u06A9\u062F",
  "CLI": "\u0633\u06CC\u200C\u0627\u0644\u200C\u0622\u06CC",
  "GUI": "\u062C\u06CC\u200C\u06CC\u0648\u200C\u0622\u06CC",
  "IDE": "\u0622\u06CC\u200C\u062F\u06CC\u200C\u0627\u06CC",
  "PR": "\u067E\u06CC\u200C\u0627\u0631",
  "CI/CD": "\u0633\u06CC\u200C\u0622\u06CC\u200C\u0633\u06CC\u200C\u062F\u06CC",
  "DevOps": "\u062F\u0650\u0648\u0627\u067E\u0633",
  "Agile": "\u0627\u062C\u0627\u06CC\u0644",
  "Scrum": "\u0627\u0633\u06A9\u0631\u0645",
  "Sprint": "\u0627\u0633\u067E\u0631\u06CC\u0646\u062A",
  "Kanban": "\u06A9\u0627\u0646\u0628\u0627\u0646",
  "Jira": "\u062C\u06CC\u0631\u0627",
  "Confluence": "\u06A9\u0627\u0646\u0641\u0644\u0648\u0626\u0646\u0633",
  "Slack": "\u0627\u0633\u064E\u0644\u06A9",
  "Discord": "\u062F\u0650\u0633\u06A9\u0648\u0631\u062F",
  "Zoom": "\u0632\u0648\u0645",
  "Google": "\u06AF\u0648\u06AF\u0644",
  "Microsoft": "\u0645\u0627\u06CC\u06A9\u0631\u0648\u0633\u0627\u0641\u062A",
  "Amazon": "\u0622\u0645\u0627\u0632\u0648\u0646",
  "AWS": "\u0627\u06D2\u200C\u0648\u0650\u06CC\u200C\u0627\u0650\u0633",
  "Azure": "\u0627\u0698\u0648\u0631",
  "GCP": "\u062C\u06CC\u200C\u0633\u06CC\u200C\u067E\u06CC",
  "Linux": "\u0644\u06CC\u0646\u0648\u06A9\u0633",
  "Ubuntu": "\u0627\u0648\u0628\u0648\u0646\u062A\u0648",
  "Debian": "\u062F\u0628\u06CC\u0627\u0646",
  "Arch": "\u0622\u0631\u0686",
  "Fedora": "\u0641\u062F\u0648\u0631\u0627",
  "Windows": "\u0648\u06CC\u0646\u062F\u0648\u0632",
  "macOS": "\u0645\u06A9\u200C\u0627\u0648\u200C\u0627\u0633",
  "iOS": "\u0622\u06CC\u200C\u0627\u0648\u200C\u0627\u0633",
  "Android": "\u0627\u0646\u062F\u0631\u0648\u06CC\u062F",
  "Chrome": "\u06A9\u0631\u0648\u0645",
  "Firefox": "\u0641\u0627\u06CC\u0631\u0641\u0627\u06A9\u0633",
  "Safari": "\u0633\u0627\u0641\u0627\u0631\u06CC",
  "Edge": "\u0627\u0650\u062C",
  "GitLab": "\u06AF\u06CC\u062A\u200C\u0644\u0628",
  "Bitbucket": "\u0628\u06CC\u062A\u200C\u0628\u0627\u06A9\u062A",
  "Docker": "\u062F\u0627\u06A9\u0631",
  "Kubernetes": "\u06A9\u0648\u0628\u0650\u0631\u0646\u062A\u06CC\u0632",
  "Terraform": "\u062A\u0631\u0627\u0641\u0631\u0645",
  "Ansible": "\u0622\u0646\u0633\u06CC\u0628\u0644",
  "Prometheus": "\u067E\u0631\u0648\u062A\u0626\u0648\u0633",
  "Grafana": "\u06AF\u0631\u0627\u0641\u0627\u0646\u0627",
  "Elasticsearch": "\u0627\u0644\u0633\u062A\u06CC\u06A9\u200C\u0633\u0631\u0686",
  "Redis": "\u0631\u062F\u06CC\u0633",
  "PostgreSQL": "\u067E\u0633\u062A\u06AF\u0631\u0633\u200C\u06A9\u06CC\u0648\u200C\u0627\u0644",
  "MySQL": "\u0645\u0627\u06CC\u200C\u0627\u0633\u200C\u06A9\u06CC\u0648\u200C\u0627\u0644",
  "MongoDB": "\u0645\u06CC\u0646\u06AF\u0648\u200C\u062F\u06CC\u200C\u0628\u06CC",
  "GraphQL": "\u06AF\u0631\u0627\u0641\u200C\u06A9\u06CC\u0648\u200C\u0627\u0644",
  "gRPC": "\u062C\u06CC\u200C\u0622\u0631\u200C\u067E\u06CC\u200C\u0633\u06CC",
  "WebAssembly": "\u0648\u0628\u200C\u0627\u0633\u0645\u0628\u0644\u06CC",
  "Rust": "\u0631\u0627\u0633\u062A",
  "Go": "\u06AF\u0648",
  "Java": "\u062C\u0627\u0648\u0627",
  "Kotlin": "\u06A9\u0627\u062A\u0644\u06CC\u0646",
  "Swift": "\u0633\u0648\u0626\u06CC\u0641\u062A",
  "Dart": "\u062F\u0627\u0631\u062A",
  "Flutter": "\u0641\u0644\u0627\u062A\u0631",
  "React Native": "\u0631\u06CC\u200C\u0627\u06A9\u062A \u0646\u06CC\u062A\u06CC\u0648",
  "Electron": "\u0627\u0644\u06A9\u062A\u0631\u0648\u0646",
  "Tauri": "\u062A\u0627\u0626\u0648\u0631\u06CC",
  "Next.js": "\u0646\u06A9\u0633\u062A\u200C\u062C\u06CC\u200C\u0627\u0633",
  "Nuxt": "\u0646\u0627\u06A9\u0633\u062A",
  "Vite": "\u0648\u0627\u06CC\u062A",
  "Webpack": "\u0648\u0628\u200C\u067E\u06A9",
  "Rollup": "\u0631\u0648\u0644\u200C\u0622\u067E",
  "ESLint": "\u0627\u0650\u0633\u200C\u0644\u06CC\u0646\u062A",
  "Prettier": "\u067E\u0631\u06CC\u062A\u06CC\u0650\u0631",
  "Jest": "\u062C\u0650\u0633\u062A",
  "Vitest": "\u0648\u06CC\u062A\u0650\u0633\u062A",
  "Cypress": "\u0633\u0627\u06CC\u067E\u0631\u0633",
  "Playwright": "\u067E\u0644\u06CC\u200C\u0631\u0627\u06CC\u062A\u0631",
  "Storybook": "\u0627\u0633\u062A\u0648\u0631\u06CC\u200C\u0628\u0648\u06A9",
  "Tailwind": "\u062A\u0650\u06CC\u0644\u200C\u0648\u0650\u0646\u062F",
  "Bootstrap": "\u0628\u0648\u062A\u200C\u0627\u0633\u062A\u0631\u067E",
  "Material UI": "\u0645\u062A\u0631\u06CC\u0627\u0644 \u06CC\u0648\u200C\u0622\u06CC",
  "Chakra UI": "\u0686\u0627\u06A9\u0631\u0627 \u06CC\u0648\u200C\u0622\u06CC",
  "Prisma": "\u067E\u0631\u06CC\u0632\u0645\u0627",
  "Drizzle": "\u062F\u0631\u06CC\u0632\u0644",
  "TypeORM": "\u062A\u0627\u06CC\u067E\u200C\u0627\u0648\u200C\u0622\u0631\u0627\u0645",
  "Sequelize": "\u0633\u06CC\u06A9\u0648\u0626\u0644\u200C\u0622\u06CC\u0632",
  "Mongoose": "\u0645\u0646\u06AF\u0648\u0633",
  "Socket.io": "\u0633\u0627\u06A9\u062A\u200C\u0622\u06CC\u200C\u0627\u0648",
  "GraphQL": "\u06AF\u0631\u0627\u0641\u200C\u06A9\u06CC\u0648\u200C\u0627\u0644",
  "Apollo": "\u0627\u067E\u0648\u0644\u0648",
  "Relay": "\u0631\u06CC\u0644",
  "URQL": "\u06CC\u0648\u200C\u0622\u0631\u200C\u06A9\u06CC\u0648\u200C\u0627\u0644",
  "React Query": "\u0631\u06CC\u200C\u0627\u06A9\u062A \u06A9\u064F\u0648\u064E\u0631\u06CC",
  "SWR": "\u0627\u0633\u200C\u062F\u0628\u0650\u0644\u200C\u06CC\u0648\u200C\u0622\u0631",
  "Zustand": "\u0632\u0648\u0633\u062A\u0627\u0646\u062F",
  "Redux": "\u0631\u06CC\u200C\u062F\u0627\u06A9\u0633",
  "MobX": "\u0645\u0627\u0628\u200C\u0627\u06CC\u06A9\u0633",
  "Recoil": "\u0631\u06CC\u06A9\u0648\u06CC\u0644",
  "Jotai": "\u062C\u0648\u062A\u0627\u06CC\u06CC",
  "Valtio": "\u0648\u0627\u0644\u062A\u06CC\u0648",
  "Signals": "\u0633\u06CC\u06AF\u0646\u0627\u0644\u200C\u0647\u0627",
  "Solid": "\u0633\u0648\u0644\u06CC\u062F",
  "Svelte": "\u0633\u0644\u0648\u062A",
  "Vue": "\u0648\u06CC\u0648",
  "Nuxt": "\u0646\u0627\u06A9\u0633\u062A",
  "Astro": "\u0622\u0633\u062A\u0631\u0648",
  "Remix": "\u0631\u0650\u0645\u0650\u06A9\u0633",
  "Gatsby": "\u06AF\u062A\u0633\u0628\u06CC",
  "Eleventy": "\u0627\u0644\u0650\u0648\u0646\u062A\u06CC",
  "Hugo": "\u0647\u064F\u06AF\u0648",
  "Jekyll": "\u062C\u0650\u06A9\u06CC\u0644",
  "Netlify": "\u0646\u062A\u0644\u06CC\u0641\u0627\u06CC",
  "Vercel": "\u0648\u0631\u0633\u0650\u0644",
  "Cloudflare": "\u06A9\u0644\u0627\u0648\u062F\u0641\u0644\u0631",
  "Firebase": "\u0641\u0627\u06CC\u0631\u0628\u06CC\u0633",
  "Supabase": "\u0633\u0648\u067E\u0627\u0628\u06CC\u0633",
  "PlanetScale": "\u067E\u0644\u0650\u0646\u062A\u0650\u0627\u0633\u06A9\u06CC\u0644",
  "Neon": "\u0646\u0626\u0648\u0646",
  "Turso": "\u062A\u0648\u0631\u0633\u0648",
  "Fly.io": "\u0641\u0644\u0627\u06CC\u200C\u0622\u06CC\u200C\u0627\u0648",
  "Railway": "\u0631\u0650\u06CC\u0644\u200C\u0648\u0650\u06CC",
  "Render": "\u0631\u0650\u0646\u062F\u0650\u0631",
  "Heroku": "\u0647\u0631\u0648\u06A9\u0648",
  "DigitalOcean": "\u062F\u06CC\u062C\u06CC\u062A\u0627\u0644\u2011\u0627\u064F\u0634\u0650\u0646",
  "Linode": "\u0644\u06CC\u0646\u0648\u062F",
  "Vultr": "\u0648\u064F\u0644\u062A\u0631",
  "Hetzner": "\u0647\u0650\u062A\u0632\u0646\u0631",
  "Scaleway": "\u0627\u0633\u0650\u06A9\u06CC\u0644\u200C\u0648\u0650\u06CC",
  "OVH": "\u0627\u0648\u0650\u06CC\u200C\u0627\u0686\u200C\u0648\u06CC",
  "OpenAI": "\u0627\u0648\u067E\u0646\u200C\u0627\u06D2\u200C\u0622\u06CC",
  "Anthropic": "\u0622\u0646\u062A\u0631\u0648\u067E\u06CC\u06A9",
  "Google AI": "\u06AF\u0648\u06AF\u0644 \u0627\u06D2\u200C\u0622\u06CC",
  "Gemini": "\u062C\u0650\u0645\u06CC\u0646\u06CC",
  "Claude": "\u06A9\u0644\u0648\u062F",
  "GPT": "\u062C\u06CC\u200C\u067E\u06CC\u200C\u062A\u06CC",
  "DALL-E": "\u062F\u0627\u0644-\u0627\u06CC",
  "Midjourney": "\u0645\u06CC\u062F\u062C\u0631\u0646\u06CC",
  "Stable Diffusion": "\u0627\u0633\u062A\u06CC\u0628\u0644 \u062F\u06CC\u0641\u06CC\u0648\u0698\u0646",
  "Whisper": "\u0648\u0650\u0633\u067E\u0650\u0631",
  "Piper": "\u067E\u0627\u06CC\u067E\u0631",
  "Kokoro": "\u06A9\u0648\u06A9\u0648\u0631\u0648",
  "Sherpa": "\u0634\u0650\u0631\u067E\u0627",
  "Shenava": "\u0634\u0646\u0627\u0648\u0627",
  "VITS": "\u0648\u0650\u06CC\u200C\u062A\u06CC\u200C\u0627\u0633",
  "XTTS": "\u0627\u0650\u06A9\u0633\u200C\u062A\u06CC\u200C\u062A\u06CC\u200C\u0627\u0633",
  "Coqui": "\u06A9\u064F\u06A9\u06CC",
  "MMS-TTS": "\u0627\u0645\u200C\u0627\u0645\u200C\u0627\u0633-\u062A\u06CC\u200C\u062A\u06CC\u200C\u0627\u0633",
  "Facebook": "\u0641\u06CC\u0633\u200C\u0628\u0648\u06A9",
  "Meta": "\u0645\u0650\u062A\u0627",
  "NVIDIA": "\u0627\u0646\u0648\u06CC\u062F\u06CC\u0627",
  "AMD": "\u0627\u0650\u06CC\u200C\u0645\u0650\u062F\u06CC",
  "Intel": "\u0627\u06CC\u0646\u062A\u0644",
  "ARM": "\u0627\u0650\u0631\u0645",
  "CUDA": "\u06A9\u064F\u0648\u062F\u0627",
  "ROCm": "\u0631\u0627\u06A9\u0645",
  "Vulkan": "\u0648\u064F\u0644\u06A9\u0627\u0646",
  "DirectX": "\u062F\u0627\u06CC\u0930\u0947\u06A9\u062A\u200C\u0627\u0650\u06A9\u0633",
  "OpenGL": "\u0627\u0648\u067E\u0646\u200C\u062C\u06CC\u200C\u0627\u0644",
  "WebGL": "\u0648\u0628\u200C\u062C\u06CC\u200C\u0627\u0644",
  "WebGPU": "\u0648\u0628\u200C\u062C\u06CC\u200C\u067E\u06CC\u200C\u06CC\u0648",
  "WASM": "\u0648\u064E\u0633\u0645",
  "WebAssembly": "\u0648\u0628\u200C\u0627\u0633\u0645\u0628\u0644\u06CC",
  "LLVM": "\u0627\u0650\u0644\u200C\u0627\u0644\u200C\u0648\u06CC\u200C\u0627\u0645",
  "Clang": "\u06A9\u0644\u0627\u064E\u0646\u06AF",
  "GCC": "\u062C\u06CC\u200C\u0633\u06CC\u200C\u0633\u06CC",
  "MSVC": "\u0627\u0645\u200C\u0627\u0633\u200C\u0648\u06CC\u200C\u0633\u06CC",
  "Rust": "\u0631\u0627\u0633\u062A",
  "Cargo": "\u06A9\u0627\u0631\u06AF\u0648",
  "Crates.io": "\u06A9\u0631\u06CC\u062A\u0633\u200C\u062F\u0627\u062A\u200C\u0622\u06CC\u200C\u0627\u0648",
  "Tokio": "\u062A\u0648\u06A9\u06CC\u0648",
  "Async": "\u0627\u0650\u0633\u0646\u0650\u06A9",
  "Await": "\u0627\u0650\u0648\u064E\u06CC\u062A",
  "Future": "\u0641\u06CC\u0648\u0686\u0650\u0631",
  "Stream": "\u0627\u0633\u062A\u0631\u06CC\u0645",
  "Iterator": "\u0627\u06CC\u062A\u0650\u0631\u0650\u06CC\u062A\u0650\u0631",
  "Trait": "\u062A\u0631\u0650\u06CC\u062A",
  "Struct": "\u0627\u0633\u062A\u0631\u0627\u06A9\u062A",
  "Enum": "\u0627\u0650\u0646\u0627\u0645",
  "Match": "\u0645\u064E\u0686",
  "Option": "\u0627\u064F\u067E\u0634\u0646",
  "Result": "\u0631\u0650\u0632\u0627\u0644\u062A",
  "Vec": "\u0648\u064E\u06A9",
  "HashMap": "\u0647\u0634\u200C\u0645\u067E",
  "BTreeMap": "\u0628\u06CC\u200C\u0E15\u0E23\u0E35\u200C\u0645\u067E",
  "String": "\u0627\u0633\u062A\u0631\u06CC\u0646\u06AF",
  "str": "\u0627\u0650\u0633\u062A\u0631",
  "Box": "\u0628\u0627\u06A9\u0633",
  "Arc": "\u0627\u0650\u0631\u0643",
  "Rc": "\u0622\u0631\u200C\u0633\u06CC",
  "RefCell": "\u0631\u0641\u200C\u0633\u06CC\u0644",
  "Mutex": "\u0645\u06CC\u0648\u062A\u0650\u06A9\u0633",
  "RwLock": "\u0622\u0631\u200C\u0648\u0627\u0644\u0627\u06A9",
  "Channel": "\u0686\u0646\u0644",
  "Sender": "\u0633\u0650\u0646\u062F\u0631",
  "Receiver": "\u0631\u06CC\u200C\u0633\u06CC\u0648\u0631",
  "Select": "\u0633\u0650\u0644\u0650\u06A9\u062A",
  "Join": "\u062C\u064F\u0648\u06CC\u0646",
  "Spawn": "\u0627\u0633\u067E\u0627\u0646",
  "Task": "\u062A\u0633\u06A9",
  "Runtime": "\u0631\u0627\u0646\u200C\u062A\u0627\u06CC\u0645",
  "Executor": "\u0627\u0650\u06AF\u0632\u0650\u06A9\u06CC\u0648\u062A\u0650\u0631",
  "Blocking": "\u0628\u0644\u0627\u06A9\u06CC\u0646\u06AF",
  "Sync": "\u0633\u0650\u0646\u06A9",
  "Send": "\u0633\u0650\u0646\u062F",
  "Pin": "\u067E\u06CC\u0646",
  "Unpin": "\u0627\u064E\u0646\u200C\u067E\u06CC\u0646",
  "Drop": "\u062F\u0631\u0627\u067E",
  "Clone": "\u06A9\u0644\u0648\u0646",
  "Copy": "\u06A9\u067E\u06CC",
  "PartialEq": "\u067E\u0627\u0631\u0634\u06CC\u0627\u0644\u200C\u0627\u0650\u06A9\u200C\u06CC\u0648",
  "Eq": "\u0627\u0650\u06A9\u200C\u06CC\u0648",
  "Hash": "\u0647\u0634",
  "Debug": "\u062F\u06CC\u0628\u0627\u06AF",
  "Display": "\u062F\u0650\u0633\u067E\u0644\u06CC",
  "Default": "\u062F\u06CC\u0641\u0627\u0644\u062A",
  "From": "\u0641\u0631\u0627\u0645",
  "Into": "\u0627\u0650\u0646\u062A\u064F",
  "TryFrom": "\u062A\u0631\u0627\u06CC\u200C\u0641\u0631\u0627\u0645",
  "TryInto": "\u062A\u0631\u0627\u06CC\u200C\u0627\u0650\u0646\u062A\u064F",
  "AsRef": "\u0627\u0650\u0632\u200C\u0631\u0641",
  "AsMut": "\u0627\u0650\u0632\u200C\u0645\u064E\u062A",
  "Deref": "\u062F\u06CC\u0631\u0650\u0641",
  "DerefMut": "\u062F\u06CC\u0631\u0650\u0641\u200C\u0645\u064E\u062A",
  "Borrow": "\u0628\u0627\u0631\u0648",
  "BorrowMut": "\u0628\u0627\u0631\u0648\u200C\u0645\u064E\u062A",
  "Cow": "\u06A9\u0627\u0648",
  "PhantomData": "\u0641\u064E\u0646\u062A\u064E\u0645\u200C\u062F\u0650\u06CC\u062A\u0627",
  "NonZero": "\u0646\u0646\u200C\u0632\u0650\u0631\u0648",
  "MaybeUninit": "\u0645\u0650\u06CC\u0628\u0650\u06CC\u200C\u0627\u064E\u0646\u0650\u06CC\u0646\u0650\u06CC\u062A",
  "ManuallyDrop": "\u0645\u064E\u0646\u0650\u06CC\u0648\u0650\u06CC\u200C\u062F\u0650\u0631\u064E\u067E",
  "Cell": "\u0633\u0644",
  "Ref": "\u0631\u0641",
  "UnsafeCell": "\u0627\u064E\u0646\u200C\u0633\u0650\u06CC\u0641\u200C\u0633\u0650\u0644",
  "Sync": "\u0633\u0650\u0646\u06A9",
  "Send": "\u0633\u0650\u0646\u062F",
  "Unpin": "\u0627\u064E\u0646\u200C\u067E\u06CC\u0646",
  "Freeze": "\u0641\u0631\u06CC\u0632",
  "Frozen": "\u0641\u0631\u064E\u0632\u0646",
  "Pin": "\u067E\u06CC\u0646",
  "Unpin": "\u0627\u064E\u0646\u200C\u067E\u06CC\u0646",
  "Pointer": "\u067E\u0648\u06CC\u0646\u062A\u0631",
  "Reference": "\u0631\u0641\u0650\u0631\u0650\u0646\u0633",
  "RawPointer": "\u0631\u0627\u0648\u200C\u067E\u0648\u06CC\u0646\u062A\u0631",
  "Function": "\u0641\u0627\u0646\u06A9\u0634\u0646",
  "Closure": "\u06A9\u0644\u064F\u062C\u0631",
  "Fn": "\u0627\u0650\u0641\u0650\u0646",
  "FnMut": "\u0627\u0650\u0641\u0650\u0646\u200C\u0645\u064E\u062A",
  "FnOnce": "\u0627\u0650\u0641\u0650\u0646-\u0627\u064F\u0646\u0633",
  "Generator": "\u062C\u0650\u0650\u0646\u0650\u0631\u0650\u06CC\u062A\u0650\u0631",
  "AsyncGenerator": "\u0627\u0650\u0633\u0646\u0650\u06A9\u200C\u062C\u0650\u0650\u0646\u0650\u0631\u0650\u06CC\u062A\u0650\u0631",
  "Stream": "\u0627\u0633\u062A\u0631\u06CC\u0645",
  "Sink": "\u0633\u0650\u0646\u06A9",
  "Future": "\u0641\u06CC\u0648\u0686\u0650\u0631",
  "Poll": "\u067E\u064F\u0644",
  "Context": "\u06A9\u0627\u0646\u0650\u062A\u0650\u06A9\u0633\u062A",
  "Waker": "\u0648\u0650\u06CC\u06A9\u0650\u0631",
  "Task": "\u062A\u0633\u06A9",
  "Executor": "\u0627\u0650\u06AF\u0632\u0650\u06A9\u06CC\u0648\u062A\u0650\u0631",
  "Runtime": "\u0631\u0627\u0646\u200C\u062A\u0627\u06CC\u0645",
  "Blocking": "\u0628\u0644\u0627\u06A9\u06CC\u0646\u06AF",
  "Spawn": "\u0627\u0633\u067E\u0627\u0646",
  "JoinHandle": "\u062C\u064F\u0648\u06CC\u0646\u0650\u0647\u064E\u0646\u062F\u0644",
  "AbortHandle": "\u0627\u064E\u0628\u064F\u0631\u062A\u0650\u0647\u064E\u0646\u062F\u0644",
  "AbortRegistration": "\u0627\u064E\u0628\u064F\u0631\u062A\u200C\u0631\u0650\u062C\u0650\u0633\u062A\u0631\u0650\u06CC\u0634\u0650\u0646",
  "LocalSet": "\u0644\u064F\u06A9\u0650\u0644\u200C\u0633\u062A",
  "Enter": "\u0627\u0650\u0646\u062A\u0650\u0631",
  "Exit": "\u0627\u0650\u06AF\u0632\u0650\u062A",
  "Park": "\u067E\u0627\u0631\u06A9",
  "Unpark": "\u0627\u064E\u0646\u200C\u067E\u0627\u0631\u06A9",
  "Coop": "\u06A9\u064F\u067E",
  "Budget": "\u0628\u064E\u062C\u0650\u062A",
  "Yield": "\u06CC\u06CC\u0644\u062F",
  "Ready": "\u0631\u0650\u062F\u06CC",
  "Pending": "\u067E\u0650\u0646\u062F\u06CC\u0646\u06AF",
  "Poll": "\u067E\u064F\u0644",
  "Context": "\u06A9\u0627\u0646\u0650\u062A\u0650\u06A9\u0633\u062A",
  "Waker": "\u0648\u0650\u06CC\u06A9\u0650\u0631",
  "Wake": "\u0648\u0650\u06CC\u06A9",
  "WakeByRef": "\u0648\u0650\u06CC\u06A9\u200C\u0628\u064E\u0627\u06CC\u200C\u0631\u0650\u0641",
  "Clone": "\u06A9\u0644\u0648\u0646",
  "Copy": "\u06A9\u067E\u06CC",
  "PartialEq": "\u067E\u0627\u0631\u0634\u06CC\u0627\u0644\u200C\u0627\u0650\u06A9\u200C\u06CC\u0648",
  "Eq": "\u0627\u0650\u06A9\u200C\u06CC\u0648",
  "Hash": "\u0647\u0634",
  "Debug": "\u062F\u06CC\u0628\u0627\u06AF",
  "Display": "\u062F\u0650\u0633\u067E\u0644\u06CC",
  "Default": "\u062F\u06CC\u0641\u0627\u0644\u062A",
  "From": "\u0641\u0631\u0627\u0645",
  "Into": "\u0627\u0650\u0646\u062A\u064F",
  "TryFrom": "\u062A\u0631\u0627\u06CC\u200C\u0641\u0631\u0627\u0645",
  "TryInto": "\u062A\u0631\u0627\u06CC\u200C\u0627\u0650\u0646\u062A\u064F",
  "AsRef": "\u0627\u0650\u0632\u200C\u0631\u0641",
  "AsMut": "\u0627\u0650\u0632\u200C\u0645\u064E\u062A",
  "Deref": "\u062F\u06CC\u0631\u0650\u0641",
  "DerefMut": "\u062F\u06CC\u0631\u0650\u0641\u200C\u0645\u064E\u062A",
  "Borrow": "\u0628\u0627\u0631\u0648",
  "BorrowMut": "\u0628\u0627\u0631\u0648\u200C\u0645\u064E\u062A",
  "Cow": "\u06A9\u0627\u0648",
  "PhantomData": "\u0641\u064E\u0646\u062A\u064E\u0645\u200C\u062F\u0650\u06CC\u062A\u0627",
  "NonZero": "\u0646\u0646\u200C\u0632\u0650\u0631\u0648",
  "MaybeUninit": "\u0645\u0650\u06CC\u0628\u0650\u06CC\u200C\u0627\u064E\u0646\u0650\u06CC\u0646\u0650\u06CC\u062A",
  "ManuallyDrop": "\u0645\u064E\u0646\u0650\u06CC\u0648\u0650\u06CC\u200C\u062F\u0650\u0631\u064E\u067E",
  "Cell": "\u0633\u0644",
  "Ref": "\u0631\u0641",
  "UnsafeCell": "\u0627\u064E\u0646\u200C\u0633\u0650\u06CC\u0641\u200C\u0633\u0650\u0644",
  "Sync": "\u0633\u0650\u0646\u06A9",
  "Send": "\u0633\u0650\u0646\u062F",
  "Unpin": "\u0627\u064E\u0646\u200C\u067E\u06CC\u0646",
  "Freeze": "\u0641\u0631\u06CC\u0632",
  "Frozen": "\u0641\u0631\u064E\u0632\u0646",
  "Pin": "\u067E\u06CC\u0646",
  "Unpin": "\u0627\u064E\u0646\u200C\u067E\u06CC\u0646",
  "Pointer": "\u067E\u0648\u06CC\u0646\u062A\u0631",
  "Reference": "\u0631\u0641\u0650\u0631\u0650\u0646\u0633",
  "RawPointer": "\u0631\u0627\u0648\u200C\u067E\u0648\u06CC\u0646\u062A\u0631",
  "Function": "\u0641\u0627\u0646\u06A9\u0634\u0646",
  "Closure": "\u06A9\u0644\u064F\u062C\u0631",
  "Fn": "\u0627\u0650\u0641\u0650\u0646",
  "FnMut": "\u0627\u0650\u0641\u0650\u0646\u200C\u0645\u064E\u062A",
  "FnOnce": "\u0627\u0650\u0641\u0650\u0646-\u0627\u064F\u0646\u0633",
  "Generator": "\u062C\u0650\u0650\u0646\u0650\u0631\u0650\u06CC\u062A\u0650\u0631",
  "AsyncGenerator": "\u0627\u0650\u0633\u0646\u0650\u06A9\u200C\u062C\u0650\u0650\u0646\u0650\u0631\u0650\u06CC\u062A\u0650\u0631",
  "Stream": "\u0627\u0633\u062A\u0631\u06CC\u0645",
  "Sink": "\u0633\u0650\u0646\u06A9",
  "Future": "\u0641\u06CC\u0648\u0686\u0650\u0631",
  "Poll": "\u067E\u064F\u0644",
  "Context": "\u06A9\u0627\u0646\u0650\u062A\u0650\u06A9\u0633\u062A",
  "Waker": "\u0648\u0650\u06CC\u06A9\u0650\u0631",
  "Wake": "\u0648\u0650\u06CC\u06A9",
  "WakeByRef": "\u0648\u0650\u06CC\u06A9\u200C\u0628\u064E\u0627\u06CC\u200C\u0631\u0650\u0641"
};
function loadCustomRules() {
  const configPath = path2.join(os2.homedir(), ".config", "opencode", "voice", "custom-rules.json");
  if (fs3.existsSync(configPath)) {
    try {
      return JSON.parse(fs3.readFileSync(configPath, "utf-8"));
    } catch {
    }
  }
  return {};
}
function detectLanguage(text) {
  const persianChars = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/;
  const hasPersian = persianChars.test(text);
  const hasEnglish = /[a-zA-Z]/.test(text);
  if (hasPersian && hasEnglish) return "mixed";
  if (hasPersian) return "fa-IR";
  if (hasEnglish) return "en-US";
  return "auto";
}
function detectLanguageSegments(text) {
  const sentences = text.split(/(?<=[.!?؟])\s+/);
  const segments = [];
  for (const sentence of sentences) {
    if (!sentence.trim()) continue;
    const lang = detectLanguage(sentence);
    const primaryLang = lang === "mixed" ? "fa-IR" : lang;
    const voiceId = DEFAULT_OPTIONS.defaultVoicePerLang[primaryLang] || DEFAULT_OPTIONS.defaultVoicePerLang.auto;
    segments.push({
      text: sentence.trim(),
      language: primaryLang,
      confidence: lang === "mixed" ? 0.7 : 0.95,
      voiceId
    });
  }
  return segments;
}
function applyCustomRules(text, rules) {
  let result = text;
  for (const [from, to] of Object.entries(rules)) {
    const regex = new RegExp(`\\b${escapeRegExp(from)}\\b`, "gi");
    result = result.replace(regex, to);
  }
  return result;
}
function escapeRegExp(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
function normalizeNumbers(text, lang) {
  if (lang === "fa-IR") {
    return text.replace(/[0-9]/g, (d) => PERSIAN_DIGITS[parseInt(d)]);
  }
  if (lang === "en-US" || lang === "en-GB") {
    return text.replace(/[۰-۹]/g, (d) => PERSIAN_DIGITS.indexOf(d).toString()).replace(/[٠-٩]/g, (d) => ARABIC_DIGITS.indexOf(d).toString());
  }
  return text;
}
function processMixedLanguage(text, options = {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const rules = { ...TECH_TERMS_PERSIAN_PRONUNCIATION, ...loadCustomRules(), ...opts.customRules };
  let processedText = applyCustomRules(text, rules);
  const segments = detectLanguageSegments(processedText);
  return segments.map((seg) => ({
    ...seg,
    text: normalizeNumbers(seg.text, seg.language)
  }));
}
function concatAudio(buffers) {
  if (buffers.length === 0) {
    return { data: new Float32Array(0), sampleRate: 22050, channels: 1 };
  }
  if (buffers.length === 1) return buffers[0];
  const sampleRate = buffers[0].sampleRate;
  const totalLength = buffers.reduce((sum, b) => sum + b.data.length, 0);
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const buf of buffers) {
    result.set(buf.data, offset);
    offset += buf.data.length;
  }
  return { data: result, sampleRate, channels: 1 };
}

// lib/tts.ts
var VOICE_REGISTRY_PATH = path3.join(os3.homedir(), ".config", "opencode", "voice", "voice-registry.json");
var voiceRegistry = null;
function loadVoiceRegistry() {
  if (voiceRegistry) return voiceRegistry;
  try {
    if (fs4.existsSync(VOICE_REGISTRY_PATH)) {
      voiceRegistry = JSON.parse(fs4.readFileSync(VOICE_REGISTRY_PATH, "utf-8"));
    } else {
      voiceRegistry = { providers: {}, defaultVoicePerLanguage: {}, fallbackChain: {} };
    }
  } catch {
    voiceRegistry = { providers: {}, defaultVoicePerLanguage: {}, fallbackChain: {} };
  }
  return voiceRegistry;
}
function getVoiceRegistry() {
  if (!voiceRegistry) loadVoiceRegistry();
  return voiceRegistry;
}
function getAllVoices() {
  const registry = getVoiceRegistry();
  const voices = [];
  for (const [providerId, provider] of Object.entries(registry.providers || {})) {
    for (const [voiceId, voice] of Object.entries(provider.voices || {})) {
      voices.push({
        id: voiceId,
        name: voice.name,
        language: voice.language,
        gender: voice.gender,
        ageGroup: voice.ageGroup,
        style: voice.style,
        sampleRate: voice.sampleRate,
        description: voice.description,
        modelId: voice.modelId,
        providerId
      });
    }
  }
  return voices;
}
function getVoiceInfo(voiceId) {
  const voices = getAllVoices();
  return voices.find((v) => v.id === voiceId) || null;
}
function getDefaultVoiceForLanguage(lang) {
  const registry = getVoiceRegistry();
  return registry.defaultVoicePerLanguage?.[lang] || registry.defaultVoicePerLanguage?.auto || null;
}
var SYSTEM_AUTO = `You are a text-to-speech narrator for a coding assistant CLI. Your job is to convert the assistant's markdown output into natural spoken text that is useful and pleasant to listen to.

You have three modes depending on the content complexity:

1. NARRATE - For simple explanations, short answers, and conversational responses. Convert to natural spoken text, normalizing code references for speech.
   - camelCase/PascalCase identifiers: split into words (parseConfig -> "parse config")
   - File paths: use just the filename (src/utils/helpers.ts -> "helpers dot ts")
   - Short code snippets in backticks: read them naturally
   - Keep the narrative flow intact

2. SUMMARIZE - For responses with significant code blocks, multiple file changes, or complex technical details. Provide a brief spoken summary of what was done and tell the user to check the screen.
   - Mention what was changed and why
   - Do not try to describe code blocks verbatim
   - End with something like "check the details on your screen" or "take a look at the output for the specifics"

3. NOTIFY - For very short confirmations, status updates, or acknowledgments. Keep it to one brief sentence.

Choose the appropriate mode based on the content. Most responses with code blocks should use SUMMARIZE mode. Simple Q&A or short explanations use NARRATE. Build results, "done", confirmations use NOTIFY.

Output ONLY the spoken text. Nothing else. No mode labels. No commentary.`;
var SYSTEM_MANUAL = `You are a text-to-speech reader for a coding assistant. The user has explicitly requested this text be read aloud. Read the prose content faithfully and in detail.

Rules:
- Read all prose text naturally and completely
- Code identifiers: split camelCase/PascalCase/snake_case into words (parseConfig -> "parse config", my_variable -> "my variable")
- File paths: read just the filename with extension (src/utils/helpers.ts -> "helpers dot ts")
- Line references: keep as is ("line 42")
- URLs: say "a link" or just the domain name
- Code blocks: skip entirely, just say "code block" or "code snippet"
- Error codes: expand naturally (ECONNREFUSED -> "connection refused")
- Shell commands: read them naturally (npm test -> "npm test")
- List items: read each item
- Remove markdown formatting but preserve all the informational content
- Do NOT summarize. Do NOT say "check the screen". Read everything that is prose.
- Output ONLY the spoken text`;
async function getTurnAssistantText(client, api) {
  const route = api.route.current;
  if (route.name !== "session") return null;
  const sessionID = route.params.sessionID;
  const stateMessages = api.state.session.messages(sessionID);
  if (!stateMessages || stateMessages.length === 0) return null;
  const assistantIDs = [];
  for (let i = stateMessages.length - 1; i >= 0; i--) {
    if (stateMessages[i].role === "user") break;
    if (stateMessages[i].role === "assistant") {
      assistantIDs.unshift(stateMessages[i].id);
    }
  }
  if (assistantIDs.length === 0) return null;
  const allText = [];
  for (const msgID of assistantIDs) {
    try {
      const fullMsg = await client.session.message({ sessionID, messageID: msgID }, { throwOnError: true }).then((r) => r.data);
      const textParts = (fullMsg?.parts || []).filter((p) => p.type === "text");
      const text = textParts.map((p) => p.text || "").join("\n\n").trim();
      if (text) allText.push(text);
    } catch {
    }
  }
  if (allText.length === 0) return null;
  return {
    lastMessageID: assistantIDs[assistantIDs.length - 1],
    text: allText.join("\n\n")
  };
}
var sherpaEngine2 = null;
var currentTTSModel = "piper-fa-gyro-medium";
function getSherpa2() {
  if (!sherpaEngine2) {
    sherpaEngine2 = getSherpaEngine();
  }
  return sherpaEngine2;
}
async function loadTTSModel(modelId = currentTTSModel) {
  const engine = getSherpa2();
  await engine.loadTTS(modelId);
  currentTTSModel = modelId;
}
function registerTTS(api, kv, complete, prompts, logger) {
  const client = api.client;
  const systemAuto = prompts?.ttsAuto || SYSTEM_AUTO;
  const systemManual = prompts?.ttsManual || SYSTEM_MANUAL;
  function toast(message, variant = "info") {
    api.ui.toast({ message, variant, duration: 3e3 });
  }
  loadTTSModel(kv.get("tts.model") || currentTTSModel).catch((err) => {
    logger?.log?.("TTS", `Failed to pre-load TTS model: ${err.message}`, "warn");
  });
  async function normalizeForSpeech(text, systemPrompt) {
    logger?.log?.("TTS", `Normalizing speech chars=${text.length}`, "debug");
    return complete({
      system: systemPrompt,
      prompt: `Convert for text-to-speech:

${text}`,
      config: { maxTokens: 4096 }
    });
  }
  async function speak(text, options = {}) {
    if (!text) return;
    const line = text.replace(/\n/g, " ").trim();
    if (!line) return;
    const voiceId = options.voice || kv.get("tts.voice") || getDefaultVoiceForLanguage("fa-IR") || "fa_IR-gyro-medium";
    const speed = options.speed || 1;
    logger?.log?.("TTS", `Speak requested chars=${line.length} voice=${voiceId} speed=${speed}`, "debug");
    try {
      const engine = getSherpa2();
      await loadTTSModel(kv.get("tts.model") || currentTTSModel);
      const audio = await engine.synthesize(line, { voice: voiceId, speed });
      await playAudio(audio);
      logger?.log?.("TTS", "Playback finished", "debug");
    } catch (err) {
      logger?.log?.("TTS", `TTS error: ${err.message}`, "error");
      toast(`TTS error: ${err.message}`, "error");
    }
  }
  async function speakMixed(text, options = {}) {
    if (!text) return;
    const speed = options.speed || 1;
    logger?.log?.("TTS", `Mixed-language speak chars=${text.length}`, "debug");
    try {
      const engine = getSherpa2();
      await loadTTSModel(kv.get("tts.model") || currentTTSModel);
      const segments = processMixedLanguage(text, {
        enabled: true,
        primaryLang: "en-US",
        secondaryLangs: ["fa-IR"],
        defaultVoicePerLang: {
          "en-US": "af_heart",
          "fa-IR": "fa_IR-gyro-medium"
        },
        segmentStrategy: "per-sentence"
      });
      const audioChunks2 = [];
      for (const seg of segments) {
        if (seg.text.trim()) {
          const audio = await engine.synthesize(seg.text, { voice: seg.voiceId, speed });
          audioChunks2.push(audio);
        }
      }
      const combined = concatAudio(audioChunks2);
      await playAudio(combined);
    } catch (err) {
      logger?.log?.("TTS", `Mixed TTS error: ${err.message}`, "error");
      toast(`TTS error: ${err.message}`, "error");
    }
  }
  async function playAudio(audio) {
    return new Promise((resolve, reject) => {
      if (process.platform === "win32") {
        playWindows(audio, resolve, reject);
      } else if (process.platform === "darwin") {
        playMacOS(audio, resolve, reject);
      } else {
        playLinux(audio, resolve, reject);
      }
    });
  }
  function playWindows(audio, resolve, reject) {
    const wavBuffer = audioBufferToWav(audio);
    const tempFile = path3.join(os3.tmpdir(), `opencode-tts-${Date.now()}.wav`);
    fs4.writeFileSync(tempFile, wavBuffer);
    const psScript = `
      $player = New-Object System.Media.SoundPlayer
      $player.SoundLocation = "${tempFile.replace(/\\/g, "\\\\")}"
      $player.PlaySync()
      Remove-Item "${tempFile.replace(/\\/g, "\\\\")}"
    `;
    try {
      execSync(`powershell -NoProfile -Command "${psScript}"`, { stdio: "ignore" });
      resolve();
    } catch (err) {
      reject(err);
    }
  }
  function playMacOS(audio, resolve, reject) {
    const wavBuffer = audioBufferToWav(audio);
    const tempFile = path3.join(os3.tmpdir(), `opencode-tts-${Date.now()}.wav`);
    fs4.writeFileSync(tempFile, wavBuffer);
    try {
      execSync(`afplay "${tempFile}"`, { stdio: "ignore" });
      fs4.unlinkSync(tempFile);
      resolve();
    } catch (err) {
      reject(err);
    }
  }
  function playLinux(audio, resolve, reject) {
    const wavBuffer = audioBufferToWav(audio);
    const tempFile = path3.join(os3.tmpdir(), `opencode-tts-${Date.now()}.wav`);
    fs4.writeFileSync(tempFile, wavBuffer);
    const players = ["mpv", "ffplay", "paplay", "aplay", "play"];
    for (const player of players) {
      try {
        execSync(`which ${player}`, { stdio: "ignore" });
        execSync(`${player} "${tempFile}"`, { stdio: "ignore" });
        fs4.unlinkSync(tempFile);
        resolve();
        return;
      } catch {
      }
    }
    reject(new Error("No audio player found. Install mpv, ffplay, paplay, aplay, or sox (play)"));
  }
  function audioBufferToWav(audio) {
    const int16Data = new Int16Array(audio.data.length);
    for (let i = 0; i < audio.data.length; i++) {
      const s = Math.max(-1, Math.min(1, audio.data[i]));
      int16Data[i] = s < 0 ? s * 32768 : s * 32767;
    }
    const buffer = Buffer.alloc(44 + int16Data.length * 2);
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    view.setUint32(0, 1380533830, false);
    view.setUint32(4, 36 + int16Data.length * 2, true);
    view.setUint32(8, 1463899717, false);
    view.setUint32(12, 1718449184, false);
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, audio.sampleRate, true);
    view.setUint32(28, audio.sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    view.setUint32(36, 1684108385, false);
    view.setUint32(40, int16Data.length * 2, true);
    for (let i = 0; i < int16Data.length; i++) {
      view.setInt16(44 + i * 2, int16Data[i], true);
    }
    return buffer;
  }
  async function speakWithSessionPrefix(sessionID, message, suffix) {
    const sessionTitle = await getSessionTitle(client, sessionID);
    const parts = [];
    if (sessionTitle) parts.push(`Session: ${sessionTitle}.`);
    parts.push(message);
    if (suffix) parts.push(suffix);
    await speak(parts.join(" "));
  }
  let lastSpokenMessageID = null;
  let wasBusy = false;
  api.event.on("session.status", (event) => {
    if (event.properties?.status?.type === "busy") wasBusy = true;
  });
  api.event.on("session.idle", async (event) => {
    if (kv.get("tts.mode", "off") !== "on") return;
    if (!wasBusy) return;
    wasBusy = false;
    const sessionID = event.properties?.sessionID;
    const result = await getTurnAssistantText(client, api);
    if (!result || !result.text) return;
    if (result.lastMessageID === lastSpokenMessageID) return;
    lastSpokenMessageID = result.lastMessageID;
    toast("Normalizing response...");
    const llmResult = await normalizeForSpeech(result.text, systemAuto);
    if (!llmResult.text) {
      logger?.log?.("TTS", `Auto normalization failed: ${llmResult.error}`, "warn");
      toast(`TTS normalization failed: ${llmResult.error}`, "warning");
      return;
    }
    logger?.log?.("TTS", `Auto normalization succeeded chars=${llmResult.text.length}`, "debug");
    await speakWithSessionPrefix(sessionID, llmResult.text, "Ready for your input.");
  });
  api.event.on("permission.asked", async (event) => {
    if (kv.get("tts.mode", "off") !== "on") return;
    await speakWithSessionPrefix(
      event.properties?.sessionID,
      "Permission requested. Please check your screen."
    );
  });
  api.event.on("question.asked", async (event) => {
    if (kv.get("tts.mode", "off") !== "on") return;
    await speakWithSessionPrefix(
      event.properties?.sessionID,
      "A question needs your answer. Please check your screen."
    );
  });
  async function speakLastResponse() {
    const result = await getTurnAssistantText(client, api);
    if (!result || !result.text) {
      toast("No assistant response to speak", "warning");
      return;
    }
    toast("Normalizing response...");
    const llmResult = await normalizeForSpeech(result.text, systemManual);
    if (!llmResult.text) {
      logger?.log?.("TTS", `Manual normalization failed: ${llmResult.error}`, "warn");
      toast(`TTS normalization failed: ${llmResult.error}`, "warning");
      return;
    }
    logger?.log?.("TTS", `Manual normalization succeeded chars=${llmResult.text.length}`, "debug");
    toast("Speaking last response");
    await speak(llmResult.text);
  }
  async function normalizeForSpeech(text, systemPrompt) {
    logger?.log?.("TTS", `Normalizing speech chars=${text.length}`, "debug");
    return complete({
      system: systemPrompt,
      prompt: `Convert for text-to-speech:

${text}`,
      config: { maxTokens: 4096 }
    });
  }
  const voices = getAllVoices();
  return [
    {
      title: "TTS: Speak Last Response",
      value: "tts.speak-last",
      description: "Read the last assistant response aloud (detailed)",
      keybind: "<leader>s",
      slash: { name: "tts-speak" },
      onSelect() {
        speakLastResponse();
      }
    },
    {
      title: "TTS: Toggle Auto Mode",
      value: "tts.mode",
      description: "Toggle auto text-to-speech on/off",
      keybind: "<leader>v",
      slash: { name: "tts-mode" },
      onSelect() {
        const current = kv.get("tts.mode", "off");
        const next = current === "on" ? "off" : "on";
        kv.set("tts.mode", next);
        toast(next === "on" ? "TTS auto mode on" : "TTS auto mode off");
      }
    },
    {
      title: "TTS: Stop Playback",
      value: "tts.stop",
      description: "Stop current TTS playback",
      keybind: "escape",
      slash: { name: "tts-stop" },
      onSelect() {
        toast("TTS stopped (not implemented)");
      }
    },
    {
      title: "TTS: Select Voice",
      value: "tts.voice",
      description: "Choose TTS voice with preview",
      slash: { name: "tts-voice" },
      async onSelect() {
        const current = kv.get("tts.voice", "fa_IR-gyro-medium");
        const allVoices = getAllVoices();
        const byLang = {};
        for (const v of allVoices) {
          if (!byLang[v.language]) byLang[v.language] = [];
          byLang[v.language].push(v);
        }
        const options = [
          { title: "\u{1F50D} Filter by language", value: "__filter_lang__", disabled: true },
          { title: "\u{1F50D} Filter by gender", value: "__filter_gender__", disabled: true },
          { title: "\u{1F50D} Filter by style", value: "__filter_style__", disabled: true },
          { title: "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500", value: "__divider__", disabled: true }
        ];
        for (const [lang, langVoices] of Object.entries(byLang)) {
          options.push({ title: `\u{1F4C1} ${lang}`, value: `__group_${lang}__`, disabled: true });
          for (const v of langVoices) {
            const genderIcon = v.gender === "male" ? "\u{1F468}" : v.gender === "female" ? "\u{1F469}" : "\u{1F464}";
            const ageIcon = v.ageGroup === "young" ? "\u{1F9D2}" : v.ageGroup === "adult" ? "\u{1F9D1}" : v.ageGroup === "senior" ? "\u{1F474}" : "";
            options.push({
              title: `${genderIcon} ${v.name} ${ageIcon}`,
              value: v.id,
              description: `${v.language} \u2022 ${v.style?.join(", ") || "neutral"} \u2022 ${v.sampleRate}Hz`,
              onSelect: async () => {
                await previewVoice(v.id);
                const confirmed = await api.ui.dialog.confirm({
                  title: `Use ${v.name}?`,
                  message: `Language: ${v.language}
Gender: ${v.gender}
Style: ${v.style?.join(", ")}`
                });
                if (confirmed) {
                  kv.set("tts.voice", v.id);
                  toast(`Voice set to ${v.name}`);
                }
                api.ui.dialog.clear();
              }
            });
          }
        }
        api.ui.dialog.replace(
          () => api.ui.DialogSelect({
            title: "Select TTS Voice",
            current,
            options
          })
        );
      }
    },
    {
      title: "TTS: Select Model",
      value: "tts.model",
      description: "Choose TTS model (Piper, VITS, Kokoro)",
      slash: { name: "tts-model" },
      async onSelect() {
        const engine = getSherpa2();
        const models = engine.listTTSModels();
        const current = currentTTSModel;
        api.ui.dialog.replace(
          () => api.ui.DialogSelect({
            title: "Select TTS Model",
            current,
            options: models.map((m) => ({
              title: m.name,
              value: m.id,
              description: m.description,
              onSelect: async () => {
                currentTTSModel = m.id;
                kv.set("tts.model", m.id);
                await loadTTSModel(m.id);
                toast(`TTS model: ${m.name}`);
                api.ui.dialog.clear();
              }
            }))
          })
        );
      }
    },
    {
      title: "TTS: Preview Voice",
      value: "tts.preview",
      description: "Preview current voice with sample text",
      slash: { name: "tts-preview" },
      async onSelect() {
        const voiceId = kv.get("tts.voice", "fa_IR-gyro-medium");
        await previewVoice(voiceId);
      }
    }
  ];
  async function previewVoice(voiceId) {
    const sampleTexts = {
      "fa-IR": "\u0633\u0644\u0627\u0645\u060C \u0627\u06CC\u0646 \u06CC\u06A9 \u067E\u06CC\u0634\u200C\u0646\u0645\u0627\u06CC\u0634 \u0635\u0648\u062A\u06CC \u0627\u0633\u062A. \u0627\u06CC\u0646 \u0433\u043E\u043B\u043E\u0441 \u067E\u0627\u0631\u0633\u06CC \u0628\u0631\u0627\u06CC \u062A\u0633\u062A \u0627\u0646\u062A\u062E\u0627\u0628 \u0634\u062F\u0647 \u0627\u0633\u062A.",
      "en-US": "Hello, this is a voice preview. This English voice has been selected for testing.",
      "en-GB": "Hello, this is a voice preview. This British English voice has been selected for testing."
    };
    const voiceInfo = getVoiceInfo(voiceId);
    const lang = voiceInfo?.language || "fa-IR";
    const text = sampleTexts[lang] || sampleTexts["fa-IR"];
    toast(`Playing preview for ${voiceInfo?.name || voiceId}...`);
    await speak(text, { voice: voiceId });
  }
}

// lib/mode-manager.ts
import { EventEmitter } from "node:events";
var DEFAULT_CONFIG = {
  mode: "manual",
  wakeWord: {
    engine: "openwakeword",
    keywords: ["hey jarvis", "jarvis", "\u0628\u06CC\u062F\u0627\u0631 \u0634\u0648"],
    sensitivity: 0.6,
    customKeywords: {}
  },
  auto: {
    vadSensitivity: 0.5,
    minSpeechDuration: 500,
    maxSilenceDuration: 2e3,
    autoSubmit: true,
    continuousListening: true
  },
  manual: {
    pushToTalkKey: "ctrl+r",
    holdToTalk: false,
    doubleTapTimeout: 300
  }
};
var VoiceModeManager = class extends EventEmitter {
  config;
  sherpaEngine;
  recordingState;
  vadInterval = null;
  audioContext = null;
  mediaStream = null;
  audioWorklet = null;
  wakeWordDetector = null;
  // openwakeword instance
  lastKeyPress = 0;
  constructor(sherpaEngine3, config = {}) {
    super();
    this.sherpaEngine = sherpaEngine3;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.recordingState = {
      isRecording: false,
      startTime: 0,
      audioChunks: [],
      silenceStart: 0,
      hasSpeech: false
    };
  }
  // ==================== Mode Management ====================
  setMode(mode) {
    const oldMode = this.config.mode;
    this.config.mode = mode;
    this.emit("modeChange", { oldMode, newMode: mode });
    if (oldMode === "auto" && mode !== "auto") {
      this.stopAutoListening();
    }
    if (mode === "auto") {
      this.startAutoListening();
    }
  }
  getMode() {
    return this.config.mode;
  }
  toggleMode() {
    const modes = ["off", "manual", "auto"];
    const currentIndex = modes.indexOf(this.config.mode);
    const nextIndex = (currentIndex + 1) % modes.length;
    this.setMode(modes[nextIndex]);
  }
  // ==================== AUTO Mode ====================
  async startAutoListening() {
    if (this.config.mode !== "auto") return;
    await this.sherpaEngine.loadVAD();
    await this.initAudioInput();
    this.emit("autoListeningStarted");
    this.startVADLoop();
  }
  stopAutoListening() {
    if (this.vadInterval) {
      clearInterval(this.vadInterval);
      this.vadInterval = null;
    }
    this.stopRecording();
    this.closeAudioInput();
    this.emit("autoListeningStopped");
  }
  startVADLoop() {
    this.vadInterval = setInterval(async () => {
      if (!this.recordingState.isRecording) return;
      if (this.recordingState.hasSpeech && this.recordingState.silenceStart > 0) {
        const silenceDuration = Date.now() - this.recordingState.silenceStart;
        if (silenceDuration >= this.config.auto.maxSilenceDuration) {
          await this.processRecording();
        }
      }
    }, 100);
  }
  // ==================== MANUAL Mode ====================
  async handlePushToTalk(isKeyDown) {
    if (this.config.mode !== "manual") return;
    const now = Date.now();
    if (isKeyDown) {
      if (this.config.manual.holdToTalk) {
        if (!this.recordingState.isRecording) {
          await this.startRecording();
        }
      } else {
        if (this.recordingState.isRecording) {
          await this.stopRecording();
        } else {
          await this.startRecording();
        }
      }
      this.lastKeyPress = now;
    } else {
      if (this.config.manual.holdToTalk && this.recordingState.isRecording) {
        await this.stopRecording();
      }
    }
  }
  async handleDoubleTap() {
    if (this.config.mode !== "manual") return;
    const now = Date.now();
    if (now - this.lastKeyPress < this.config.manual.doubleTapTimeout) {
      this.toggleMode();
      this.lastKeyPress = 0;
    }
  }
  // ==================== Recording ====================
  async startRecording() {
    if (this.recordingState.isRecording) return;
    await this.initAudioInput();
    this.recordingState = {
      isRecording: true,
      startTime: Date.now(),
      audioChunks: [],
      silenceStart: 0,
      hasSpeech: false
    };
    this.emit("recordingStart", { timestamp: this.recordingState.startTime });
    this.captureAudio();
  }
  async stopRecording() {
    if (!this.recordingState.isRecording) return;
    this.recordingState.isRecording = false;
    this.emit("recordingStop", { duration: Date.now() - this.recordingState.startTime });
    if (this.recordingState.audioChunks.length > 0) {
      await this.processRecording();
    }
  }
  async processRecording() {
    if (this.recordingState.audioChunks.length === 0) return;
    const totalLength = this.recordingState.audioChunks.reduce((sum, c) => sum + c.length, 0);
    const audioData = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.recordingState.audioChunks) {
      audioData.set(chunk, offset);
      offset += chunk.length;
    }
    this.recordingState.audioChunks = [];
    this.recordingState.silenceStart = 0;
    this.recordingState.hasSpeech = false;
    try {
      this.emit("transcribing");
      const result = await this.sherpaEngine.transcribe(audioData, 16e3);
      this.emit("transcribed", result);
      if (result.text.trim()) {
        if (this.config.mode === "auto" && this.config.auto.autoSubmit) {
          this.emit("submit", result.text);
        } else {
          this.emit("textReady", result.text);
        }
      }
    } catch (err) {
      this.emit("error", err);
    }
    if (this.config.mode === "auto" && this.config.auto.continuousListening) {
      setTimeout(() => this.startRecording(), 500);
    }
  }
  // ==================== Audio Input ====================
  async initAudioInput() {
    if (typeof navigator !== "undefined" && navigator.mediaDevices) {
      try {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16e3,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });
        this.audioContext = new AudioContext({ sampleRate: 16e3 });
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        await this.audioContext.audioWorklet.addModule("vad-processor.js");
        this.audioWorklet = new AudioWorkletNode(this.audioContext, "vad-processor");
        source.connect(this.audioWorklet);
        this.audioWorklet.connect(this.audioContext.destination);
        this.audioWorklet.port.onmessage = (event) => {
          this.handleAudioChunk(event.data);
        };
      } catch (err) {
        this.emit("error", new Error(`Microphone access denied: ${err.message}`));
      }
    } else {
      this.emit("error", new Error("Node.js audio input not implemented. Use browser or native recorder."));
    }
  }
  closeAudioInput() {
    if (this.audioWorklet) {
      this.audioWorklet.disconnect();
      this.audioWorklet = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
  }
  captureAudio() {
  }
  handleAudioChunk(audioData) {
    if (!this.recordingState.isRecording) return;
    this.recordingState.audioChunks.push(audioData);
    if (this.sherpaEngine.vad) {
      const vadResult = this.sherpaEngine.vad.process(audioData);
      if (vadResult.hasSpeech) {
        this.recordingState.hasSpeech = true;
        this.recordingState.silenceStart = 0;
      } else if (this.recordingState.hasSpeech) {
        if (this.recordingState.silenceStart === 0) {
          this.recordingState.silenceStart = Date.now();
        }
      }
    }
  }
  // ==================== Wake Word ====================
  async initWakeWord() {
    if (this.config.wakeWord.engine === "disabled") return;
    if (this.config.wakeWord.engine === "openwakeword") {
      try {
        const { OpenWakeWord } = await import("openwakeword");
        this.wakeWordDetector = new OpenWakeWord({
          model: this.config.wakeWord.modelPath || "models/hey_jarvis.onnx",
          threshold: this.config.wakeWord.sensitivity
        });
        this.wakeWordDetector.on("detected", (keyword) => {
          this.emit("wakeWordDetected", { keyword, timestamp: Date.now() });
          if (this.config.mode === "off" || this.config.mode === "manual") {
            this.setMode("auto");
          }
        });
      } catch (err) {
        this.emit("error", new Error(`Wake word init failed: ${err.message}`));
      }
    }
  }
  stopWakeWord() {
    if (this.wakeWordDetector) {
      this.wakeWordDetector.stop();
      this.wakeWordDetector = null;
    }
  }
  // ==================== Configuration ====================
  updateConfig(updates) {
    this.config = { ...this.config, ...updates };
    this.emit("configChange", this.config);
  }
  getConfig() {
    return { ...this.config };
  }
  // ==================== Cleanup ====================
  destroy() {
    this.stopAutoListening();
    this.stopWakeWord();
    this.closeAudioInput();
    this.removeAllListeners();
  }
};
function registerModeManager(api, kv, logger, config = {}) {
  const modeManager = new VoiceModeManager(api, kv, logger, config);
  return modeManager;
}

// lib/llm-client.ts
var DEFAULTS = {
  maxTokens: 2048,
  reasoningEffort: null,
  chatTemplateKwargs: null,
  retries: 2
};
function normalizeRetries(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) return DEFAULTS.retries;
  return Math.floor(parsed);
}
function normalizeChatTemplateKwargs(value) {
  if (!value) return null;
  if (typeof value === "object") return value;
  try {
    const parsed = JSON.parse(value);
    return typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}
function shouldRetry(status) {
  return status === 408 || status === 429 || status >= 500;
}
function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
function createClient(pluginOptions, logger) {
  function getConfig() {
    return {
      endpoint: pluginOptions?.endpoint,
      model: pluginOptions?.model,
      apiKeyEnv: pluginOptions?.apiKeyEnv,
      maxTokens: pluginOptions?.maxTokens ?? DEFAULTS.maxTokens,
      reasoningEffort: pluginOptions?.reasoningEffort ?? DEFAULTS.reasoningEffort,
      chatTemplateKwargs: normalizeChatTemplateKwargs(
        pluginOptions?.chatTemplateKwargs ?? DEFAULTS.chatTemplateKwargs
      ),
      retries: normalizeRetries(pluginOptions?.retries ?? DEFAULTS.retries)
    };
  }
  async function complete({ system, prompt, config: overrides }) {
    const cfg = { ...getConfig(), ...overrides };
    if (!cfg.endpoint) {
      logger?.log?.("LLM", "completion skipped: endpoint not configured", "warn");
      return { text: null, error: "LLM endpoint not configured" };
    }
    if (!cfg.model) {
      logger?.log?.("LLM", "completion skipped: model not configured", "warn");
      return { text: null, error: "LLM model not configured" };
    }
    const apiKey = cfg.apiKeyEnv ? process.env[cfg.apiKeyEnv] : null;
    const endpoint = cfg.endpoint.replace(/\/+$/, "") + "/chat/completions";
    const messages = [];
    if (system) messages.push({ role: "system", content: system });
    messages.push({ role: "user", content: prompt });
    const body = {
      model: cfg.model,
      max_tokens: cfg.maxTokens,
      messages
    };
    if (cfg.reasoningEffort) body.reasoning_effort = cfg.reasoningEffort;
    if (cfg.chatTemplateKwargs) body.chat_template_kwargs = cfg.chatTemplateKwargs;
    for (let attempt = 0; attempt <= cfg.retries; attempt++) {
      try {
        logger?.log?.(
          "LLM",
          `Completion request attempt=${attempt + 1} model=${cfg.model} maxTokens=${cfg.maxTokens} promptChars=${prompt.length}`,
          "debug"
        );
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...apiKey ? { Authorization: "Bearer " + apiKey } : {}
          },
          body: JSON.stringify(body)
        });
        if (!response.ok) {
          logger?.log?.(
            "LLM",
            `Completion response status=${response.status}`,
            shouldRetry(response.status) ? "warn" : "error"
          );
          if (attempt < cfg.retries && shouldRetry(response.status)) {
            await wait(250 * 2 ** attempt);
            continue;
          }
          return { text: null, error: `LLM request failed (${response.status})` };
        }
        const data = await response.json();
        const text = data?.choices?.[0]?.message?.content || null;
        if (text) {
          logger?.log?.("LLM", `Completion succeeded chars=${text.length}`, "debug");
          return { text };
        }
        logger?.log?.("LLM", "Completion returned empty content", "warn");
        if (attempt < cfg.retries) {
          await wait(250 * 2 ** attempt);
          continue;
        }
        return { text: null, error: "Empty LLM response" };
      } catch (err) {
        logger?.log?.("LLM", `Completion error attempt=${attempt + 1}: ${err.message}`, "warn");
        if (attempt < cfg.retries) {
          await wait(250 * 2 ** attempt);
          continue;
        }
        return { text: null, error: `LLM error: ${err.message}` };
      }
    }
    return { text: null, error: "LLM request failed after retries" };
  }
  return { complete };
}

// lib/logger.ts
function createLogger(client) {
  async function log(scope, message, level = "debug") {
    try {
      await client?.app?.log?.({
        body: {
          service: "opencode-voice",
          level,
          message,
          extra: { scope }
        }
      });
    } catch {
    }
  }
  return { log };
}

// index.ts
function loadPromptFile(filePath, logger, name) {
  if (!filePath) return null;
  const resolved = filePath.replace(/^~(?=\/|$)/, os4.homedir());
  try {
    const prompt = fs5.readFileSync(resolved, "utf-8").trim() || null;
    logger?.log(
      "plugin",
      prompt ? `Loaded ${name} prompt: ${resolved}` : `Ignored empty ${name} prompt: ${resolved}`,
      "debug"
    );
    return prompt;
  } catch (err) {
    logger?.log("Plugin", `Failed to load ${name} prompt ${resolved}: ${err.message}`, "warn");
    return null;
  }
}
var index_default = {
  id: "opencode-voice-universal",
  tui: async (api, options) => {
    const { kv } = api;
    const logger = createLogger(api.client);
    logger.log("plugin", "Initializing opencode-voice-universal", "debug");
    const { complete } = createClient(options, logger);
    const prompts = {
      stt: loadPromptFile(options?.sttPrompt, logger, "STT"),
      ttsAuto: loadPromptFile(options?.ttsAutoPrompt, logger, "TTS auto"),
      ttsManual: loadPromptFile(options?.ttsManualPrompt, logger, "TTS manual")
    };
    const modeManager = registerModeManager(api, kv, logger, {
      mode: kv.get("voice.mode", "manual"),
      wakeWord: {
        engine: kv.get("voice.wakeWord.engine", "disabled"),
        keywords: kv.get("voice.wakeWord.keywords", ["hey jarvis", "jarvis", "\u0628\u06CC\u062F\u0627\u0631 \u0634\u0648"]),
        sensitivity: kv.get("voice.wakeWord.sensitivity", 0.6),
        customKeywords: kv.get("voice.wakeWord.customKeywords", {})
      },
      auto: {
        vadSensitivity: kv.get("voice.auto.vadSensitivity", 0.5),
        minSpeechDuration: kv.get("voice.auto.minSpeechDuration", 500),
        maxSilenceDuration: kv.get("voice.auto.maxSilenceDuration", 2e3),
        autoSubmit: kv.get("voice.auto.autoSubmit", true),
        continuousListening: kv.get("voice.auto.continuousListening", true)
      },
      manual: {
        pushToTalkKey: kv.get("voice.manual.pushToTalkKey", "ctrl+r"),
        holdToTalk: kv.get("voice.manual.holdToTalk", false),
        doubleTapTimeout: kv.get("voice.manual.doubleTapTimeout", 300)
      }
    });
    const sttCommands = registerSTT(api, kv, complete, prompts, options, logger);
    const ttsCommands = registerTTS(api, kv, complete, prompts, logger);
    const modeCommands = [
      {
        title: "Voice: Toggle AUTO Mode",
        value: "voice.auto",
        description: "Enable always-listening mode with VAD (voice activity detection)",
        keybind: "ctrl+shift+a",
        slash: { name: "auto-voice-mode" },
        onSelect() {
          const currentMode = modeManager.getMode();
          const newMode = currentMode === "auto" ? "off" : "auto";
          modeManager.setMode(newMode);
          kv.set("voice.mode", newMode);
          api.ui.toast({
            message: `Voice mode: ${newMode.toUpperCase()}`,
            variant: newMode === "off" ? "warning" : "success",
            duration: 3e3
          });
        }
      },
      {
        title: "Voice: Toggle MANUAL Mode",
        value: "voice.manual",
        description: "Enable push-to-talk mode (press key to record)",
        keybind: "ctrl+shift+m",
        slash: { name: "manual-voice-mode" },
        onSelect() {
          const currentMode = modeManager.getMode();
          const newMode = currentMode === "manual" ? "off" : "manual";
          modeManager.setMode(newMode);
          kv.set("voice.mode", newMode);
          api.ui.toast({
            message: `Voice mode: ${newMode.toUpperCase()}`,
            variant: newMode === "off" ? "warning" : "success",
            duration: 3e3
          });
        }
      },
      {
        title: "Voice: Cycle Mode",
        value: "voice.cycle",
        description: "Cycle through OFF \u2192 MANUAL \u2192 AUTO \u2192 OFF",
        keybind: "ctrl+shift+v",
        slash: { name: "voice-mode" },
        onSelect() {
          modeManager.toggleMode();
          const newMode = modeManager.getMode();
          kv.set("voice.mode", newMode);
          api.ui.toost({
            message: `Voice mode: ${newMode.toUpperCase()}`,
            variant: newMode === "off" ? "warning" : "success",
            duration: 3e3
          });
        }
      },
      {
        title: "Voice: Wake Word Settings",
        value: "voice.wake-word",
        description: "Configure wake word detection",
        slash: { name: "voice-wake-word" },
        async onSelect() {
          const config = modeManager.getConfig();
          const engine = await api.ui.dialog.select({
            title: "Wake Word Engine",
            current: config.wakeWord.engine,
            options: [
              { title: "Disabled", value: "disabled", description: "No wake word detection" },
              { title: "openwakeword (ONNX)", value: "openwakeword", description: "Local ONNX model, private" },
              { title: "Porcupine", value: "porcupine", description: "Picovoice engine (requires license)" }
            ].map((o) => ({ ...o, onSelect: () => {
              config.wakeWord.engine = o.value;
              modeManager.updateConfig({ wakeWord: config.wakeWord });
              kv.set("voice.wakeWord.engine", o.value);
            } }))
          });
          if (!engine) return;
          config.wakeWord.engine = engine;
          if (engine !== "disabled") {
            const keywords = await api.ui.dialog.input({
              title: "Wake Keywords (comma-separated)",
              placeholder: "hey jarvis, jarvis, \u0628\u06CC\u062F\u0627\u0631 \u0634\u0648",
              value: config.wakeWord.keywords.join(", ")
            });
            if (keywords) {
              config.wakeWord.keywords = keywords.split(",").map((k) => k.trim());
              modeManager.updateConfig({ wakeWord: config.wakeWord });
              kv.set("voice.wakeWord.keywords", config.wakeWord.keywords);
            }
            const sensitivity = await api.ui.dialog.input({
              title: "Sensitivity (0.0 - 1.0)",
              placeholder: "0.6",
              value: config.wakeWord.sensitivity.toString()
            });
            if (sensitivity) {
              config.wakeWord.sensitivity = parseFloat(sensitivity);
              modeManager.updateConfig({ wakeWord: config.wakeWord });
              kv.set("voice.wakeWord.sensitivity", config.wakeWord.sensitivity);
            }
          }
          api.ui.toast({ message: "Wake word settings updated", variant: "success" });
        }
      },
      {
        title: "Voice: AUTO Mode Settings",
        value: "voice.auto-settings",
        description: "Configure VAD sensitivity and auto-submit behavior",
        slash: { name: "voice-auto-settings" },
        async onSelect() {
          const config = modeManager.getConfig();
          const vadSensitivity = await api.ui.dialog.input({
            title: "VAD Sensitivity (0.0 - 1.0)",
            placeholder: "0.5",
            value: config.auto.vadSensitivity.toString()
          });
          if (vadSensitivity) {
            config.auto.vadSensitivity = parseFloat(vadSensitivity);
            kv.set("voice.auto.vadSensitivity", config.auto.vadSensitivity);
          }
          const minSpeech = await api.ui.dialog.input({
            title: "Min Speech Duration (ms)",
            placeholder: "500",
            value: config.auto.minSpeechDuration.toString()
          });
          if (minSpeech) {
            config.auto.minSpeechDuration = parseInt(minSpeech);
            kv.set("voice.auto.minSpeechDuration", config.auto.minSpeechDuration);
          }
          const maxSilence = await api.ui.dialog.input({
            title: "Max Silence Before Submit (ms)",
            placeholder: "2000",
            value: config.auto.maxSilenceDuration.toString()
          });
          if (maxSilence) {
            config.auto.maxSilenceDuration = parseInt(maxSilence);
            kv.set("voice.auto.maxSilenceDuration", config.auto.maxSilenceDuration);
          }
          const autoSubmit = await api.ui.dialog.confirm({
            title: "Auto-submit after transcription?",
            message: "Automatically submit the prompt after transcription in AUTO mode"
          });
          config.auto.autoSubmit = autoSubmit;
          kv.set("voice.auto.autoSubmit", autoSubmit);
          const continuous = await api.ui.dialog.confirm({
            title: "Continuous listening?",
            message: "Keep listening after each transcription in AUTO mode"
          });
          config.auto.continuousListening = continuous;
          kv.set("voice.auto.continuousListening", continuous);
          modeManager.updateConfig({ auto: config.auto });
          api.ui.toast({ message: "AUTO mode settings updated", variant: "success" });
        }
      },
      {
        title: "Voice: MANUAL Mode Settings",
        value: "voice.manual-settings",
        description: "Configure push-to-talk key and behavior",
        slash: { name: "voice-manual-settings" },
        async onSelect() {
          const config = modeManager.getConfig();
          const key = await api.ui.dialog.input({
            title: "Push-to-Talk Key",
            placeholder: "ctrl+r",
            value: config.manual.pushToTalkKey
          });
          if (key) {
            config.manual.pushToTalkKey = key;
            kv.set("voice.manual.pushToTalkKey", key);
          }
          const holdToTalk = await api.ui.dialog.confirm({
            title: "Hold-to-talk mode?",
            message: "Hold key to record, release to stop (vs toggle)"
          });
          config.manual.holdToTalk = holdToTalk;
          kv.set("voice.manual.holdToTalk", holdToTalk);
          modeManager.updateConfig({ manual: config.manual });
          api.ui.toast({ message: "MANUAL mode settings updated", variant: "success" });
        }
      },
      {
        title: "Voice: Status",
        value: "voice.status",
        description: "Show current voice mode status",
        slash: { name: "voice-status" },
        onSelect() {
          const config = modeManager.getConfig();
          const mode = modeManager.getMode();
          const voices = kv.get("tts.voice") || "default";
          const sttModel = kv.get("stt.model") || "shenava-koochik-int8";
          const status = [
            `\u{1F3A4} Voice Mode: ${mode.toUpperCase()}`,
            `\u{1F5E3}\uFE0F TTS Voice: ${voices}`,
            `\u{1F3A7} STT Model: ${sttModel}`,
            `\u{1F50A} Wake Word: ${config.wakeWord.engine === "disabled" ? "OFF" : config.wakeWord.engine} (${config.wakeWord.keywords.join(", ")})`,
            `\u{1F3AF} AUTO: VAD=${config.auto.vadSensitivity}, Submit=${config.auto.autoSubmit}`,
            `\u2328\uFE0F MANUAL: Key=${config.manual.pushToTalkKey}, Hold=${config.manual.holdToTalk}`
          ].join("\n");
          api.ui.dialog.info({
            title: "Voice Status",
            message: status
          });
        }
      },
      {
        title: "Voice: Mixed Language Config",
        value: "voice.mixed-lang",
        description: "Configure mixed-language (Persian + English) synthesis",
        slash: { name: "voice-mixed-lang" },
        async onSelect() {
          const primary = await api.ui.dialog.select({
            title: "Primary Language",
            current: "en-US",
            options: [
              { title: "English (US)", value: "en-US" },
              { title: "Persian (Farsi)", value: "fa-IR" },
              { title: "English (UK)", value: "en-GB" }
            ].map((o) => ({ ...o, onSelect: () => {
            } }))
          });
          const secondary = await api.ui.dialog.input({
            title: "Secondary Languages (comma-separated)",
            placeholder: "fa-IR,en-US",
            value: "fa-IR"
          });
          const enInFa = await api.ui.dialog.select({
            title: "English words in Persian text",
            current: "pronounce",
            options: [
              { title: "Pronounce with Persian accent", value: "pronounce", description: "API -> \u0627\u06D2 \u067E\u06CC \u0622\u0626\u06CC" },
              { title: "Spell out", value: "spell", description: "API -> \u0627\u06D2 \u067E\u06CC \u0622\u06CC" },
              { title: "Keep as-is", value: "keep" }
            ].map((o) => ({ ...o, onSelect: () => {
            } }))
          });
          api.ui.toast({ message: "Mixed language settings saved", variant: "success" });
        }
      }
    ];
    const sttCommandsResult = registerSTT(api, kv, complete, prompts, options, logger);
    const ttsCommandsResult = registerTTS(api, kv, complete, prompts, logger);
    const modeCommandsWithHandler = modeCommands.map((cmd) => ({
      ...cmd,
      onSelect: cmd.onSelect
    }));
    api.command.register(() => [...sttCommandsResult, ...ttsCommandsResult, ...modeCommandsWithHandler]);
  }
};
export {
  index_default as default
};
//# sourceMappingURL=voice-universal.js.map
