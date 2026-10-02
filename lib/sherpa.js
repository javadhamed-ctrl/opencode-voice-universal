/**
 * Sherpa-onnx Universal Wrapper
 * Single runtime for STT (Shenava Koochik, Whisper, NeMo) + TTS (Piper, VITS, Matcha) + VAD (Silero)
 */

import { OfflineRecognizer, OfflineTts, Vad } from "sherpa-onnx";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";

export interface STTModelConfig {
  id: string;
  name: string;
  type: "transducer" | "nemo_ctc" | "whisper" | "zipformer";
  encoder: string;
  decoder?: string;
  joiner?: string;
  tokens: string;
  numThreads?: number;
  sampleRate?: number;
  featureDim?: number;
  decodingMethod?: "greedy_search" | "modified_beam_search";
  modelType?: string; // "nemo_transducer" for Shenava
  languages: string[];
  description: string;
}

export interface TTSModelConfig {
  id: string;
  name: string;
  type: "piper" | "vits" | "matcha" | "kokoro";
  model: string;
  tokens: string;
  dataDir?: string; // espeak-ng-data for Piper
  numThreads?: number;
  voices: VoiceInfo[];
  languages: string[];
  description: string;
}

export interface VoiceInfo {
  id: string;
  name: string;
  language: string;
  gender: "male" | "female" | "neutral";
  ageGroup: "child" | "young" | "adult" | "senior" | "unknown";
  style: string[];
  sampleRate: number;
  description?: string;
  // For multi-speaker models:
  speakerId?: number;      // VITS/XTTS
  styleVector?: number[];  // Kokoro
}

export interface VADConfig {
  id: string;
  name: string;
  model: string;
  threshold?: number;
  minSpeechDuration?: number;
  minSilenceDuration?: number;
  windowSize?: number;
}

export interface STTResult {
  text: string;
  language?: string;
  confidence?: number;
  segments?: STTSegment[];
}

export interface STTSegment {
  start: number;
  end: number;
  text: string;
  language?: string;
}

export interface TTSOptions {
  voice: string;           // voice id
  speed?: number;          // 0.5 - 2.0
  pitch?: number;          // 0.5 - 2.0 (if supported)
}

export interface AudioBuffer {
  data: Float32Array;
  sampleRate: number;
  channels: number;
}

class SherpaEngine {
  private sttRecognizer: OfflineRecognizer | null = null;
  private ttsEngine: OfflineTts | null = null;
  private vad: Vad | null = null;
  private currentSTTModel: string | null = null;
  private currentTTSModel: string | null = null;
  private modelsDir: string;

  // Built-in model registry
  private readonly STT_MODELS: Record<string, STTModelConfig> = {
    "shenava-koochik-int8": {
      id: "shenava-koochik-int8",
      name: "Shenava Koochik (Persian, INT8)",
      type: "transducer",
      encoder: "encoder.int8.onnx",
      decoder: "decoder.int8.onnx",
      joiner: "joiner.int8.onnx",
      tokens: "tokens.txt",
      numThreads: 2,
      sampleRate: 16000,
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
      sampleRate: 16000,
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
      sampleRate: 16000,
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
      sampleRate: 16000,
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
      sampleRate: 16000,
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
      sampleRate: 16000,
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
      sampleRate: 16000,
      languages: ["en-US"],
      description: "Fast English-only STT"
    }
  };

  private readonly TTS_MODELS: Record<string, TTSModelConfig> = {
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
        { id: "af_heart", name: "Heart (Female US)", language: "en-US", gender: "female", ageGroup: "young", style: ["premium", "warm"], sampleRate: 24000 },
        { id: "af_bella", name: "Bella (Female US)", language: "en-US", gender: "female", ageGroup: "young", style: ["warm", "friendly"], sampleRate: 24000 },
        { id: "af_nicole", name: "Nicole (Female US)", language: "en-US", gender: "female", ageGroup: "adult", style: ["professional"], sampleRate: 24000 },
        { id: "am_adam", name: "Adam (Male US)", language: "en-US", gender: "male", ageGroup: "adult", style: ["professional"], sampleRate: 24000 },
        { id: "am_michael", name: "Michael (Male US)", language: "en-US", gender: "male", ageGroup: "adult", style: ["deep", "calm"], sampleRate: 24000 },
        { id: "bf_emma", name: "Emma (Female UK)", language: "en-GB", gender: "female", ageGroup: "young", style: ["british"], sampleRate: 24000 },
        { id: "bm_george", name: "George (Male UK)", language: "en-GB", gender: "male", ageGroup: "adult", style: ["british"], sampleRate: 24000 }
      ],
      languages: ["en-US", "en-GB", "ja", "zh", "es", "fr", "hi", "it", "pt"],
      description: "82M param multi-lingual TTS, 54 voices"
    }
  };

  private readonly VAD_MODELS: Record<string, VADConfig> = {
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

  constructor(modelsDir?: string) {
    this.modelsDir = modelsDir || path.join(os.homedir(), ".local", "share", "opencode-voice", "models");
    this.ensureModelsDir();
  }

  private ensureModelsDir() {
    if (!fs.existsSync(this.modelsDir)) {
      fs.mkdirSync(this.modelsDir, { recursive: true });
    }
  }

  private getModelPath(modelFile: string): string {
    const fullPath = path.join(this.modelsDir, modelFile);
    if (!fs.existsSync(fullPath)) {
      throw new Error(`Model file not found: ${fullPath}. Run download:models or place manually in ${this.modelsDir}`);
    }
    return fullPath;
  }

  // ==================== STT ====================

  async loadSTT(modelId: string, options: Partial<STTModelConfig> = {}): Promise<void> {
    if (this.currentSTTModel === modelId && this.sttRecognizer) return;

    const config = { ...this.STT_MODELS[modelId], ...options };
    if (!config) throw new Error(`Unknown STT model: ${modelId}`);

    const modelPath = this.getModelPath(config.encoder);
    const tokensPath = this.getModelPath(config.tokens);

    let recognizer: OfflineRecognizer;

    switch (config.type) {
      case "transducer":
        if (!config.decoder || !config.joiner) throw new Error("Transducer needs decoder and joiner");
        recognizer = OfflineRecognizer.fromTransducer({
          encoder: this.getModelPath(config.decoder), // Note: sherpa-onnx expects decoder path here for transducer
          decoder: this.getModelPath(config.joiner),
          joiner: this.getModelPath(config.encoder),  // encoder is actually the main model
          tokens: tokensPath,
          numThreads: config.numThreads || 2,
          sampleRate: config.sampleRate || 16000,
          featureDim: config.featureDim || 80,
          decodingMethod: config.decodingMethod || "greedy_search",
          modelType: config.modelType || "nemo_transducer",
        });
        break;

      case "nemo_ctc":
        recognizer = OfflineRecognizer.fromNemoCTC({
          model: modelPath,
          tokens: tokensPath,
          numThreads: config.numThreads || 2,
          sampleRate: config.sampleRate || 16000,
        });
        break;

      case "whisper":
        if (!config.decoder) throw new Error("Whisper needs decoder");
        recognizer = OfflineRecognizer.fromWhisper({
          encoder: modelPath,
          decoder: this.getModelPath(config.decoder),
          tokens: tokensPath,
          numThreads: config.numThreads || 2,
          sampleRate: config.sampleRate || 16000,
          language: "auto",
          task: "transcribe",
          tailPadding: -1,
        });
        break;

      default:
        throw new Error(`Unsupported STT type: ${config.type}`);
    }

    this.sttRecognizer = recognizer;
    this.currentSTTModel = modelId;
  }

  async transcribe(audio: Float32Array, sampleRate: number = 16000, language?: string): Promise<STTResult> {
    if (!this.sttRecognizer) throw new Error("STT model not loaded. Call loadSTT() first.");

    // Resample if needed (sherpa-onnx expects 16kHz)
    let processedAudio = audio;
    if (sampleRate !== 16000) {
      processedAudio = this.resample(audio, sampleRate, 16000);
    }

    const stream = this.sttRecognizer.createStream();
    stream.acceptWaveform(16000, processedAudio);
    this.sttRecognizer.decodeStream(stream);

    const result = stream.result;
    return {
      text: result.text.trim(),
      language: language || "auto",
      confidence: 1.0, // sherpa-onnx doesn't expose confidence directly
    };
  }

  async transcribeStreaming(audioChunks: AsyncIterable<Float32Array>, sampleRate: number = 16000): AsyncGenerator<STTResult> {
    if (!this.sttRecognizer) throw new Error("STT model not loaded");

    const stream = this.sttRecognizer.createStream();
    for await (const chunk of audioChunks) {
      stream.acceptWaveform(16000, chunk);
      this.sttRecognizer.decodeStream(stream);
      const result = stream.result;
      if (result.text.trim()) {
        yield { text: result.text.trim(), language: "auto" };
      }
    }
  }

  // ==================== TTS ====================

  async loadTTS(modelId: string, options: Partial<TTSModelConfig> = {}): Promise<void> {
    if (this.currentTTSModel === modelId && this.ttsEngine) return;

    const config = { ...this.TTS_MODELS[modelId], ...options };
    if (!config) throw new Error(`Unknown TTS model: ${modelId}`);

    const modelPath = this.getModelPath(config.model);
    const tokensPath = this.getModelPath(config.tokens);

    let tts: OfflineTts;

    switch (config.type) {
      case "piper":
      case "vits":
        if (!config.dataDir) throw new Error("Piper/VITS needs dataDir (espeak-ng-data)");
        tts = new OfflineTts({
          model: modelPath,
          tokens: tokensPath,
          dataDir: this.getModelPath(config.dataDir),
          numThreads: config.numThreads || 2,
        });
        break;

      case "matcha":
        tts = new OfflineTts({
          model: modelPath,
          tokens: tokensPath,
          numThreads: config.numThreads || 2,
        });
        break;

      default:
        throw new Error(`Unsupported TTS type: ${config.type}`);
    }

    this.ttsEngine = tts;
    this.currentTTSModel = modelId;
  }

  async synthesize(text: string, options: TTSOptions): Promise<AudioBuffer> {
    if (!this.ttsEngine) throw new Error("TTS model not loaded. Call loadTTS() first.");

    // For multi-speaker models, voice maps to speaker_id or style
    const audio = this.ttsEngine.generate(text, options.voice, {
      speed: options.speed || 1.0,
    });

    return {
      data: audio.samples,
      sampleRate: audio.sampleRate,
      channels: 1,
    };
  }

  async *synthesizeStreaming(text: string, options: TTSOptions): AsyncGenerator<AudioBuffer> {
    if (!this.ttsEngine) throw new Error("TTS model not loaded");

    // Split text into sentences for streaming
    const sentences = text.split(/(?<=[.!?])\s+/);
    for (const sentence of sentences) {
      if (sentence.trim()) {
        yield this.synthesize(sentence.trim(), options);
      }
    }
  }

  getAvailableVoices(modelId?: string): VoiceInfo[] {
    if (modelId) {
      return this.TTS_MODELS[modelId]?.voices || [];
    }
    // All voices from all models
    return Object.values(this.TTS_MODELS).flatMap(m => m.voices);
  }

  // ==================== VAD ====================

  async loadVAD(modelId: string = "silero-vad"): Promise<void> {
    if (this.vad) return;

    const config = this.VAD_MODELS[modelId];
    if (!config) throw new Error(`Unknown VAD model: ${modelId}`);

    this.vad = new Vad({
      model: this.getModelPath(config.model),
      threshold: config.threshold || 0.5,
      minSpeechDuration: config.minSpeechDuration || 250,
      minSilenceDuration: config.minSilenceDuration || 100,
      windowSize: config.windowSize || 512,
    });
  }

  async detectVoiceActivity(audio: Float32Array): Promise<{ hasSpeech: boolean; probability: number }> {
    if (!this.vad) await this.loadVAD();
    const result = this.vad!.process(audio);
    return {
      hasSpeech: result.hasSpeech,
      probability: result.probability,
    };
  }

  // ==================== Utility ====================

  private resample(audio: Float32Array, fromRate: number, toRate: number): Float32Array {
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

  listSTTModels(): STTModelConfig[] {
    return Object.values(this.STT_MODELS);
  }

  listTTSModels(): TTSModelConfig[] {
    return Object.values(this.TTS_MODELS);
  }

  listVADModels(): VADConfig[] {
    return Object.values(this.VAD_MODELS);
  }

  getCurrentSTTModel(): string | null {
    return this.currentSTTModel;
  }

  getCurrentTTSModel(): string | null {
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
}

// Singleton instance
let sherpaInstance: SherpaEngine | null = null;

export function getSherpaEngine(modelsDir?: string): SherpaEngine {
  if (!sherpaInstance) {
    sherpaInstance = new SherpaEngine(modelsDir);
  }
  return sherpaInstance;
}

export function resetSherpaEngine() {
  if (sherpaInstance) {
    sherpaInstance.unloadAll();
    sherpaInstance = null;
  }
}

export { SherpaEngine };