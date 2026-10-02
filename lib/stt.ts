/**
 * Speech-to-Text with sherpa-onnx (Shenava Koochik, Whisper, NeMo)
 * Supports local transcription with multiple models and API fallback
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { spawn, execSync } from "node:child_process";
import { getSherpaEngine, SherpaEngine, STTResult } from "./sherpa.js";
import { getActiveSessionTitle } from "./session.js";

let sttApiEndpoint = null;
let sttApiModel = null;
let sttApiKeyEnv = null;

const WAV_FILENAME = "opencode-stt.wav";
let tmpDir = "/tmp";

// sherpa-onnx STT Models (local)
const SHERPA_STT_MODELS = {
  "shenava-koochik-int8": {
    label: "Shenava Koochik (Persian, INT8) ★ Best",
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

const DEFAULT_STT_MODEL = "shenava-koochik-int8";
const DEFAULT_LANGUAGE = "auto";

// Language codes for sherpa-onnx
const LANGUAGES = {
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

let sttDefaultLanguage = DEFAULT_LANGUAGE;

// Sherpa engine instance
let sherpaEngine: SherpaEngine | null = null;
let currentSTTModel: string = DEFAULT_STT_MODEL;

// Recording state (for browser AudioWorklet)
let recording = false;
let processing = false;
let audioChunks: Float32Array[] = [];
let recordingStartTime = 0;

// Audio backend detection (for fallback)
function detectAudioBackend() {
  if (process.platform === "darwin") return "coreaudio";
  try {
    execSync("pactl --version", { stdio: "ignore", timeout: 3000 });
    return "pulseaudio";
  } catch {
    return "default";
  }
}

// Get or create sherpa engine
function getSherpa(): SherpaEngine {
  if (!sherpaEngine) {
    sherpaEngine = getSherpaEngine();
  }
  return sherpaEngine;
}

// Load STT model
async function loadSTTModel(modelId: string = currentSTTModel) {
  const engine = getSherpa();
  await engine.loadSTT(modelId);
  currentSTTModel = modelId;
}

// Transcribe audio using sherpa-onnx
async function transcribeLocal(audioData: Float32Array, sampleRate: number, language?: string): Promise<STTResult> {
  const engine = getSherpa();
  await loadSTTModel(currentSTTModel);
  return engine.transcribe(audioData, sampleRate, language);
}

// Transcribe with API fallback
async function transcribeApi(audioData: Float32Array, kv: any, logger: any) {
  if (!sttApiEndpoint || !sttApiModel) {
    logger?.log("STT", "STT API transcription skipped: API not configured", "warn");
    return { error: "STT API not configured" };
  }

  const model = kv.get("stt.api.model") || sttApiModel;
  logger?.log("STT", `STT API transcription requested model=${model}`, "debug");

  // Convert Float32Array to WAV buffer
  const wavBuffer = float32ToWav(audioData, 16000);

  try {
    const apiKey = sttApiKeyEnv ? process.env[sttApiKeyEnv] : null;
    const useOpenRouterFormat = isOpenRouterEndpoint(sttApiEndpoint);

    const url = sttApiEndpoint.endsWith("/")
      ? `${sttApiEndpoint}audio/transcriptions`
      : `${sttApiEndpoint}/audio/transcriptions`;

    const request = useOpenRouterFormat
      ? buildOpenRouterTranscriptionRequest(model, wavBuffer, apiKey)
      : buildMultipartTranscriptionRequest(model, wavBuffer, apiKey);

    const resp = await fetch(url, {
      method: "POST",
      headers: request.headers,
      body: request.body,
      signal: AbortSignal.timeout(60000),
    });
    logger?.log("STT", `STT API response status=${resp.status}`, resp.ok ? "debug" : "error");

    if (!resp.ok) {
      const responseBody = await resp.text();
      let msg = `STT API error ${resp.status}`;
      try {
        const err = JSON.parse(responseBody);
        msg = err?.error?.message || msg;
      } catch {}
      return { error: msg };
    }

    let data;
    try {
      data = await resp.json();
    } catch (err) {
      logger?.log("STT", `STT API returned invalid JSON: ${err.message}`, "error");
      return { error: `STT API returned invalid JSON: ${err.message}` };
    }
    logger?.log("STT", `STT API transcription succeeded chars=${data.text?.length || 0}`, "debug");
    return { text: data.text?.trim() || "" };
  } catch (err) {
    logger?.log("STT", `STT API request failed: ${err.message}`, "error");
    if (err.name === "TimeoutError" || err.name === "AbortError") {
      return { error: "STT API request timed out (60s)" };
    }
    return { error: `STT API request failed: ${err.message}` };
  }
}

// Convert Float32Array to WAV buffer
function float32ToWav(audioData: Float32Array, sampleRate: number): Buffer {
  const int16Data = new Int16Array(audioData.length);
  for (let i = 0; i < audioData.length; i++) {
    const s = Math.max(-1, Math.min(1, audioData[i]));
    int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }

  const buffer = Buffer.alloc(44 + int16Data.length * 2);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  // RIFF header
  view.setUint32(0, 0x52494646, false); // "RIFF"
  view.setUint32(4, 36 + int16Data.length * 2, true); // file size - 8
  view.setUint32(8, 0x57415645, false); // "WAVE"

  // fmt chunk
  view.setUint32(12, 0x666D7420, false); // "fmt "
  view.setUint32(16, 16, true); // chunk size
  view.setUint16(20, 1, true); // audio format (PCM)
  view.setUint16(22, 1, true); // channels
  view.setUint32(24, 16000, true); // sample rate
  view.setUint32(28, 16000 * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample

  // data chunk
  view.setUint32(36, 0x64617461, false); // "data"
  view.setUint32(40, int16Data.length * 2, true); // data size

  // Write audio data
  for (let i = 0; i < int16Data.length; i++) {
    view.setInt16(44 + i * 2, int16Data[i], true);
  }

  return buffer;
}

// STT System Prompt (same as original)
const STT_SYSTEM_PROMPT = `You are a speech-to-text normalizer for a coding assistant CLI.

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

async function normalizeTranscription(complete, rawText, sessionTitle, systemPrompt, logger) {
  const contextLine = sessionTitle ? ` The user is currently working on: "${sessionTitle}"` : "";
  const system = `${systemPrompt}${contextLine}`;

  logger?.log("STT", `Normalizing transcription chars=${rawText.length}`, "debug");
  const result = await complete({
    system,
    prompt: `Clean up this speech-to-text transcription:\n\n${rawText}`,
  });
  return result;
}

async function appendTranscription(client, text, submit) {
  let appendResult = await client.tui.appendPrompt({ body: { text } });

  if (appendResult?.error?.data?.message === "Expected object, got undefined") {
    appendResult = await client.tui.appendPrompt({ text });
  }

  if (appendResult?.error) {
    throw new Error(
      `appendPrompt failed: ${appendResult.error.data?.message || appendResult.error.name}`,
    );
  }

  if (submit) {
    await client.tui.submitPrompt();
  }
}

async function doTranscribePipeline(
  kv,
  complete,
  client,
  toast,
  systemPrompt,
  submit = false,
  logger,
) {
  processing = true;
  try {
    logger?.log("STT", `Pipeline started submit=${submit}`, "debug");

    // Get recorded audio from the mode manager
    const audioData = getRecordedAudio();
    if (!audioData || audioData.length === 0) {
      toast("No audio recorded", "warning");
      return;
    }

    toast("Transcribing...");

    let result;
    if (sttApiEndpoint) {
      result = await transcribeApi(audioData, kv, logger);
    } else {
      result = await transcribeLocal(audioData, 16000, kv.get("stt.language") || sttDefaultLanguage);
    }

    if (result.error) {
      logger?.log("STT", `Transcription failed: ${result.error}`, "error");
      toast(result.error, "error");
      return;
    }
    if (!result.text) {
      logger?.log("STT", "Transcription produced no text", "warn");
      toast("No speech detected", "warning");
      return;
    }

    toast("Normalizing...");
    const sessionTitle = await getActiveSessionTitle(client);
    const llmResult = await normalizeTranscription(
      complete,
      result.text,
      sessionTitle,
      systemPrompt,
      logger,
    );

    if (!llmResult.text) {
      logger?.log("STT", `Normalization failed, using raw input: ${llmResult.error}`, "warn");
      toast(`Normalization failed, using raw input: ${llmResult.error}`, "warning");
      await appendTranscription(client, result.text, submit);
      return;
    }

    await appendTranscription(client, llmResult.text, submit);
    logger?.log("STT", `Pipeline completed normalizedChars=${llmResult.text.length}`, "debug");
    toast(submit ? "Transcription submitted" : "Transcription added to prompt", "success");
  } catch (err) {
    logger?.log("STT", `Pipeline error: ${err.message}`, "error");
    toast(`STT error: ${err.message}`, "error");
  } finally {
    processing = false;
    recording = false;
    audioChunks = [];
  }
}

// Get recorded audio from mode manager (will be set by mode manager events)
function getRecordedAudio(): Float32Array | null {
  // This will be populated by the mode manager's recording events
  // For now, return null - the actual audio comes from mode manager
  return null;
}

function setRecordedAudio(data: Float32Array) {
  // Called by mode manager when recording stops
}

// API helper functions (same as original)
export function isOpenRouterEndpoint(endpoint) {
  return /(^https?:\/\/)?([^/]+\.)?openrouter\.ai(\/|$)/i.test(endpoint || "");
}

function buildMultipartTranscriptionRequest(model, audioBuffer, apiKey) {
  const blob = new Blob([audioBuffer], { type: "audio/wav" });
  const form = new FormData();
  form.append("file", blob, "audio.wav");
  form.append("model", model);
  form.append("response_format", "json");

  const headers = {};
  if (apiKey) headers["Authorization"] = "Bearer " + apiKey;

  return { headers, body: form };
}

export function buildOpenRouterTranscriptionRequest(model, audioBuffer, apiKey) {
  const headers = { "Content-Type": "application/json" };
  if (apiKey) headers["Authorization"] = "Bearer " + apiKey;

  const payload = {
    model,
    input_audio: {
      data: audioBuffer.toString("base64"),
      format: "wav",
    },
  };

  return { headers, body: JSON.stringify(payload) };
}

// Public API
export function registerSTT(api, kv, complete, prompts, opts, logger) {
  const client = api.client;
  const systemPrompt = prompts?.stt || STT_SYSTEM_PROMPT;

  function toast(message, variant = "info") {
    api.ui.toast({ message, variant, duration: 3000 });
  }

  // Configure API if provided
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
    fs.mkdirSync(tmpDir, { recursive: true });
  } catch (err) {
    logger?.log("STT", `Failed to create tmpDir ${tmpDir}: ${err.message}`, "warn");
  }

  // Pre-load default model
  loadSTTModel(currentSTTModel).catch(err => {
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
          // Mode manager will handle the actual recording stop
          api.event.emit("stt:stopRecording", { submit: false });
        } else {
          api.event.emit("stt:startRecording");
          toast("Recording... press again to transcribe");
        }
      },
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
      },
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
      },
    },
    {
      title: "STT: Select Model",
      value: "stt.model",
      description: "Choose STT model (Shenava, Whisper, NeMo)",
      slash: { name: "stt-model" },
      async onSelect() {
        const current = currentSTTModel;
        api.ui.dialog.replace(() =>
          api.ui.DialogSelect({
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
                }).catch(err => {
                  toast(`Failed to load model: ${err.message}`, "error");
                });
                api.ui.dialog.clear();
              },
            })),
          }),
        );
      },
    },
    {
      title: "STT: Select Language",
      value: "stt.language",
      description: "Choose transcription language",
      slash: { name: "stt-language" },
      onSelect() {
        const current = kv.get("stt.language") || sttDefaultLanguage;
        api.ui.dialog.replace(() =>
          api.ui.DialogSelect({
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
              },
            })),
          }),
        );
      },
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
      },
    },
  ];
}

// Event handlers for mode manager integration
export function onRecordingStart(audioData: Float32Array) {
  recording = true;
  audioChunks = [audioData];
  recordingStartTime = Date.now();
}

export function onRecordingChunk(audioData: Float32Array) {
  if (recording) {
    audioChunks.push(audioData);
  }
}

export function onRecordingStop(options: { submit: boolean }) {
  if (!recording) return;

  recording = false;
  processing = true;

  // Concatenate chunks
  const totalLength = audioChunks.reduce((sum, c) => sum + c.length, 0);
  const audioData = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of audioChunks) {
    audioData.set(chunk, offset);
    offset += chunk.length;
  }
  audioChunks = [];

  // The actual transcription is triggered by the command handler
  // which will call doTranscribePipeline
}