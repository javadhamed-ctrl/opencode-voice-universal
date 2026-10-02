/**
 * Text-to-Speech with sherpa-onnx (Piper, VITS, Kokoro, Matcha)
 * Supports local synthesis with multiple models and voices
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { getSherpaEngine, SherpaEngine, AudioBuffer, VoiceInfo } from "./sherpa.js";
import { getSessionTitle } from "./session.js";
import { processMixedLanguage, concatAudio } from "./mixed-lang.js";

// Load voice registry
const VOICE_REGISTRY_PATH = path.join(os.homedir(), ".config", "opencode", "voice", "voice-registry.json");
let voiceRegistry: any = null;

function loadVoiceRegistry() {
  if (voiceRegistry) return voiceRegistry;
  try {
    if (fs.existsSync(VOICE_REGISTRY_PATH)) {
      voiceRegistry = JSON.parse(fs.readFileSync(VOICE_REGISTRY_PATH, "utf-8"));
    } else {
      // Fallback to built-in registry
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

function getAllVoices(): VoiceInfo[] {
  const registry = getVoiceRegistry();
  const voices: VoiceInfo[] = [];
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
      } as VoiceInfo);
    }
  }
  return voices;
}

function getVoiceInfo(voiceId: string): VoiceInfo | null {
  const voices = getAllVoices();
  return voices.find(v => v.id === voiceId) || null;
}

function getVoicesByLanguage(lang: string): VoiceInfo[] {
  return getAllVoices().filter(v => v.language === lang);
}

function getDefaultVoiceForLanguage(lang: string): string | null {
  const registry = getVoiceRegistry();
  return registry.defaultVoicePerLanguage?.[lang] || registry.defaultVoicePerLanguage?.auto || null;
}

function getFallbackVoicesForLanguage(lang: string): string[] {
  const registry = getVoiceRegistry();
  return registry.fallbackChain?.[lang] || [];
}

// ---- System prompts ----

const SYSTEM_AUTO = `You are a text-to-speech narrator for a coding assistant CLI. Your job is to convert the assistant's markdown output into natural spoken text that is useful and pleasant to listen to.

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

const SYSTEM_MANUAL = `You are a text-to-speech reader for a coding assistant. The user has explicitly requested this text be read aloud. Read the prose content faithfully and in detail.

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

// ---- Session helpers ----

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
      const fullMsg = await client.session
        .message({ sessionID, messageID: msgID }, { throwOnError: true })
        .then((r) => r.data);

      const textParts = (fullMsg?.parts || []).filter((p) => p.type === "text");
      const text = textParts
        .map((p) => p.text || "")
        .join("\n\n")
        .trim();
      if (text) allText.push(text);
    } catch {
      // Skip messages that fail to fetch
    }
  }

  if (allText.length === 0) return null;

  return {
    lastMessageID: assistantIDs[assistantIDs.length - 1],
    text: allText.join("\n\n"),
  };
}

// Sherpa engine instance
let sherpaEngine: SherpaEngine | null = null;
let currentTTSModel: string = "piper-fa-gyro-medium";

function getSherpa(): SherpaEngine {
  if (!sherpaEngine) {
    sherpaEngine = getSherpaEngine();
  }
  return sherpaEngine;
}

async function loadTTSModel(modelId: string = currentTTSModel) {
  const engine = getSherpa();
  await engine.loadTTS(modelId);
  currentTTSModel = modelId;
}

// ---- Public API for TUI plugin ----

export function registerTTS(api, kv, complete, prompts, logger) {
  const client = api.client;
  const systemAuto = prompts?.ttsAuto || SYSTEM_AUTO;
  const systemManual = prompts?.ttsManual || SYSTEM_MANUAL;

  function toast(message, variant = "info") {
    api.ui.toast({ message, variant, duration: 3000 });
  }

  // Load TTS model on startup
  loadTTSModel(kv.get("tts.model") || currentTTSModel).catch(err => {
    logger?.log?.("TTS", `Failed to pre-load TTS model: ${err.message}`, "warn");
  });

  async function normalizeForSpeech(text, systemPrompt) {
    logger?.log?.("TTS", `Normalizing speech chars=${text.length}`, "debug");
    return complete({
      system: systemPrompt,
      prompt: `Convert for text-to-speech:\n\n${text}`,
      config: { maxTokens: 4096 },
    });
  }

  // ---- Audio synthesis with sherpa-onnx ----

  async function speak(text: string, options: { voice?: string; speed?: number } = {}): Promise<void> {
    if (!text) return;
    const line = text.replace(/\n/g, " ").trim();
    if (!line) return;

    const voiceId = options.voice || kv.get("tts.voice") || getDefaultVoiceForLanguage("fa-IR") || "fa_IR-gyro-medium";
    const speed = options.speed || 1.0;

    logger?.log?.("TTS", `Speak requested chars=${line.length} voice=${voiceId} speed=${speed}`, "debug");

    try {
      const engine = getSherpa();
      await loadTTSModel(kv.get("tts.model") || currentTTSModel);

      const audio = await engine.synthesize(line, { voice: voiceId, speed });
      
      // Play audio (platform-specific)
      await playAudio(audio);
      
      logger?.log?.("TTS", "Playback finished", "debug");
    } catch (err) {
      logger?.log?.("TTS", `TTS error: ${err.message}`, "error");
      toast(`TTS error: ${err.message}`, "error");
    }
  }

  // Mixed-language synthesis
  async function speakMixed(text: string, options: { speed?: number } = {}): Promise<void> {
    if (!text) return;
    const speed = options.speed || 1.0;

    logger?.log?.("TTS", `Mixed-language speak chars=${text.length}`, "debug");

    try {
      const engine = getSherpa();
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

      const audioChunks = [];
      for (const seg of segments) {
        if (seg.text.trim()) {
          const audio = await engine.synthesize(seg.text, { voice: seg.voiceId, speed });
          audioChunks.push(audio);
        }
      }

      const combined = concatAudio(audioChunks);
      await playAudio(combined);
    } catch (err) {
      logger?.log?.("TTS", `Mixed TTS error: ${err.message}`, "error");
      toast(`TTS error: ${err.message}`, "error");
    }
  }

  // Platform-specific audio playback
  async function playAudio(audio: AudioBuffer): Promise<void> {
    return new Promise((resolve, reject) => {
      // Platform-specific playback
      if (process.platform === "win32") {
        playWindows(audio, resolve, reject);
      } else if (process.platform === "darwin") {
        playMacOS(audio, resolve, reject);
      } else {
        playLinux(audio, resolve, reject);
      }
    });
  }

  function playWindows(audio: AudioBuffer, resolve: Function, reject: Function) {
    // Use PowerShell to play WAV via .NET
    const wavBuffer = audioBufferToWav(audio);
    const tempFile = path.join(os.tmpdir(), `opencode-tts-${Date.now()}.wav`);
    fs.writeFileSync(tempFile, wavBuffer);
    
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

  function playMacOS(audio: AudioBuffer, resolve: Function, reject: Function) {
    const wavBuffer = audioBufferToWav(audio);
    const tempFile = path.join(os.tmpdir(), `opencode-tts-${Date.now()}.wav`);
    fs.writeFileSync(tempFile, wavBuffer);
    
    try {
      execSync(`afplay "${tempFile}"`, { stdio: "ignore" });
      fs.unlinkSync(tempFile);
      resolve();
    } catch (err) {
      reject(err);
    }
  }

  function playLinux(audio: AudioBuffer, resolve: Function, reject: Function) {
    const wavBuffer = audioBufferToWav(audio);
    const tempFile = path.join(os.tmpdir(), `opencode-tts-${Date.now()}.wav`);
    fs.writeFileSync(tempFile, wavBuffer);
    
    // Try multiple players
    const players = ["mpv", "ffplay", "paplay", "aplay", "play"];
    for (const player of players) {
      try {
        execSync(`which ${player}`, { stdio: "ignore" });
        execSync(`${player} "${tempFile}"`, { stdio: "ignore" });
        fs.unlinkSync(tempFile);
        resolve();
        return;
      } catch {}
    }
    reject(new Error("No audio player found. Install mpv, ffplay, paplay, aplay, or sox (play)"));
  }

  function audioBufferToWav(audio: AudioBuffer): Buffer {
    const int16Data = new Int16Array(audio.data.length);
    for (let i = 0; i < audio.data.length; i++) {
      const s = Math.max(-1, Math.min(1, audio.data[i]));
      int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    const buffer = Buffer.alloc(44 + int16Data.length * 2);
    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

    // RIFF header
    view.setUint32(0, 0x52494646, false);
    view.setUint32(4, 36 + int16Data.length * 2, true);
    view.setUint32(8, 0x57415645, false);

    // fmt chunk
    view.setUint32(12, 0x666D7420, false);
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, audio.sampleRate, true);
    view.setUint32(28, audio.sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);

    // data chunk
    view.setUint32(36, 0x64617461, false);
    view.setUint32(40, int16Data.length * 2, true);

    for (let i = 0; i < int16Data.length; i++) {
      view.setInt16(44 + i * 2, int16Data[i], true);
    }

    return buffer;
  }

  // ---- Session-prefixed announcements ----

  async function speakWithSessionPrefix(sessionID, message, suffix) {
    const sessionTitle = await getSessionTitle(client, sessionID);
    const parts = [];
    if (sessionTitle) parts.push(`Session: ${sessionTitle}.`);
    parts.push(message);
    if (suffix) parts.push(suffix);
    await speak(parts.join(" "));
  }

  // ---- Auto mode ----

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
      "Permission requested. Please check your screen.",
    );
  });

  api.event.on("question.asked", async (event) => {
    if (kv.get("tts.mode", "off") !== "on") return;
    await speakWithSessionPrefix(
      event.properties?.sessionID,
      "A question needs your answer. Please check your screen.",
    );
  });

  // ---- Manual mode ----

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

  // ---- Normalization ----

  async function normalizeForSpeech(text, systemPrompt) {
    logger?.log?.("TTS", `Normalizing speech chars=${text.length}`, "debug");
    return complete({
      system: systemPrompt,
      prompt: `Convert for text-to-speech:\n\n${text}`,
      config: { maxTokens: 4096 },
    });
  }

  // ---- Commands ----

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
      },
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
      },
    },
    {
      title: "TTS: Stop Playback",
      value: "tts.stop",
      description: "Stop current TTS playback",
      keybind: "escape",
      slash: { name: "tts-stop" },
      onSelect() {
        toast("TTS stopped (not implemented)");
      },
    },
    {
      title: "TTS: Select Voice",
      value: "tts.voice",
      description: "Choose TTS voice with preview",
      slash: { name: "tts-voice" },
      async onSelect() {
        const current = kv.get("tts.voice", "fa_IR-gyro-medium");
        const allVoices = getAllVoices();

        // Group by language
        const byLang: Record<string, VoiceInfo[]> = {};
        for (const v of allVoices) {
          if (!byLang[v.language]) byLang[v.language] = [];
          byLang[v.language].push(v);
        }

        const options = [
          { title: "🔍 Filter by language", value: "__filter_lang__", disabled: true },
          { title: "🔍 Filter by gender", value: "__filter_gender__", disabled: true },
          { title: "🔍 Filter by style", value: "__filter_style__", disabled: true },
          { title: "─────────────", value: "__divider__", disabled: true },
        ];

        for (const [lang, langVoices] of Object.entries(byLang)) {
          options.push({ title: `📁 ${lang}`, value: `__group_${lang}__`, disabled: true });
          for (const v of langVoices) {
            const genderIcon = v.gender === "male" ? "👨" : v.gender === "female" ? "👩" : "👤";
            const ageIcon = v.ageGroup === "young" ? "🧒" : v.ageGroup === "adult" ? "🧑" : v.ageGroup === "senior" ? "👴" : "";
            options.push({
              title: `${genderIcon} ${v.name} ${ageIcon}`,
              value: v.id,
              description: `${v.language} • ${v.style?.join(", ") || "neutral"} • ${v.sampleRate}Hz`,
              onSelect: async () => {
                await previewVoice(v.id);
                const confirmed = await api.ui.dialog.confirm({
                  title: `Use ${v.name}?`,
                  message: `Language: ${v.language}\nGender: ${v.gender}\nStyle: ${v.style?.join(", ")}`
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

        api.ui.dialog.replace(() =>
          api.ui.DialogSelect({
            title: "Select TTS Voice",
            current,
            options
          })
        );
      },
    },
    {
      title: "TTS: Select Model",
      value: "tts.model",
      description: "Choose TTS model (Piper, VITS, Kokoro)",
      slash: { name: "tts-model" },
      async onSelect() {
        const engine = getSherpa();
        const models = engine.listTTSModels();
        const current = currentTTSModel;

        api.ui.dialog.replace(() =>
          api.ui.DialogSelect({
            title: "Select TTS Model",
            current,
            options: models.map(m => ({
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
      },
    },
    {
      title: "TTS: Preview Voice",
      value: "tts.preview",
      description: "Preview current voice with sample text",
      slash: { name: "tts-preview" },
      async onSelect() {
        const voiceId = kv.get("tts.voice", "fa_IR-gyro-medium");
        await previewVoice(voiceId);
      },
    }
  ];

  async function previewVoice(voiceId: string) {
    const sampleTexts = {
      "fa-IR": "سلام، این یک پیش‌نمایش صوتی است. این голос پارسی برای تست انتخاب شده است.",
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