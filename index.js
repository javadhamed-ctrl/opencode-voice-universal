/**
 * opencode-voice-universal: Universal Voice Interface for OpenCode
 * 
 * Features:
 * - STT: sherpa-onnx (Shenava Koochik, Whisper, NeMo) + API fallback
 * - TTS: sherpa-onnx (Piper, VITS, Kokoro, Matcha) with mixed-language support
 * - Modes: AUTO (VAD always-listening), MANUAL (push-to-talk), OFF
 * - Wake Word: openwakeword (hey jarvis, jarvis, بیدار شو)
 * - Mixed-language: Persian + English in same paragraph
 * - Voice selection with preview
 * - Pluggable provider architecture
 */

import fs from "node:fs";
import os from "node:os";
import { registerSTT } from "./lib/stt.js";
import { registerTTS } from "./lib/tts.js";
import { registerModeManager, VoiceModeManager } from "./lib/mode-manager.js";
import { createClient } from "./lib/llm-client.js";
import { createLogger } from "./lib/logger.js";

function loadPromptFile(filePath, logger, name) {
  if (!filePath) return null;
  const resolved = filePath.replace(/^~(?=\/|$)/, os.homedir());
  try {
    const prompt = fs.readFileSync(resolved, "utf-8").trim() || null;
    logger?.log(
      "plugin",
      prompt ? `Loaded ${name} prompt: ${resolved}` : `Ignored empty ${name} prompt: ${resolved}`,
      "debug",
    );
    return prompt;
  } catch (err) {
    logger?.log("Plugin", `Failed to load ${name} prompt ${resolved}: ${err.message}`, "warn");
    return null;
  }
}

export default {
  id: "opencode-voice-universal",
  tui: async (api, options) => {
    const { kv } = api;
    const logger = createLogger(api.client);
    logger.log("plugin", "Initializing opencode-voice-universal", "debug");
    const { complete } = createClient(options, logger);

    const prompts = {
      stt: loadPromptFile(options?.sttPrompt, logger, "STT"),
      ttsAuto: loadPromptFile(options?.ttsAutoPrompt, logger, "TTS auto"),
      ttsManual: loadPromptFile(options?.ttsManualPrompt, logger, "TTS manual"),
    };

    // Initialize mode manager
    const modeManager = registerModeManager(api, kv, logger, {
      mode: kv.get("voice.mode", "manual"),
      wakeWord: {
        engine: kv.get("voice.wakeWord.engine", "disabled"),
        keywords: kv.get("voice.wakeWord.keywords", ["hey jarvis", "jarvis", "بیدار شو"]),
        sensitivity: kv.get("voice.wakeWord.sensitivity", 0.6),
        customKeywords: kv.get("voice.wakeWord.customKeywords", {})
      },
      auto: {
        vadSensitivity: kv.get("voice.auto.vadSensitivity", 0.5),
        minSpeechDuration: kv.get("voice.auto.minSpeechDuration", 500),
        maxSilenceDuration: kv.get("voice.auto.maxSilenceDuration", 2000),
        autoSubmit: kv.get("voice.auto.autoSubmit", true),
        continuousListening: kv.get("voice.auto.continuousListening", true)
      },
      manual: {
        pushToTalkKey: kv.get("voice.manual.pushToTalkKey", "ctrl+r"),
        holdToTalk: kv.get("voice.manual.holdToTalk", false),
        doubleTapTimeout: kv.get("voice.manual.doubleTapTimeout", 300)
      }
    });

    // Load STT and TTS
    const sttCommands = registerSTT(api, kv, complete, prompts, options, logger);
    const ttsCommands = registerTTS(api, kv, complete, prompts, logger);

    // Voice mode commands
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
            duration: 3000 
          });
        },
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
            duration: 3000 
          });
        },
      },
      {
        title: "Voice: Cycle Mode",
        value: "voice.cycle",
        description: "Cycle through OFF → MANUAL → AUTO → OFF",
        keybind: "ctrl+shift+v",
        slash: { name: "voice-mode" },
        onSelect() {
          modeManager.toggleMode();
          const newMode = modeManager.getMode();
          kv.set("voice.mode", newMode);
          api.ui.toost({ 
            message: `Voice mode: ${newMode.toUpperCase()}`, 
            variant: newMode === "off" ? "warning" : "success",
            duration: 3000 
          });
        },
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
            ].map(o => ({ ...o, onSelect: () => {
              config.wakeWord.engine = o.value;
              modeManager.updateConfig({ wakeWord: config.wakeWord });
              kv.set("voice.wakeWord.engine", o.value);
            }}))
          });
          if (!engine) return;
          config.wakeWord.engine = engine;
          
          if (engine !== "disabled") {
            const keywords = await api.ui.dialog.input({
              title: "Wake Keywords (comma-separated)",
              placeholder: "hey jarvis, jarvis, بیدار شو",
              value: config.wakeWord.keywords.join(", ")
            });
            if (keywords) {
              config.wakeWord.keywords = keywords.split(",").map(k => k.trim());
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
        },
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
        },
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
        },
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
            `🎤 Voice Mode: ${mode.toUpperCase()}`,
            `🗣️ TTS Voice: ${voices}`,
            `🎧 STT Model: ${sttModel}`,
            `🔊 Wake Word: ${config.wakeWord.engine === "disabled" ? "OFF" : config.wakeWord.engine} (${config.wakeWord.keywords.join(", ")})`,
            `🎯 AUTO: VAD=${config.auto.vadSensitivity}, Submit=${config.auto.autoSubmit}`,
            `⌨️ MANUAL: Key=${config.manual.pushToTalkKey}, Hold=${config.manual.holdToTalk}`
          ].join("\n");
          
          api.ui.dialog.info({
            title: "Voice Status",
            message: status
          });
        },
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
            ].map(o => ({ ...o, onSelect: () => {} }))
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
              { title: "Pronounce with Persian accent", value: "pronounce", description: "API -> اے پی آئی" },
              { title: "Spell out", value: "spell", description: "API -> اے پی آی" },
              { title: "Keep as-is", value: "keep" }
            ].map(o => ({ ...o, onSelect: () => {} }))
          });
          
          api.ui.toast({ message: "Mixed language settings saved", variant: "success" });
        },
      }
    ];

    // Register all commands
    const sttCommands = registerSTT(api, kv, complete, prompts, options, logger);
    const ttsCommands = registerTTS(api, kv, complete, prompts, logger);
    const modeCommandsWithHandler = modeCommands.map(cmd => ({
      ...cmd,
      onSelect: cmd.onSelect
    }));

    api.command.register(() => [...sttCommands, ...ttsCommands, ...modeCommandsWithHandler]);
  },
};

// Configuration helpers
export interface VoiceConfig {
  mode: "auto" | "manual" | "off";
  stt: {
    provider: string;
    model: string;
    language: string;
    endpoint?: string;
    apiModel?: string;
    apiKeyEnv?: string;
  };
  tts: {
    provider: string;
    model: string;
    voice: string;
    language: string;
  };
  mixedLanguage: {
    enabled: boolean;
    primaryLang: string;
    secondaryLangs: string[];
    defaultVoicePerLang: Record<string, string>;
    enInFaStrategy: "spell" | "pronounce" | "keep";
    faInEnStrategy: "transliterate" | "pronounce" | "keep";
  };
  wakeWord: {
    engine: "openwakeword" | "porcupine" | "disabled";
    keywords: string[];
    sensitivity: number;
  };
  mode: {
    current: "auto" | "manual" | "off";
    auto: {
      vadSensitivity: number;
      minSpeechDuration: number;
      maxSilenceDuration: number;
      autoSubmit: boolean;
      continuousListening: boolean;
    };
    manual: {
      pushToTalkKey: string;
      holdToTalk: boolean;
      doubleTapTimeout: number;
    };
  };
}