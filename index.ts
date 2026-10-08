/**
 * opencode-voice-universal: Universal Voice Interface for OpenCode
 * Registers the voice_control tool (mode/status/settings management).
 */

import type { PluginInput, Hooks, ToolResult } from "@opencode-ai/plugin";
import { z } from "zod";
import * as fs from "node:fs";
import * as path from "node:path";
import { execFile } from "node:child_process";

const PYTHON = "C:\\Python314\\python.exe";
const HELPER = path.join(
  process.env.USERPROFILE ?? ".",
  ".config",
  "opencode",
  "voice",
  "voice_helper.py"
);
const TMP = path.join(process.env.TEMP ?? ".", "opencode");

function runHelper(
  args: string[],
  timeoutMs: number
): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const resultFile = path.join(TMP, `vh-${Date.now()}-${Math.random().toString(16).slice(2)}.json`);
    const full = [...args, "--result", resultFile];
    execFile(
      PYTHON,
      ["-X", "utf8", HELPER, ...full],
      { timeout: timeoutMs, windowsHide: true, maxBuffer: 8 * 1024 * 1024 },
      (err) => {
        let data: Record<string, unknown> = {};
        try {
          data = JSON.parse(fs.readFileSync(resultFile, "utf-8"));
        } catch {
          data = { ok: false, error: err?.message ?? "no result file" };
        } finally {
          try { fs.unlinkSync(resultFile); } catch { /* ignore */ }
        }
        resolve(data);
      }
    );
  });
}

type VoiceMode = "off" | "manual" | "auto";

type VoiceState = {
  mode: VoiceMode;
  pushToTalkKey: string;
  holdToTalk: boolean;
  wakeWordEngine: "disabled" | "openwakeword" | "porcupine";
  wakeWordKeywords: string[];
  wakeWordSensitivity: number;
  vadSensitivity: number;
  autoSubmit: boolean;
  continuousListening: boolean;
  ttsVoice: string;
  sttModel: string;
  primaryLang: string;
  secondaryLangs: string[];
};

const STATE_FILE = path.join(
  process.env.USERPROFILE ?? ".",
  ".config",
  "opencode",
  "voice-state.json"
);

function defaultState(): VoiceState {
  return {
    mode: "manual",
    pushToTalkKey: "ctrl+r",
    holdToTalk: false,
    wakeWordEngine: "disabled",
    wakeWordKeywords: ["hey jarvis", "jarvis"],
    wakeWordSensitivity: 0.6,
    vadSensitivity: 0.5,
    autoSubmit: true,
    continuousListening: true,
    ttsVoice: "fa_IR-gyro-medium",
    sttModel: "shenava-koochik-int8",
    primaryLang: "fa-IR",
    secondaryLangs: ["en-US"],
  };
}

function loadState(): VoiceState {
  try {
    const raw = fs.readFileSync(STATE_FILE, "utf-8");
    return { ...defaultState(), ...JSON.parse(raw) };
  } catch {
    return defaultState();
  }
}

function saveState(state: VoiceState): void {
  try {
    fs.mkdirSync(path.dirname(STATE_FILE), { recursive: true });
    fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2), "utf-8");
  } catch {
    // state persistence is best-effort
  }
}

function describeState(state: VoiceState): string {
  return JSON.stringify(
    {
      mode: state.mode,
      pushToTalk: state.pushToTalkKey,
      holdToTalk: state.holdToTalk,
      wakeWord: {
        engine: state.wakeWordEngine,
        keywords: state.wakeWordKeywords,
        sensitivity: state.wakeWordSensitivity,
      },
      auto: {
        vadSensitivity: state.vadSensitivity,
        autoSubmit: state.autoSubmit,
        continuousListening: state.continuousListening,
      },
      stt: { model: state.sttModel, language: state.primaryLang },
      tts: { voice: state.ttsVoice },
      mixedLanguage: {
        primary: state.primaryLang,
        secondary: state.secondaryLangs,
      },
    },
    null,
    2
  );
}

export default async function voicePlugin(
  input: PluginInput,
  options: Record<string, unknown> = {}
): Promise<Hooks> {
  const hooks: Hooks = {
    tool: {
      voice_listen: {
        description:
          "Record the microphone (default 5s) and transcribe the speech to text (Persian/English, sherpa-onnx). " +
          "Returns recognized text. Use when the user speaks a voice command.",
        args: {
          duration: z
            .number()
            .optional()
            .describe("Recording length in seconds (default 5, max 20)"),
          vad: z
            .boolean()
            .optional()
            .describe("Trim leading/trailing silence (default true)"),
        },
        async execute({ duration, vad }, ctx): Promise<ToolResult> {
          const state = loadState();
          if (state.mode === "off") {
            return { title: "Voice mode OFF", output: "Voice mode is OFF. Switch to manual or auto first." };
          }
          const dur = Math.min(Math.max(duration ?? 5, 1), 20);
          const n = Math.trunc(dur * 16000);
          const pcm = path.join(TMP, `ot-${Date.now()}.wav`);
          const rec = await runHelper(["record", "--out", pcm, "--duration", String(dur), "--vad"], 30000);
          if (rec.ok !== true) {
            return { title: "Record failed", output: String(rec.error ?? "unknown") };
          }
          const tr = await runHelper(["transcribe", "--wav", pcm, "--lang", "fa"], 60000);
          try { fs.unlinkSync(pcm); } catch { /* ignore */ }
          if (tr.ok !== true) {
            return { title: "Transcribe failed", output: String(tr.error ?? "unknown") };
          }
          let text = String(tr.text ?? "").trim();
          if (text === "") {
            return { title: "No speech detected", output: "No speech was detected. Please speak after the tone and try again." };
          }
          return { title: "Voice transcript", output: text, metadata: { lang: "fa" } };
        },
      },
      voice_speak: {
        description:
          "Synthesize text to speech (TTS) with a Persian (Piper fa_IR) or English voice and playback the audio. " +
          "Use when the assistant wants to respond to the user by voice.",
        args: {
          text: z.string().describe("The text to speak aloud"),
          voice: z
            .string()
            .optional()
            .describe("Voice name: fa_IR-gyro-medium (Persian) or en_US-lessac-medium (English)"),
          play: z
            .boolean()
            .optional()
            .describe("Play the audio through speakers (default true)"),
        },
        async execute({ text, voice, play = true }, ctx): Promise<ToolResult> {
          const state = loadState();
          if (state.mode === "off") {
            return { title: "Voice mode OFF", output: "Voice mode is OFF. Switch to manual or auto first." };
          }
          const wav = path.join(TMP, `ot-tts-${Date.now()}.wav`);
          const sp = voice ?? state.ttsVoice;
          const res = await runHelper(
            ["speak", "--text", text, "--out", wav, "--voice", sp, ...(play ? ["--play"] : [])],
            60000
          );
          if (res.ok !== true) {
            return { title: "TTS failed", output: String(res.error ?? "unknown") };
          }
          const dur = Number(res.duration_s ?? 0).toFixed(2);
          const out = `Spoken (${dur}s, ${String(res.sample_rate)} Hz).\nWAV: ${wav}`;
          return { title: "TTS spoken", output: out };
        },
      },
      voice_control: {
        description:
          "Manage the voice interface: switch voice mode (auto/manual/off), inspect status, " +
          "or change voice settings (push-to-talk key, STT model, TTS voice, wake word, languages).",
        args: {
          action: z
            .enum([
              "status",
              "auto",
              "manual",
              "off",
              "cycle",
              "set",
            ])
            .describe(
              "status=current settings; auto/manual/off=switch mode; cycle=next mode; set=change a setting via optional args"
            ),
          pushToTalkKey: z
            .string()
            .optional()
            .describe('Keyboard shortcut for manual mode, e.g. "ctrl+r" (use with action "set")'),
          sttModel: z
            .string()
            .optional()
            .describe("Speech-to-text model name (use with action \"set\")"),
          ttsVoice: z
            .string()
            .optional()
            .describe("Text-to-speech voice name, e.g. fa_IR-gyro-medium (use with action \"set\")"),
          wakeWord: z
            .string()
            .optional()
            .describe('Comma-separated wake words, e.g. "hey jarvis, jarvis" (use with action "set")'),
          language: z
            .string()
            .optional()
            .describe("Primary language tag, e.g. fa-IR or en-US (use with action \"set\")"),
        },
        async execute({ action, pushToTalkKey, sttModel, ttsVoice, wakeWord, language }): Promise<ToolResult> {
          const state = loadState();

          switch (action) {
            case "status": {
              return { title: `Voice mode: ${state.mode}`, output: describeState(state) };
            }
            case "auto": {
              state.mode = "auto";
              saveState(state);
              return "Voice mode switched to AUTO (continuous listening with VAD).";
            }
            case "manual": {
              state.mode = "manual";
              saveState(state);
              return `Voice mode switched to MANUAL (push-to-talk: ${state.pushToTalkKey}).`;
            }
            case "off": {
              state.mode = "off";
              saveState(state);
              return "Voice mode OFF (no listening, no speaking).";
            }
            case "cycle": {
              const modes: VoiceMode[] = ["off", "manual", "auto"];
              state.mode = modes[(modes.indexOf(state.mode) + 1) % modes.length];
              saveState(state);
              return `Voice mode switched to ${state.mode.toUpperCase()}.`;
            }
            case "set": {
              const changed: string[] = [];
              if (pushToTalkKey !== undefined) {
                state.pushToTalkKey = pushToTalkKey;
                changed.push(`pushToTalkKey=${pushToTalkKey}`);
              }
              if (sttModel !== undefined) {
                state.sttModel = sttModel;
                changed.push(`sttModel=${sttModel}`);
              }
              if (ttsVoice !== undefined) {
                state.ttsVoice = ttsVoice;
                changed.push(`ttsVoice=${ttsVoice}`);
              }
              if (wakeWord !== undefined) {
                state.wakeWordKeywords = wakeWord.split(",").map((k) => k.trim()).filter(Boolean);
                state.wakeWordEngine = state.wakeWordKeywords.length > 0 ? "openwakeword" : "disabled";
                changed.push(`wakeWord=[${state.wakeWordKeywords.join(", ")}]`);
              }
              if (language !== undefined) {
                state.primaryLang = language;
                changed.push(`language=${language}`);
              }
              if (changed.length === 0) {
                return "Nothing to change. Provide at least one of: pushToTalkKey, sttModel, ttsVoice, wakeWord, language.";
              }
              saveState(state);
              return `Updated: ${changed.join(", ")}\n${describeState(state)}`;
            }
          }
        },
      },
    },
  };

  return hooks;
}
