/**
 * Voice Mode Manager
 * Handles AUTO (always listening with VAD) and MANUAL (push-to-talk) modes
 * Also manages Wake Word detection
 */

import { EventEmitter } from "node:events";
import { SherpaEngine, AudioBuffer } from "./sherpa.js";

export type VoiceMode = "auto" | "manual" | "off";

export interface WakeWordConfig {
  engine: "openwakeword" | "porcupine" | "disabled";
  modelPath?: string;
  keywords: string[];                    // ["hey jarvis", "jarvis", "بیدار شو"]
  sensitivity: number;                   // 0.5 default
  customKeywords: Record<string, string>; // user keyword -> model mapping
}

export interface AutoModeConfig {
  vadSensitivity: number;                // 0.0 - 1.0
  minSpeechDuration: number;             // ms, minimum speech to trigger
  maxSilenceDuration: number;            // ms, silence before processing
  autoSubmit: boolean;                   // true = submit after transcribe
  continuousListening: boolean;          // true = keep listening after submit
}

export interface ManualModeConfig {
  pushToTalkKey: string;                 // "ctrl+r"
  holdToTalk: boolean;                   // false = toggle, true = hold
  doubleTapTimeout: number;              // ms for double-tap detection
}

export interface VoiceModeConfig {
  mode: VoiceMode;
  wakeWord: WakeWordConfig;
  auto: AutoModeConfig;
  manual: ManualModeConfig;
}

const DEFAULT_CONFIG: VoiceModeConfig = {
  mode: "manual",
  wakeWord: {
    engine: "openwakeword",
    keywords: ["hey jarvis", "jarvis", "بیدار شو"],
    sensitivity: 0.6,
    customKeywords: {}
  },
  auto: {
    vadSensitivity: 0.5,
    minSpeechDuration: 500,
    maxSilenceDuration: 2000,
    autoSubmit: true,
    continuousListening: true
  },
  manual: {
    pushToTalkKey: "ctrl+r",
    holdToTalk: false,
    doubleTapTimeout: 300
  }
};

interface RecordingState {
  isRecording: boolean;
  startTime: number;
  audioChunks: Float32Array[];
  silenceStart: number;
  hasSpeech: boolean;
}

class VoiceModeManager extends EventEmitter {
  private config: VoiceModeConfig;
  private sherpaEngine: SherpaEngine;
  private recordingState: RecordingState;
  private vadInterval: NodeJS.Timeout | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private audioWorklet: AudioWorkletNode | null = null;
  private wakeWordDetector: any = null; // openwakeword instance
  private lastKeyPress: number = 0;

  constructor(sherpaEngine: SherpaEngine, config: Partial<VoiceModeConfig> = {}) {
    super();
    this.sherpaEngine = sherpaEngine;
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

  setMode(mode: VoiceMode) {
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

  getMode(): VoiceMode {
    return this.config.mode;
  }

  toggleMode() {
    const modes: VoiceMode[] = ["off", "manual", "auto"];
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

  private startVADLoop() {
    this.vadInterval = setInterval(async () => {
      if (!this.recordingState.isRecording) return;

      // Check for silence timeout
      if (this.recordingState.hasSpeech && this.recordingState.silenceStart > 0) {
        const silenceDuration = Date.now() - this.recordingState.silenceStart;
        if (silenceDuration >= this.config.auto.maxSilenceDuration) {
          await this.processRecording();
        }
      }
    }, 100);
  }

  // ==================== MANUAL Mode ====================

  async handlePushToTalk(isKeyDown: boolean) {
    if (this.config.mode !== "manual") return;

    const now = Date.now();

    if (isKeyDown) {
      // Key pressed
      if (this.config.manual.holdToTalk) {
        // Hold-to-talk: start recording
        if (!this.recordingState.isRecording) {
          await this.startRecording();
        }
      } else {
        // Toggle mode
        if (this.recordingState.isRecording) {
          await this.stopRecording();
        } else {
          await this.startRecording();
        }
      }
      this.lastKeyPress = now;
    } else {
      // Key released
      if (this.config.manual.holdToTalk && this.recordingState.isRecording) {
        await this.stopRecording();
      }
    }
  }

  async handleDoubleTap() {
    if (this.config.mode !== "manual") return;
    const now = Date.now();
    if (now - this.lastKeyPress < this.config.manual.doubleTapTimeout) {
      // Double tap detected - toggle mode
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

    // Start capturing audio
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

  private async processRecording() {
    if (this.recordingState.audioChunks.length === 0) return;

    // Concatenate audio chunks
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

    // Transcribe
    try {
      this.emit("transcribing");
      const result = await this.sherpaEngine.transcribe(audioData, 16000);
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
      // Restart recording for continuous listening
      setTimeout(() => this.startRecording(), 500);
    }
  }

  // ==================== Audio Input ====================

  private async initAudioInput() {
    if (typeof navigator !== "undefined" && navigator.mediaDevices) {
      // Browser environment
      try {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: 16000,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        });

        this.audioContext = new AudioContext({ sampleRate: 16000 });
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);

        // Use AudioWorklet for VAD processing
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
      // Node.js environment - use sox or similar
      this.emit("error", new Error("Node.js audio input not implemented. Use browser or native recorder."));
    }
  }

  private closeAudioInput() {
    if (this.audioWorklet) {
      this.audioWorklet.disconnect();
      this.audioWorklet = null;
    }
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
      this.mediaStream = null;
    }
  }

  private captureAudio() {
    // In browser, audio is captured via AudioWorklet
    // In Node.js, would use sox/rec process
  }

  private handleAudioChunk(audioData: Float32Array) {
    if (!this.recordingState.isRecording) return;

    this.recordingState.audioChunks.push(audioData);

    // VAD check
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
        // Dynamic import for openwakeword
        const { OpenWakeWord } = await import("openwakeword");
        this.wakeWordDetector = new OpenWakeWord({
          model: this.config.wakeWord.modelPath || "models/hey_jarvis.onnx",
          threshold: this.config.wakeWord.sensitivity
        });

        this.wakeWordDetector.on("detected", (keyword: string) => {
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

  updateConfig(updates: Partial<VoiceModeConfig>) {
    this.config = { ...this.config, ...updates };
    this.emit("configChange", this.config);
  }

  getConfig(): VoiceModeConfig {
    return { ...this.config };
  }

  // ==================== Cleanup ====================

  destroy() {
    this.stopAutoListening();
    this.stopWakeWord();
    this.closeAudioInput();
    this.removeAllListeners();
  }
}

// ==================== Factory Function ====================

function registerModeManager(
  api: any,
  kv: any,
  logger: any,
  config: Partial<VoiceModeConfig> = {}
): VoiceModeManager {
  const modeManager = new VoiceModeManager(api, kv, logger, config);
  return modeManager;
}

// Browser AudioWorklet for VAD
const VAD_PROCESSOR_CODE = `
class VADProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(512);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0) {
      const channel = input[0];
      for (let i = 0; i < channel.length; i++) {
        this.buffer[this.bufferIndex++] = channel[i];
        if (this.bufferIndex >= 512) {
          this.port.postMessage(new Float32Array(this.buffer));
          this.bufferIndex = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("vad-processor", VADProcessor);
`;

export {
  VoiceModeConfig,
  WakeWordConfig,
  AutoModeConfig,
  ManualModeConfig,
  VoiceMode,
  DEFAULT_CONFIG,
  VoiceModeManager,
  registerModeManager,
  VAD_PROCESSOR_CODE
};