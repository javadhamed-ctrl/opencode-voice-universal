/**
 * opencode-voice-universal: Universal Voice Interface for OpenCode - Main Entry Point
 * 
 * This file re-exports all the necessary modules for the voice plugin.
 */

export { registerSTT } from "./stt.js";
export { registerTTS } from "./tts.js";
export { createClient } from "./llm-client.js";
export { createLogger } from "./logger.js";
export { createVoiceTool } from "./voice-tool.js";
export { processMixedLanguage, concatAudio } from "./mixed-lang.js";
export { SherpaEngine, getSherpaEngine } from "./sherpa.js";

export type { 
  STTModelConfig, 
  TTSModelConfig, 
  VADConfig, 
  STTResult, 
  TTSOptions, 
  AudioBuffer, 
  VoiceInfo 
} from "./sherpa.js";

export type { MixedLangOptions, TextSegment } from "./mixed-lang.js";

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
} from "./mode-manager.js";