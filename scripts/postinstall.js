/**
 * Post-install script
 * Sets up the voice plugin environment
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execSync } from "node:child_process";

const CONFIG_DIR = path.join(os.homedir(), ".config", "opencode", "voice");
const MODELS_DIR = path.join(os.homedir(), ".local", "share", "opencode-voice", "models");

function createDirs() {
  if (!fs.existsSync(CONFIG_DIR)) fs.mkdirSync(CONFIG_DIR, { recursive: true });
  if (!fs.existsSync(MODELS_DIR)) fs.mkdirSync(MODELS_DIR, { recursive: true });
}

function checkDependencies() {
  const checks = [
    { name: "Node.js", cmd: "node --version" },
    { name: "npm", cmd: "npm --version" },
    { name: "sherpa-onnx", check: () => {
      try {
        require.resolve("sherpa-onnx");
        return true;
      } catch {
        return false;
      }
    }}
  ];

  console.log("🔍 Checking dependencies...\n");
  for (const check of checks) {
    try {
      if (check.cmd) {
        execSync(check.cmd, { stdio: "pipe" });
        console.log(`  ✅ ${check.name}`);
      } else if (check.check && check.check()) {
        console.log(`  ✅ ${check.name}`);
      } else {
        console.log(`  ❌ ${check.name} - not found`);
      }
    } catch {
      console.log(`  ❌ ${check.name} - not found`);
    }
  }
}

function createDefaultConfig() {
  const configPath = path.join(CONFIG_DIR, "config.json");
  if (!fs.existsSync(configPath)) {
    const defaultConfig = {
      stt: {
        provider: "sherpa-onnx",
        model: "shenava-koochik-int8",
        language: "fa-IR"
      },
      tts: {
        provider: "sherpa-onnx",
        model: "piper-fa-gyro-medium",
        voice: "fa_IR-gyro-medium",
        language: "fa-IR"
      },
      mixedLanguage: {
        enabled: true,
        primaryLang: "en-US",
        secondaryLangs: ["fa-IR"],
        defaultVoicePerLang: {
          "en-US": "af_heart",
          "fa-IR": "fa_IR-gyro-medium"
        }
      },
      mode: {
        current: "manual",
        auto: { vadSensitivity: 0.5, autoSubmit: true },
        manual: { pushToTalkKey: "ctrl+r", holdToTalk: false }
      },
      wakeWord: {
        engine: "disabled",
        keywords: ["hey jarvis", "jarvis", "بیدار شو"],
        sensitivity: 0.6
      }
    };
    fs.writeFileSync(configPath, JSON.stringify(defaultConfig, null, 2));
    console.log(`\n📝 Created default config: ${configPath}`);
  }
}

function checkModels() {
  console.log("\n📦 Checking models...");
  const requiredModels = [
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
    "vits-piper-fa_IR-gyro-medium.onnx",
    "silero_vad.onnx"
  ];

  let missing = 0;
  for (const model of requiredModels) {
    const modelPath = path.join(MODELS_DIR, model);
    if (fs.existsSync(modelPath)) {
      const stats = fs.statSync(modelPath);
      const mb = (stats.size / 1024 / 1024).toFixed(1);
      console.log(`  ✅ ${model} (${mb} MB)`);
    } else {
      console.log(`  ❌ ${model} - MISSING`);
      missing++;
    }
  }

  if (missing > 0) {
    console.log(`\n⚠️  ${missing} model(s) missing. Run: npm run download:models`);
  } else {
    console.log("\n✅ All required models present!");
  }
}

function main() {
  console.log("🎤 opencode-voice-universal - Post-install Setup\n");
  createDirs();
  checkDependencies();
  createDefaultConfig();
  checkModels();
  console.log("\n✨ Setup complete!");
  console.log("\nNext steps:");
  console.log("  1. Download models: npm run download:models");
  console.log("  2. Add to opencode tui.json:");
  console.log('     "plugin": [["opencode-voice-universal", { ... }]]');
  console.log("  3. Restart opencode and use /voice-config");
}

main();