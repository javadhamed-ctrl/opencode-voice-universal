/**
 * Model Download Script (CommonJS for compatibility)
 * Downloads sherpa-onnx models for STT (Shenava) and TTS (Piper Persian)
 */

const fs = require("fs");
const path = require("path");
const os = require("os");
const https = require("https");

const MODELS_DIR = path.join(os.homedir(), ".local", "share", "opencode-voice", "models");

const MODELS = {
  "encoder.int8.onnx": {
    url: "https://huggingface.co/Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx/resolve/main/encoder.int8.onnx",
    size: 131 * 1024 * 1024,
    description: "Shenava Koochik INT8 Encoder (~131MB)"
  },
  "decoder.int8.onnx": {
    url: "https://huggingface.co/Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx/resolve/main/decoder.int8.onnx",
    size: 4 * 1024 * 1024,
    description: "Shenava Koochik INT8 Decoder (~4MB)"
  },
  "joiner.int8.onnx": {
    url: "https://huggingface.co/Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx/resolve/main/joiner.int8.onnx",
    size: 1.4 * 1024 * 1024,
    description: "Shenava Koochik INT8 Joiner (~1.4MB)"
  },
  "tokens.txt": {
    url: "https://huggingface.co/Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx/resolve/main/tokens.txt",
    size: 50 * 1024,
    description: "Shenava Tokens (~50KB)"
  },
  "vits-piper-fa_IR-gyro-medium.onnx": {
    url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/gyro/medium/fa_IR-gyro-medium.onnx?download=true",
    size: 50 * 1024 * 1024,
    description: "Piper Persian Gyro Male Medium (~50MB)"
  },
  "vits-piper-fa_IR-amir-medium.onnx": {
    url: "https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx?download=true",
    size: 50 * 1024 * 1024,
    description: "Piper Persian Amir Male Medium (~50MB)"
  },
  "silero_vad.onnx": {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    size: 1.5 * 1024 * 1024,
    description: "Silero VAD (~1.5MB)"
  },
  "espeak-ng-data.zip": {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.zip",
    size: 9 * 1024 * 1024,
    description: "espeak-ng data for Piper (~9MB)"
  }
};

function downloadFile(url, destPath, description) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);
    let downloaded = 0;
    let lastLog = Date.now();

    const request = https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301 || response.statusCode === 307 || response.statusCode === 308) {
        // Handle redirect
        const location = response.headers.location;
        if (location) {
          downloadFile(location, destPath, description).then(resolve).catch(reject);
        } else {
          reject(new Error(`Redirect without location header: ${response.statusCode}`));
        }
        return;
      }
      if (response.statusCode !== 200) {
        reject(new Error(`Failed to download ${description}: ${response.statusCode}`));
        return;
      }

      const total = parseInt(response.headers["content-length"] || "0", 10);

      response.on("data", (chunk) => {
        downloaded += chunk.length;
        const now = Date.now();
        if (now - lastLog > 2000) {
          const pct = total ? ((downloaded / total) * 100).toFixed(1) : "?";
          const mb = (downloaded / 1024 / 1024).toFixed(1);
          console.log(`  ${description}: ${mb} MB (${pct}%)`);
          lastLog = now;
        }
      });

      response.pipe(file);
      file.on("finish", () => {
        file.close();
        const mb = (downloaded / 1024 / 1024).toFixed(1);
        console.log(`  ✅ ${description} downloaded (${mb} MB)`);
        resolve();
      });
    }).on("error", (err) => {
      fs.unlink(destPath, () => {});
      reject(err);
    });
  });
}

function extractTarGz(filePath, destDir) {
  const { execSync } = require("child_process");
  try {
    execSync(`tar -xzf "${filePath}" -C "${destDir}"`, { stdio: "inherit" });
    fs.unlinkSync(filePath);
  } catch (err) {
    console.error(`Failed to extract ${filePath}:`, err);
  }
}

async function main() {
  console.log("📥 Downloading sherpa-onnx models for opencode-voice-universal\n");

  if (!fs.existsSync(MODELS_DIR)) {
    fs.mkdirSync(MODELS_DIR, { recursive: true });
  }

  const toDownload = process.argv.slice(2);
  const modelsToDownload = toDownload.length > 0
    ? toDownload.filter(k => MODELS[k])
    : Object.keys(MODELS);

  console.log(`Models to download: ${modelsToDownload.join(", ")}\n`);

  for (const modelName of modelsToDownload) {
    const model = MODELS[modelName];
    if (!model) {
      console.log(`⚠️  Unknown model: ${modelName}`);
      continue;
    }

    const destPath = path.join(MODELS_DIR, modelName);

    if (fs.existsSync(destPath)) {
      console.log(`⏭️  ${model.description} already exists, skipping`);
      continue;
    }

console.log(`\n⬇️  Downloading: ${model.description}`);
      try {
        await downloadFile(model.url, destPath, model.description);

        if (modelName.endsWith(".zip")) {
          console.log(`📦 Extracting ${modelName}...`);
          const { execSync } = require("child_process");
          // Use PowerShell to extract zip on Windows
          execSync(`powershell -Command "Expand-Archive -Path '${destPath}' -DestinationPath '${MODELS_DIR}' -Force"`, { stdio: "inherit" });
          fs.unlinkSync(destPath);
        }
      } catch (err) {
      console.error(`❌ Failed to download ${modelName}:`, err);
      process.exitCode = 1;
    }
  }

  console.log("\n✅ All models downloaded!");
  console.log(`Models location: ${MODELS_DIR}`);
}

main().catch(console.error);