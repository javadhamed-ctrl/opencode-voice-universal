const esbuild = require('esbuild');
const fs = require('fs');
const path = require('path');

async function bundle() {
  const outDir = path.join(__dirname, '..', 'dist');
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true });
  }

  await esbuild.build({
    entryPoints: ['index.ts'],
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node18',
    outfile: path.join(outDir, 'voice-universal.js'),
    external: [
      'sherpa-onnx',
      '@xenova/transformers',
      'node:fs',
      'node:path',
      'node:os',
      'node:events',
      'node:child_process',
      'node:https',
      'node:url'
    ],
    format: 'esm',
    sourcemap: true,
    treeShaking: true,
  });

  // Copy the bundled file to plugins directory
  const pluginDir = path.join(process.env.HOME || process.env.USERPROFILE, '.config', 'opencode', 'plugins');
  const destFile = path.join(pluginDir, 'voice-universal.js');
  fs.copyFileSync(path.join(outDir, 'voice-universal.js'), destFile);
  console.log(`Bundled to ${destFile}`);
}

bundle().catch(() => process.exit(1));