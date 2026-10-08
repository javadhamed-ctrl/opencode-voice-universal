import { defineConfig } from "tsdown";

export default defineConfig({
  entry: "./index.ts",
  outDir: "dist",
  format: "esm",
  target: "node18",
  platform: "node",
  external: ["sherpa-onnx", "@opencode-ai/plugin", "@opencode-ai/sdk"],
  noExternal: ["zod"],
  dts: false,
  sourcemap: true,
  clean: true,
  treeshake: true,
  minify: true,
});