/**
 * Mixed-Language Text Processor
 * Handles text with multiple languages (e.g., Persian + English) for TTS synthesis
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";

export interface MixedLangOptions {
  enabled: boolean;
  primaryLang: string;                    // Default: "en-US"
  secondaryLangs: string[];               // ["fa-IR"]
  defaultVoicePerLang: Record<string, string>;  // lang -> voiceId
  enInFaStrategy: "spell" | "pronounce" | "keep" | "transliterate";  // How to handle English in Persian text
  faInEnStrategy: "transliterate" | "pronounce" | "keep";             // How to handle Persian in English text
  customRules: Record<string, string>;    // User-defined word replacements
  segmentStrategy: "per-sentence" | "per-word" | "single-voice";
}

export interface TextSegment {
  text: string;
  language: string;
  confidence: number;
  voiceId: string;
}

const DEFAULT_OPTIONS: MixedLangOptions = {
  enabled: true,
  primaryLang: "en-US",
  secondaryLangs: ["fa-IR"],
  defaultVoicePerLang: {
    "fa-IR": "fa_IR-gyro-medium",
    "en-US": "af_heart",
    "en-GB": "bf_emma",
    "auto": "af_heart"
  },
  enInFaStrategy: "pronounce",
  faInEnStrategy: "transliterate",
  customRules: {},
  segmentStrategy: "per-sentence"
};

// Persian digits and numbers
const PERSIAN_DIGITS = ["۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹"];
const ARABIC_DIGITS = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"];

// Common English tech terms that should be pronounced in Persian context
const TECH_TERMS_PERSIAN_PRONUNCIATION: Record<string, string> = {
  "API": "اے پی آئی",
  "GitHub": "گیت‌هاب",
  "Git": "گیت",
  "TypeScript": "تایپ‌اسکریپت",
  "JavaScript": "جاوااسکریپت",
  "Python": "پایتون",
  "React": "ری‌اکت",
  "Node": "نود",
  "npm": "ان‌پی‌ام",
  "Docker": "داکر",
  "Kubernetes": "کوبِرنتیز",
  "AI": "اے‌آی",
  "ML": "ام‌ال",
  "LLM": "ال‌ال‌آم",
  "GPU": "جی‌پی‌یو",
  "CPU": "سی‌پی‌یو",
  "RAM": "رم",
  "SSD": "اس‌اس‌دی",
  "JSON": "جِی‌سان",
  "HTTP": "اچ‌تی‌تی‌پی",
  "HTTPS": "اچ‌تی‌تی‌پی‌اس",
  "URL": "یو‌ار‌ال",
  "SQL": "اس‌کیو‌ال",
  "NoSQL": "ناس‌کیو‌ال",
  "REST": "рест",
  "GraphQL": "گراف‌کیو‌ال",
  "WebSocket": "وب‌سوکت",
  "OAuth": "اُات",
  "JWT": "جِی‌وِی‌تی",
  "CSS": "سی‌اس‌اس",
  "HTML": "اچ‌تی‌ام‌ال",
  "VS Code": "وی‌اس کد",
  "CLI": "سی‌ال‌آی",
  "GUI": "جی‌یو‌آی",
  "IDE": "آی‌دی‌ای",
  "PR": "پی‌ار",
  "CI/CD": "سی‌آی‌سی‌دی",
  "DevOps": "دِواپس",
  "Agile": "اجایل",
  "Scrum": "اسکرم",
  "Sprint": "اسپرینت",
  "Kanban": "کانبان",
  "Jira": "جیرا",
  "Confluence": "کانفلوئنس",
  "Slack": "اسَلک",
  "Discord": "دِسکورد",
  "Zoom": "زوم",
  "Google": "گوگل",
  "Microsoft": "مایکروسافت",
  "Amazon": "آمازون",
  "AWS": "اے‌وِی‌اِس",
  "Azure": "اژور",
  "GCP": "جی‌سی‌پی",
  "Linux": "لینوکس",
  "Ubuntu": "اوبونتو",
  "Debian": "دبیان",
  "Arch": "آرچ",
  "Fedora": "فدورا",
  "Windows": "ویندوز",
  "macOS": "مک‌او‌اس",
  "iOS": "آی‌او‌اس",
  "Android": "اندروید",
  "Chrome": "کروم",
  "Firefox": "فایرفاکس",
  "Safari": "سافاری",
  "Edge": "اِج",
  "GitLab": "گیت‌لب",
  "Bitbucket": "بیت‌باکت",
  "Docker": "داکر",
  "Kubernetes": "کوبِرنتیز",
  "Terraform": "ترافرم",
  "Ansible": "آنسیبل",
  "Prometheus": "پروتئوس",
  "Grafana": "گرافانا",
  "Elasticsearch": "الستیک‌سرچ",
  "Redis": "ردیس",
  "PostgreSQL": "پستگرس‌کیو‌ال",
  "MySQL": "مای‌اس‌کیو‌ال",
  "MongoDB": "مینگو‌دی‌بی",
  "GraphQL": "گراف‌کیو‌ال",
  "gRPC": "جی‌آر‌پی‌سی",
  "WebAssembly": "وب‌اسمبلی",
  "Rust": "راست",
  "Go": "گو",
  "Java": "جاوا",
  "Kotlin": "کاتلین",
  "Swift": "سوئیفت",
  "Dart": "دارت",
  "Flutter": "فلاتر",
  "React Native": "ری‌اکت نیتیو",
  "Electron": "الکترون",
  "Tauri": "تائوری",
  "Next.js": "نکست‌جی‌اس",
  "Nuxt": "ناکست",
  "Vite": "وایت",
  "Webpack": "وب‌پک",
  "Rollup": "رول‌آپ",
  "ESLint": "اِس‌لینت",
  "Prettier": "پریتیِر",
  "Jest": "جِست",
  "Vitest": "ویتِست",
  "Cypress": "سایپرس",
  "Playwright": "پلی‌رایتر",
  "Storybook": "استوری‌بوک",
  "Tailwind": "تِیل‌وِند",
  "Bootstrap": "بوت‌استرپ",
  "Material UI": "متریال یو‌آی",
  "Chakra UI": "چاکرا یو‌آی",
  "Prisma": "پریزما",
  "Drizzle": "دریزل",
  "TypeORM": "تایپ‌او‌آرام",
  "Sequelize": "سیکوئل‌آیز",
  "Mongoose": "منگوس",
  "Socket.io": "ساکت‌آی‌او",
  "GraphQL": "گراف‌کیو‌ال",
  "Apollo": "اپولو",
  "Relay": "ریل",
  "URQL": "یو‌آر‌کیو‌ال",
  "React Query": "ری‌اکت کُوَری",
  "SWR": "اس‌دبِل‌یو‌آر",
  "Zustand": "زوستاند",
  "Redux": "ری‌داکس",
  "MobX": "ماب‌ایکس",
  "Recoil": "ریکویل",
  "Jotai": "جوتایی",
  "Valtio": "والتیو",
  "Signals": "سیگنال‌ها",
  "Solid": "سولید",
  "Svelte": "سلوت",
  "Vue": "ویو",
  "Nuxt": "ناکست",
  "Astro": "آسترو",
  "Remix": "رِمِکس",
  "Gatsby": "گتسبی",
  "Eleventy": "الِونتی",
  "Hugo": "هُگو",
  "Jekyll": "جِکیل",
  "Netlify": "نتلیفای",
  "Vercel": "ورسِل",
  "Cloudflare": "کلاودفلر",
  "Firebase": "فایربیس",
  "Supabase": "سوپابیس",
  "PlanetScale": "پلِنتِاسکیل",
  "Neon": "نئون",
  "Turso": "تورسو",
  "Fly.io": "فلای‌آی‌او",
  "Railway": "رِیل‌وِی",
  "Render": "رِندِر",
  "Heroku": "هروکو",
  "DigitalOcean": "دیجیتال‑اُشِن",
  "Linode": "لینود",
  "Vultr": "وُلتر",
  "Hetzner": "هِتزنر",
  "Scaleway": "اسِکیل‌وِی",
  "OVH": "اوِی‌اچ‌وی",
  "OpenAI": "اوپن‌اے‌آی",
  "Anthropic": "آنتروپیک",
  "Google AI": "گوگل اے‌آی",
  "Gemini": "جِمینی",
  "Claude": "کلود",
  "GPT": "جی‌پی‌تی",
  "DALL-E": "دال-ای",
  "Midjourney": "میدجرنی",
  "Stable Diffusion": "استیبل دیفیوژن",
  "Whisper": "وِسپِر",
  "Piper": "پایپر",
  "Kokoro": "کوکورو",
  "Sherpa": "شِرپا",
  "Shenava": "شناوا",
  "VITS": "وِی‌تی‌اس",
  "XTTS": "اِکس‌تی‌تی‌اس",
  "Coqui": "کُکی",
  "MMS-TTS": "ام‌ام‌اس-تی‌تی‌اس",
  "Facebook": "فیس‌بوک",
  "Meta": "مِتا",
  "NVIDIA": "انویدیا",
  "AMD": "اِی‌مِدی",
  "Intel": "اینتل",
  "ARM": "اِرم",
  "CUDA": "کُودا",
  "ROCm": "راکم",
  "Vulkan": "وُلکان",
  "DirectX": "دایरेکت‌اِکس",
  "OpenGL": "اوپن‌جی‌ال",
  "WebGL": "وب‌جی‌ال",
  "WebGPU": "وب‌جی‌پی‌یو",
  "WASM": "وَسم",
  "WebAssembly": "وب‌اسمبلی",
  "LLVM": "اِل‌ال‌وی‌ام",
  "Clang": "کلاَنگ",
  "GCC": "جی‌سی‌سی",
  "MSVC": "ام‌اس‌وی‌سی",
  "Rust": "راست",
  "Cargo": "کارگو",
  "Crates.io": "کریتس‌دات‌آی‌او",
  "Tokio": "توکیو",
  "Async": "اِسنِک",
  "Await": "اِوَیت",
  "Future": "فیوچِر",
  "Stream": "استریم",
  "Iterator": "ایتِرِیتِر",
  "Trait": "ترِیت",
  "Struct": "استراکت",
  "Enum": "اِنام",
  "Match": "مَچ",
  "Option": "اُپشن",
  "Result": "رِزالت",
  "Vec": "وَک",
  "HashMap": "هش‌مپ",
  "BTreeMap": "بی‌ตรี‌مپ",
  "String": "استرینگ",
  "str": "اِستر",
  "Box": "باکس",
  "Arc": "اِرك",
  "Rc": "آر‌سی",
  "RefCell": "رف‌سیل",
  "Mutex": "میوتِکس",
  "RwLock": "آر‌والاک",
  "Channel": "چنل",
  "Sender": "سِندر",
  "Receiver": "ری‌سیور",
  "Select": "سِلِکت",
  "Join": "جُوین",
  "Spawn": "اسپان",
  "Task": "تسک",
  "Runtime": "ران‌تایم",
  "Executor": "اِگزِکیوتِر",
  "Blocking": "بلاکینگ",
  "Sync": "سِنک",
  "Send": "سِند",
  "Pin": "پین",
  "Unpin": "اَن‌پین",
  "Drop": "دراپ",
  "Clone": "کلون",
  "Copy": "کپی",
  "PartialEq": "پارشیال‌اِک‌یو",
  "Eq": "اِک‌یو",
  "Hash": "هش",
  "Debug": "دیباگ",
  "Display": "دِسپلی",
  "Default": "دیفالت",
  "From": "فرام",
  "Into": "اِنتُ",
  "TryFrom": "ترای‌فرام",
  "TryInto": "ترای‌اِنتُ",
  "AsRef": "اِز‌رف",
  "AsMut": "اِز‌مَت",
  "Deref": "دیرِف",
  "DerefMut": "دیرِف‌مَت",
  "Borrow": "بارو",
  "BorrowMut": "بارو‌مَت",
  "Cow": "کاو",
  "PhantomData": "فَنتَم‌دِیتا",
  "NonZero": "نن‌زِرو",
  "MaybeUninit": "مِیبِی‌اَنِینِیت",
  "ManuallyDrop": "مَنِیوِی‌دِرَپ",
  "Cell": "سل",
  "Ref": "رف",
  "UnsafeCell": "اَن‌سِیف‌سِل",
  "Sync": "سِنک",
  "Send": "سِند",
  "Unpin": "اَن‌پین",
  "Freeze": "فریز",
  "Frozen": "فرَزن",
  "Pin": "پین",
  "Unpin": "اَن‌پین",
  "Pointer": "پوینتر",
  "Reference": "رفِرِنس",
  "RawPointer": "راو‌پوینتر",
  "Function": "فانکشن",
  "Closure": "کلُجر",
  "Fn": "اِفِن",
  "FnMut": "اِفِن‌مَت",
  "FnOnce": "اِفِن-اُنس",
  "Generator": "جِِنِرِیتِر",
  "AsyncGenerator": "اِسنِک‌جِِنِرِیتِر",
  "Stream": "استریم",
  "Sink": "سِنک",
  "Future": "فیوچِر",
  "Poll": "پُل",
  "Context": "کانِتِکست",
  "Waker": "وِیکِر",
  "Task": "تسک",
  "Executor": "اِگزِکیوتِر",
  "Runtime": "ران‌تایم",
  "Blocking": "بلاکینگ",
  "Spawn": "اسپان",
  "JoinHandle": "جُوینِهَندل",
  "AbortHandle": "اَبُرتِهَندل",
  "AbortRegistration": "اَبُرت‌رِجِسترِیشِن",
  "LocalSet": "لُکِل‌ست",
  "Enter": "اِنتِر",
  "Exit": "اِگزِت",
  "Park": "پارک",
  "Unpark": "اَن‌پارک",
  "Coop": "کُپ",
  "Budget": "بَجِت",
  "Yield": "ییلد",
  "Ready": "رِدی",
  "Pending": "پِندینگ",
  "Poll": "پُل",
  "Context": "کانِتِکست",
  "Waker": "وِیکِر",
  "Wake": "وِیک",
  "WakeByRef": "وِیک‌بَای‌رِف",
  "Clone": "کلون",
  "Copy": "کپی",
  "PartialEq": "پارشیال‌اِک‌یو",
  "Eq": "اِک‌یو",
  "Hash": "هش",
  "Debug": "دیباگ",
  "Display": "دِسپلی",
  "Default": "دیفالت",
  "From": "فرام",
  "Into": "اِنتُ",
  "TryFrom": "ترای‌فرام",
  "TryInto": "ترای‌اِنتُ",
  "AsRef": "اِز‌رف",
  "AsMut": "اِز‌مَت",
  "Deref": "دیرِف",
  "DerefMut": "دیرِف‌مَت",
  "Borrow": "بارو",
  "BorrowMut": "بارو‌مَت",
  "Cow": "کاو",
  "PhantomData": "فَنتَم‌دِیتا",
  "NonZero": "نن‌زِرو",
  "MaybeUninit": "مِیبِی‌اَنِینِیت",
  "ManuallyDrop": "مَنِیوِی‌دِرَپ",
  "Cell": "سل",
  "Ref": "رف",
  "UnsafeCell": "اَن‌سِیف‌سِل",
  "Sync": "سِنک",
  "Send": "سِند",
  "Unpin": "اَن‌پین",
  "Freeze": "فریز",
  "Frozen": "فرَزن",
  "Pin": "پین",
  "Unpin": "اَن‌پین",
  "Pointer": "پوینتر",
  "Reference": "رفِرِنس",
  "RawPointer": "راو‌پوینتر",
  "Function": "فانکشن",
  "Closure": "کلُجر",
  "Fn": "اِفِن",
  "FnMut": "اِفِن‌مَت",
  "FnOnce": "اِفِن-اُنس",
  "Generator": "جِِنِرِیتِر",
  "AsyncGenerator": "اِسنِک‌جِِنِرِیتِر",
  "Stream": "استریم",
  "Sink": "سِنک",
  "Future": "فیوچِر",
  "Poll": "پُل",
  "Context": "کانِتِکست",
  "Waker": "وِیکِر",
  "Wake": "وِیک",
  "WakeByRef": "وِیک‌بَای‌رِف",
};

function loadCustomRules(): Record<string, string> {
  const configPath = path.join(os.homedir(), ".config", "opencode", "voice", "custom-rules.json");
  if (fs.existsSync(configPath)) {
    try {
      return JSON.parse(fs.readFileSync(configPath, "utf-8"));
    } catch {}
  }
  return {};
}

function saveCustomRules(rules: Record<string, string>) {
  const configDir = path.join(os.homedir(), ".config", "opencode", "voice");
  if (!fs.existsSync(configDir)) fs.mkdirSync(configDir, { recursive: true });
  const configPath = path.join(configDir, "custom-rules.json");
  fs.writeFileSync(configPath, JSON.stringify(rules, null, 2));
}

function detectLanguage(text: string): string {
  // Simple heuristic: check for Persian characters
  const persianChars = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/;
  const hasPersian = persianChars.test(text);
  const hasEnglish = /[a-zA-Z]/.test(text);

  if (hasPersian && hasEnglish) return "mixed";
  if (hasPersian) return "fa-IR";
  if (hasEnglish) return "en-US";
  return "auto";
}

function detectLanguageSegments(text: string): TextSegment[] {
  // Split by sentence boundaries
  const sentences = text.split(/(?<=[.!?؟])\s+/);
  const segments: TextSegment[] = [];

  for (const sentence of sentences) {
    if (!sentence.trim()) continue;

    const lang = detectLanguage(sentence);
    const primaryLang = lang === "mixed" ? "fa-IR" : lang; // Default to Persian for mixed
    const voiceId = DEFAULT_OPTIONS.defaultVoicePerLang[primaryLang] || DEFAULT_OPTIONS.defaultVoicePerLang.auto;

    segments.push({
      text: sentence.trim(),
      language: primaryLang,
      confidence: lang === "mixed" ? 0.7 : 0.95,
      voiceId
    });
  }

  return segments;
}

function applyCustomRules(text: string, rules: Record<string, string>): string {
  let result = text;
  for (const [from, to] of Object.entries(rules)) {
    const regex = new RegExp(`\\b${escapeRegExp(from)}\\b`, "gi");
    result = result.replace(regex, to);
  }
  return result;
}

function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeNumbers(text: string, lang: string): string {
  if (lang === "fa-IR") {
    // Convert English digits to Persian
    return text.replace(/[0-9]/g, (d) => PERSIAN_DIGITS[parseInt(d)]);
  }
  if (lang === "en-US" || lang === "en-GB") {
    // Convert Persian/Arabic digits to English
    return text
      .replace(/[۰-۹]/g, (d) => PERSIAN_DIGITS.indexOf(d).toString())
      .replace(/[٠-٩]/g, (d) => ARABIC_DIGITS.indexOf(d).toString());
  }
  return text;
}

function processMixedLanguage(text: string, options: Partial<MixedLangOptions> = {}): TextSegment[] {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const rules = { ...TECH_TERMS_PERSIAN_PRONUNCIATION, ...loadCustomRules(), ...opts.customRules };

  // Apply custom rules first
  let processedText = applyCustomRules(text, rules);

  // Detect segments
  const segments = detectLanguageSegments(processedText);

  // Normalize numbers per segment language
  return segments.map(seg => ({
    ...seg,
    text: normalizeNumbers(seg.text, seg.language)
  }));
}

function synthesizeMixedText(
  text: string,
  sherpaEngine: any,
  options: Partial<MixedLangOptions> = {}
): Promise<AudioBuffer[]> {
  const segments = processMixedText(text, options);
  const audioChunks: Promise<AudioBuffer>[] = [];

  for (const seg of segments) {
    if (seg.text.trim()) {
      audioChunks.push(
        sherpaEngine.synthesize(seg.text, { voice: seg.voiceId, speed: 1.0 })
      );
    }
  }

  return Promise.all(audioChunks);
}

function concatAudio(buffers: AudioBuffer[]): AudioBuffer {
  if (buffers.length === 0) {
    return { data: new Float32Array(0), sampleRate: 22050, channels: 1 };
  }
  if (buffers.length === 1) return buffers[0];

  const sampleRate = buffers[0].sampleRate;
  const totalLength = buffers.reduce((sum, b) => sum + b.data.length, 0);
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const buf of buffers) {
    result.set(buf.data, offset);
    offset += buf.data.length;
  }
  return { data: result, sampleRate, channels: 1 };
}

function playAudioBuffer(audio: AudioBuffer): Promise<void> {
  // This would use the platform's audio playback (sox, ffplay, etc.)
  // Implementation depends on the platform
  return Promise.resolve();
}

export {
  MixedLangOptions,
  TextSegment,
  DEFAULT_OPTIONS,
  processMixedText,
  synthesizeMixedText,
  concatAudio,
  playAudioBuffer,
  loadCustomRules,
  saveCustomRules,
  TECH_TERMS_PERSIAN_PRONUNCIATION
};