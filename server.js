require("dotenv").config();

const express = require("express");
const path = require("path");
const multer = require("multer");
const OpenAI = require("openai");
const { toFile } = require("openai/uploads");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { GoogleGenAI } = require("@google/genai");

const app = express();
const port = Number(process.env.PORT || 3000);
const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;
const openAiKey = process.env.OPENAI_API_KEY;
const defaultTemperature = Number(process.env.GEMINI_TEMPERATURE || 0.15);
const defaultTopP = Number(process.env.GEMINI_TOP_P || 0.7);
const defaultTopK = Number(process.env.GEMINI_TOP_K || 10);
const defaultMaxOutputTokens = Number(process.env.GEMINI_MAX_OUTPUT_TOKENS || 120);
const defaultHistoryTurns = Number(process.env.GEMINI_MAX_HISTORY_TURNS || 4);
const defaultHardCharLimit = Number(process.env.GEMINI_HARD_CHAR_LIMIT || 650);
const preferredImageModel = process.env.GEMINI_IMAGE_MODEL || "gemini-2.5-flash-image";
const huggingFaceToken = process.env.HF_TOKEN;
const huggingFaceProvider = process.env.HF_PROVIDER || "wavespeed";
const huggingFaceTextProvider = process.env.HF_TEXT_PROVIDER || "hf-inference";
const preferredHfImageModel = process.env.HF_IMAGE_MODEL || "black-forest-labs/FLUX.1-schnell";
const preferredHfTextModel = process.env.HF_TEXT_MODEL || "google/gemma-3-12b-it";
const preferredHfSttModel = process.env.HF_STT_MODEL || "openai/whisper-large-v3";
const preferredHfTtsModel = process.env.HF_TTS_MODEL || "facebook/mms-tts-eng";
const huggingFaceTtsEndpoint = (process.env.HF_TTS_ENDPOINT || "").trim();

if (!apiKey) {
  console.error("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable. Add it to a .env file in the project root.");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(apiKey);
const imageAI = new GoogleGenAI({ apiKey });
const openai = openAiKey ? new OpenAI({ apiKey: openAiKey }) : null;
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 8 * 1024 * 1024,
    files: 6
  }
});

app.use(express.json({ limit: "2mb" }));
app.use(express.static(path.join(__dirname)));

function clamp(value, min, max, fallback) {
  if (Number.isNaN(value)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, value));
}

function toGeminiContents(history, maxHistoryTurns) {
  const textOnlyMessages = history
    .filter((message) => message && message.type === "text" && typeof message.text === "string" && message.text.trim())
    .slice(-Math.max(1, maxHistoryTurns) * 2);

  return textOnlyMessages.map((message) => ({
    role: message.role === "assistant" ? "model" : "user",
    parts: [{ text: message.text }]
  }));
}

function parseJsonField(value, fallback) {
  if (typeof value !== "string") {
    return fallback;
  }

  try {
    return JSON.parse(value);
  } catch (error) {
    return fallback;
  }
}

function isImageFile(file) {
  return file && typeof file.mimetype === "string" && file.mimetype.startsWith("image/");
}

function isTextLikeFile(file) {
  if (!file) {
    return false;
  }

  const mime = (file.mimetype || "").toLowerCase();
  const ext = path.extname(file.originalname || "").toLowerCase();
  const textExt = [".txt", ".md", ".csv", ".json"];

  return (
    mime.startsWith("text/")
    || mime === "application/json"
    || textExt.includes(ext)
  );
}

function buildAttachmentParts(files) {
  const parts = [];
  const notices = [];

  files.forEach((file) => {
    if (isImageFile(file)) {
      parts.push({
        inlineData: {
          mimeType: file.mimetype,
          data: file.buffer.toString("base64")
        }
      });
      return;
    }

    if (isTextLikeFile(file)) {
      const rawText = file.buffer.toString("utf8");
      const maxChars = 12000;
      const clipped = rawText.length > maxChars
        ? `${rawText.slice(0, maxChars)}\n\n[File truncated due to size]`
        : rawText;

      parts.push({ text: `Attached file: ${file.originalname}\n${clipped}` });
      return;
    }

    notices.push(`Skipped unsupported file type: ${file.originalname}`);
  });

  return { parts, notices };
}

function attachFilesToLatestUserTurn(contents, files) {
  if (!files || files.length === 0) {
    return;
  }

  const { parts, notices } = buildAttachmentParts(files);
  if (parts.length === 0 && notices.length === 0) {
    return;
  }

  for (let i = contents.length - 1; i >= 0; i -= 1) {
    if (contents[i].role === "user") {
      contents[i].parts.push(...parts);

      if (notices.length > 0) {
        contents[i].parts.push({ text: notices.join("\n") });
      }
      return;
    }
  }

  const fallbackParts = [{ text: "Attached files" }, ...parts];
  if (notices.length > 0) {
    fallbackParts.push({ text: notices.join("\n") });
  }

  contents.push({
    role: "user",
    parts: fallbackParts
  });
}

function enforceCharLimit(text, maxChars) {
  if (!text || text.length <= maxChars) {
    return text;
  }

  const truncated = text.slice(0, Math.max(0, maxChars - 3)).trimEnd();
  return `${truncated}...`;
}

function getInlineImagePart(response) {
  const candidates = response && Array.isArray(response.candidates) ? response.candidates : [];
  for (const candidate of candidates) {
    const parts = candidate && candidate.content && Array.isArray(candidate.content.parts)
      ? candidate.content.parts
      : [];

    for (const part of parts) {
      if (part && part.inlineData && part.inlineData.data && part.inlineData.mimeType) {
        return part.inlineData;
      }
    }
  }

  return null;
}

function getImageModelCandidates(chosenModel) {
  const candidates = [
    chosenModel,
    preferredImageModel,
    "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image",
    "gemini-2.0-flash-exp-image-generation",
    "gemini-2.0-flash-preview-image-generation"
  ].filter((name) => typeof name === "string" && name.trim());

  return [...new Set(candidates)];
}

function isHuggingFaceImageModel(modelName) {
  if (typeof modelName !== "string") {
    return false;
  }

  if (modelName.startsWith("hf-image:")) {
    return true;
  }

  if (!modelName.startsWith("hf:")) {
    return false;
  }

  const legacyId = modelName.replace(/^hf:/, "").toLowerCase();
  return legacyId.includes("flux") || legacyId.includes("stable-diffusion") || legacyId.includes("image");
}

function isHuggingFaceTextModel(modelName) {
  if (typeof modelName !== "string") {
    return false;
  }

  if (modelName.startsWith("hf-text:")) {
    return true;
  }

  if (!modelName.startsWith("hf:")) {
    return false;
  }

  return !isHuggingFaceImageModel(modelName);
}

function getHfModelId(modelName) {
  if (typeof modelName !== "string") {
    return "";
  }

  return modelName
    .replace(/^hf-tts:/, "")
    .replace(/^hf-stt:/, "")
    .replace(/^hf-image:/, "")
    .replace(/^hf-text:/, "")
    .replace(/^hf:/, "")
    .trim();
}

function isHuggingFaceSttModel(modelName) {
  return typeof modelName === "string" && modelName.startsWith("hf-stt:");
}

function isHuggingFaceTtsModel(modelName) {
  return typeof modelName === "string" && modelName.startsWith("hf-tts:");
}

function getHuggingFaceTtsModelCandidates(modelName) {
  const selectedModel = getHfModelId(modelName);
  const candidates = [
    selectedModel,
    preferredHfTtsModel,
    "facebook/mms-tts-eng",
    "espnet/kan-bayashi_ljspeech_vits"
  ].filter((name) => typeof name === "string" && name.trim());

  return [...new Set(candidates)];
}

function getHuggingFaceModelCandidates(modelName) {
  const selectedModel = getHfModelId(modelName);
  const candidates = [
    selectedModel,
    preferredHfImageModel,
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0"
  ].filter((name) => typeof name === "string" && name.trim());

  return [...new Set(candidates)];
}

function getHuggingFaceTextModelCandidates(modelName) {
  const selectedModel = getHfModelId(modelName);
  const candidates = [
    selectedModel,
    preferredHfTextModel,
    "google/gemma-3-12b-it",
    "Qwen/Qwen2.5-7B-Instruct"
  ].filter((name) => typeof name === "string" && name.trim());

  return [...new Set(candidates)];
}

function toHuggingFaceMessages(history, maxHistoryTurns) {
  return history
    .filter((message) => message && message.type === "text" && typeof message.text === "string" && message.text.trim())
    .slice(-Math.max(1, maxHistoryTurns) * 2)
    .map((message) => ({
      role: message.role === "assistant" ? "assistant" : "user",
      content: message.text
    }));
}

function buildHuggingFaceAttachmentContext(files) {
  if (!Array.isArray(files) || files.length === 0) {
    return "";
  }

  const sections = [];

  files.forEach((file) => {
    if (isTextLikeFile(file)) {
      const rawText = file.buffer.toString("utf8");
      const maxChars = 12000;
      const clipped = rawText.length > maxChars
        ? `${rawText.slice(0, maxChars)}\n\n[File truncated due to size]`
        : rawText;

      sections.push(`Attached file (${file.originalname}):\n${clipped}`);
      return;
    }

    if (isImageFile(file)) {
      sections.push(
        `Attached image (${file.originalname}, ${file.mimetype || "image"}) is present. If image understanding is required, choose a vision-capable model.`
      );
      return;
    }

    sections.push(`Attached file (${file.originalname}) could not be parsed in this model path.`);
  });

  return sections.join("\n\n");
}

async function generateHuggingFaceImage(prompt, modelName) {
  if (!huggingFaceToken) {
    const tokenError = new Error("HF_TOKEN is missing. Add it to your .env file to use Hugging Face image models.");
    tokenError.status = 400;
    throw tokenError;
  }

  const initialModel = getHfModelId(modelName);
  if (!initialModel) {
    const modelError = new Error("Invalid Hugging Face model format. Use hf:org/model.");
    modelError.status = 400;
    throw modelError;
  }

  const modelCandidates = getHuggingFaceModelCandidates(modelName);
  const errors = [];

  for (const hfModelId of modelCandidates) {
    const requestPlans = [
      {
        url: `https://router.huggingface.co/${encodeURIComponent(huggingFaceProvider)}/v1/images/generations`,
        body: {
          model: hfModelId,
          prompt
        }
      },
      {
        url: `https://router.huggingface.co/hf-inference/models/${encodeURIComponent(hfModelId)}`,
        body: {
          inputs: prompt,
          options: {
            wait_for_model: true
          }
        }
      }
    ];

    const uniquePlans = requestPlans.filter((plan, index, arr) => arr.findIndex((p) => p.url === plan.url) === index);

    for (const plan of uniquePlans) {
      const response = await fetch(plan.url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${huggingFaceToken}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(plan.body)
      });

      if (!response.ok) {
        const errorBody = await response.text();
        const hfError = new Error(errorBody || "Hugging Face image request failed.");
        hfError.status = response.status;
        errors.push(hfError);
        continue;
      }

      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        const payload = await response.json();
        const first = payload && Array.isArray(payload.data) ? payload.data[0] : null;
        const b64 = first && first.b64_json ? first.b64_json : null;

        if (!b64) {
          const parseError = new Error("Hugging Face returned JSON but no image data was found.");
          parseError.status = 502;
          errors.push(parseError);
          continue;
        }

        return {
          mimeType: "image/png",
          data: b64,
          modelUsed: hfModelId
        };
      }

      const imageBuffer = Buffer.from(await response.arrayBuffer());
      const finalMime = contentType || "image/png";

      return {
        mimeType: finalMime,
        data: imageBuffer.toString("base64"),
        modelUsed: hfModelId
      };
    }
  }

  throw errors[errors.length - 1] || new Error("Hugging Face image request failed.");
}

async function generateHuggingFaceText(history, modelName, options, attachments) {
  if (!huggingFaceToken) {
    const tokenError = new Error("HF_TOKEN is missing. Add it to your .env file to use Hugging Face models.");
    tokenError.status = 400;
    throw tokenError;
  }

  const modelCandidates = getHuggingFaceTextModelCandidates(modelName);
  const errors = [];

  for (const hfModelId of modelCandidates) {
    const messages = toHuggingFaceMessages(history, options.maxHistoryTurns);
    const attachmentContext = buildHuggingFaceAttachmentContext(attachments);

    if (attachmentContext) {
      let lastUserIndex = -1;
      for (let i = messages.length - 1; i >= 0; i -= 1) {
        if (messages[i].role === "user") {
          lastUserIndex = i;
          break;
        }
      }

      if (lastUserIndex >= 0) {
        messages[lastUserIndex].content = `${messages[lastUserIndex].content}\n\nAttachment context:\n${attachmentContext}`;
      } else {
        messages.push({
          role: "user",
          content: `Attachment context:\n${attachmentContext}`
        });
      }
    }

    const lastUser = [...messages].reverse().find((msg) => msg.role === "user");
    const latestPrompt = lastUser ? lastUser.content : "";

    const requestPlans = [
      {
        url: `https://router.huggingface.co/${encodeURIComponent(huggingFaceTextProvider)}/v1/chat/completions`,
        body: {
          model: hfModelId,
          messages,
          temperature: options.temperature,
          top_p: options.topP,
          max_tokens: options.maxOutputTokens
        },
        parser: (payload) => {
          if (!payload || !Array.isArray(payload.choices) || !payload.choices[0] || !payload.choices[0].message) {
            return "";
          }

          const content = payload.choices[0].message.content;
          if (typeof content === "string") {
            return content.trim();
          }

          if (Array.isArray(content)) {
            return content.map((part) => (part && part.text ? part.text : "")).join("\n").trim();
          }

          return "";
        }
      },
      {
        url: `https://router.huggingface.co/v1/chat/completions`,
        body: {
          model: hfModelId,
          messages,
          temperature: options.temperature,
          top_p: options.topP,
          max_tokens: options.maxOutputTokens
        },
        parser: (payload) => {
          if (!payload || !Array.isArray(payload.choices) || !payload.choices[0] || !payload.choices[0].message) {
            return "";
          }

          const content = payload.choices[0].message.content;
          return typeof content === "string" ? content.trim() : "";
        }
      },
      {
        url: `https://router.huggingface.co/hf-inference/models/${encodeURIComponent(hfModelId)}`,
        body: {
          inputs: latestPrompt,
          parameters: {
            max_new_tokens: options.maxOutputTokens,
            temperature: options.temperature,
            top_p: options.topP,
            return_full_text: false
          },
          options: {
            wait_for_model: true
          }
        },
        parser: (payload) => {
          if (Array.isArray(payload) && payload[0] && payload[0].generated_text) {
            return String(payload[0].generated_text).trim();
          }
          if (payload && payload.generated_text) {
            return String(payload.generated_text).trim();
          }
          return "";
        }
      }
    ];

    for (const plan of requestPlans) {
      const response = await fetch(plan.url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${huggingFaceToken}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(plan.body)
      });

      if (!response.ok) {
        const errorBody = await response.text();
        const hfError = new Error(errorBody || "Hugging Face text request failed.");
        hfError.status = response.status;
        errors.push(hfError);
        continue;
      }

      const payload = await response.json();
      const text = plan.parser(payload);
      if (text) {
        return {
          text,
          modelUsed: hfModelId
        };
      }
    }
  }

  throw errors[errors.length - 1] || new Error("No compatible Hugging Face text model returned output.");
}

async function transcribeWithHuggingFace(fileBuffer, mimeType, modelName) {
  if (!huggingFaceToken) {
    const tokenError = new Error("HF_TOKEN is missing. Add it to your .env file to use Hugging Face STT models.");
    tokenError.status = 400;
    throw tokenError;
  }

  const modelId = getHfModelId(modelName) || preferredHfSttModel;
  const response = await fetch(`https://router.huggingface.co/hf-inference/models/${encodeURIComponent(modelId)}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${huggingFaceToken}`,
      "Content-Type": mimeType || "audio/webm"
    },
    body: fileBuffer
  });

  if (!response.ok) {
    const raw = await response.text();
    const error = new Error(raw || "Hugging Face STT request failed.");
    error.status = response.status;
    throw error;
  }

  const payload = await response.json();
  if (payload && typeof payload.text === "string") {
    return payload.text.trim();
  }

  if (Array.isArray(payload) && payload[0] && typeof payload[0].text === "string") {
    return payload[0].text.trim();
  }

  return "";
}

async function synthesizeWithHuggingFace(text, modelName) {
  if (!huggingFaceToken) {
    const tokenError = new Error("HF_TOKEN is missing. Add it to your .env file to use Hugging Face TTS models.");
    tokenError.status = 400;
    throw tokenError;
  }

  const errors = [];
  const modelCandidates = getHuggingFaceTtsModelCandidates(modelName);

  for (const modelId of modelCandidates) {
    const modelPathVariants = [encodeURIComponent(modelId), modelId].filter((value, index, arr) => arr.indexOf(value) === index);
    const requestPlans = [
      ...(huggingFaceTtsEndpoint
        ? [{
          url: huggingFaceTtsEndpoint,
          body: {
            inputs: text,
            model: modelId,
            options: {
              wait_for_model: true
            }
          }
        }]
        : []),
      ...modelPathVariants.map((modelPath) => ({
        url: `https://router.huggingface.co/hf-inference/models/${modelPath}`,
        body: {
          inputs: text,
          options: {
            wait_for_model: true
          }
        }
      }))
    ];

    const uniquePlans = requestPlans.filter((plan, index, arr) => arr.findIndex((p) => p.url === plan.url) === index);

    for (const plan of uniquePlans) {
      const response = await fetch(plan.url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${huggingFaceToken}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(plan.body)
      });

      if (!response.ok) {
        const raw = await response.text();
        const error = new Error(raw || "Hugging Face TTS request failed.");
        error.status = response.status;
        errors.push(error);
        continue;
      }

      const contentType = response.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        const payload = await response.json();
        const b64 = payload && (
          payload.audio
          || payload.audio_base64
          || (payload.data && payload.data.audio)
          || (Array.isArray(payload) && payload[0] && payload[0].audio)
        );

        if (!b64) {
          const providerError = payload && payload.error ? String(payload.error) : "";
          if (providerError) {
            const error = new Error(providerError);
            error.status = 502;
            errors.push(error);
            continue;
          }

          const parseError = new Error("Hugging Face TTS returned JSON but no audio payload was found.");
          parseError.status = 502;
          errors.push(parseError);
          continue;
        }

        return {
          mimeType: "audio/mpeg",
          audioBuffer: Buffer.from(b64, "base64")
        };
      }

      const audioBuffer = Buffer.from(await response.arrayBuffer());
      return {
        mimeType: contentType || "audio/mpeg",
        audioBuffer
      };
    }
  }

  const hasNotFound = errors.some((error) => {
    const msg = error && error.message ? error.message.toLowerCase() : "";
    return (error && error.status === 404) || msg.includes("not found") || msg.includes("could not find model");
  });

  if (hasNotFound) {
    const debugErrors = errors
      .slice(-3)
      .map((error) => {
        const status = error && error.status ? error.status : "unknown";
        const message = error && error.message ? String(error.message).replace(/\s+/g, " ").trim() : "unknown error";
        return `status=${status}: ${message.slice(0, 220)}`;
      })
      .join(" | ");
    const attempted = modelCandidates.join(", ");
    const notFoundError = new Error(
      `No compatible Hugging Face TTS model endpoint was found for this token/provider. Tried models: ${attempted}. Recent provider responses: ${debugErrors}`
    );
    notFoundError.status = 404;
    throw notFoundError;
  }

  throw errors[errors.length - 1] || new Error("Hugging Face TTS request failed.");
}

async function generateImageWithFallback(prompt, chosenModel) {
  const modelCandidates = getImageModelCandidates(chosenModel);
  const errors = [];

  for (const modelName of modelCandidates) {
    try {
      const imageResult = await imageAI.models.generateContent({
        model: modelName,
        contents: prompt
      });

      const imagePart = getInlineImagePart(imageResult);
      if (imagePart) {
        return {
          imagePart,
          modelUsed: modelName
        };
      }
    } catch (error) {
      errors.push(error);
    }
  }

  const hasQuotaError = errors.some((error) => {
    const msg = error && error.message ? error.message : "";
    return msg.includes("RESOURCE_EXHAUSTED") || msg.includes('"code":429') || msg.includes("quota");
  });

  if (hasQuotaError) {
    const quotaError = new Error(
      "Image generation is blocked by current quota limits for your key/project (free tier may have 0 image quota)."
    );
    quotaError.status = 429;
    throw quotaError;
  }

  const hasNotFoundError = errors.some((error) => {
    const msg = error && error.message ? error.message : "";
    return msg.includes("not found for API version") || msg.includes('"status":"NOT_FOUND"');
  });

  if (hasNotFoundError) {
    const notFoundError = new Error(
      "No compatible image model is available for your API version/key in this project."
    );
    notFoundError.status = 404;
    throw notFoundError;
  }

  throw errors[errors.length - 1] || new Error("No compatible image model returned image data.");
}

app.post("/api/message", upload.array("attachments", 6), async (req, res) => {
  let imageModeRequested = false;
  let hfModeRequested = false;

  try {
    const isMultipart = (req.headers["content-type"] || "").includes("multipart/form-data");
    const body = req.body || {};
    const model = isMultipart ? body.model : body.model;
    const history = isMultipart ? parseJsonField(body.history, []) : body.history;
    const settings = isMultipart ? parseJsonField(body.settings, {}) : body.settings;
    const attachments = Array.isArray(req.files) ? req.files : [];

    if (!Array.isArray(history) || history.length === 0) {
      return res.status(400).json({ error: "History is required." });
    }

    const chosenModel = typeof model === "string" && model.trim() ? model.trim() : "gemini-2.5-flash";

    const latestUserMessage = [...history].reverse().find((message) => message.role === "user" && message.type === "text" && message.text);

    if (!latestUserMessage) {
      return res.status(400).json({ error: "A user text message is required." });
    }

    const temperature = clamp(Number(settings && settings.temperature), 0, 1, defaultTemperature);
    const topP = clamp(Number(settings && settings.topP), 0, 1, defaultTopP);
    const topK = clamp(Number(settings && settings.topK), 1, 100, defaultTopK);
    const maxOutputTokens = clamp(Number(settings && settings.maxOutputTokens), 64, 768, defaultMaxOutputTokens);
    const maxHistoryTurns = clamp(Number(settings && settings.maxHistoryTurns), 2, 30, defaultHistoryTurns);
    const hardCharLimit = clamp(Number(settings && settings.hardCharLimit), 180, 2000, defaultHardCharLimit);
    const isHfImageModel = isHuggingFaceImageModel(chosenModel);
    const isHfTextModel = isHuggingFaceTextModel(chosenModel);
    const isImageModel = chosenModel.includes("image");
    hfModeRequested = isHfImageModel || isHfTextModel;
    imageModeRequested = isImageModel || isHfImageModel;

    if (isHfImageModel) {
      const imageResponse = await generateHuggingFaceImage(latestUserMessage.text, chosenModel);

      return res.json({
        type: "image",
        imageUrl: `data:${imageResponse.mimeType};base64,${imageResponse.data}`,
        modelUsed: imageResponse.modelUsed
      });
    }

    if (isImageModel) {
      const imageResponse = await generateImageWithFallback(latestUserMessage.text, chosenModel);

      return res.json({
        type: "image",
        imageUrl: `data:${imageResponse.imagePart.mimeType};base64,${imageResponse.imagePart.data}`,
        modelUsed: imageResponse.modelUsed
      });
    }

    if (isHfTextModel) {
      const textResponse = await generateHuggingFaceText(history, chosenModel, {
        temperature,
        topP,
        maxOutputTokens,
        maxHistoryTurns
      }, attachments);

      return res.json({
        type: "text",
        text: enforceCharLimit(textResponse.text, hardCharLimit) || "I could not generate a response.",
        modelUsed: textResponse.modelUsed
      });
    }

    const modelClient = genAI.getGenerativeModel({
      model: chosenModel,
      systemInstruction:
        "You are a precise assistant. Keep responses short by default (max 6 sentences). Only provide long answers when the user explicitly asks. If unsure, state uncertainty and ask a clarifying question. Do not fabricate facts, links, or citations.",
      generationConfig: {
        temperature,
        topP,
        topK,
        maxOutputTokens
      }
    });

    const contents = toGeminiContents(history, maxHistoryTurns);
    attachFilesToLatestUserTurn(contents, attachments);
    const result = await modelClient.generateContent({ contents });
    const text = result && result.response ? result.response.text().trim() : "";
    const boundedText = enforceCharLimit(text, hardCharLimit);

    return res.json({
      type: "text",
      text: boundedText || "I could not generate a response."
    });
  } catch (error) {
    const status = error && error.status ? error.status : 500;
    const rawMessage = error && error.message ? error.message : "Unexpected server error.";
    let message = rawMessage;

    if (imageModeRequested && (status === 429 || rawMessage.includes("RESOURCE_EXHAUSTED") || rawMessage.includes("quota"))) {
      message =
        "Image generation quota is exhausted or unavailable for this key/project. Enable billing or use a key/project with image quota, then retry.";
    } else if (hfModeRequested && status === 403 && rawMessage.includes("sufficient permissions")) {
      message =
        "Your HF_TOKEN lacks Inference Providers permission for this account/provider. Create a token with provider access, then retry.";
    } else if (imageModeRequested && rawMessage.toLowerCase().includes("deprecated")) {
      message =
        "The selected HF model is deprecated for this provider. Switched fallback models automatically; set HF_IMAGE_MODEL to a currently supported model (for example black-forest-labs/FLUX.1-schnell).";
    } else if (imageModeRequested && status === 405 && rawMessage.includes("Not allowed to POST /v1/images/generations")) {
      message =
        "This provider does not support /v1/images/generations for your token. The server now retries with hf-inference automatically; if this persists, switch HF_PROVIDER=hf-inference in .env.";
    } else if (hfModeRequested && status === 400 && rawMessage.includes("HF_TOKEN is missing")) {
      message = "HF_TOKEN is missing. Add your Hugging Face token to .env before using HF models.";
    } else if (status === 429 || rawMessage.includes("RESOURCE_EXHAUSTED") || rawMessage.includes("quota")) {
      message =
        "Quota is exhausted or unavailable for this key/project. Wait and retry, or enable billing/use a project with available quota.";
    } else if (imageModeRequested && (rawMessage.includes("is not found for API version") || rawMessage.includes("No compatible image model"))) {
      message =
        "Selected image model is unavailable for your API version/key. Set GEMINI_IMAGE_MODEL in .env to an image-capable model exposed in ListModels.";
    }

    return res.status(status).json({ error: message });
  }
});

app.post("/api/stt", upload.single("audio"), async (req, res) => {
  try {
    if (!req.file || !req.file.buffer) {
      return res.status(400).json({ error: "Audio file is required." });
    }

    const model = typeof req.body.sttModel === "string" && req.body.sttModel.trim()
      ? req.body.sttModel.trim()
      : "gpt-4o-mini-transcribe";

    if (isHuggingFaceSttModel(model)) {
      const text = await transcribeWithHuggingFace(req.file.buffer, req.file.mimetype, model);
      return res.json({ text });
    }

    if (!openai) {
      return res.status(501).json({ error: "OPENAI_API_KEY is required for STT." });
    }

    const audioFile = await toFile(req.file.buffer, req.file.originalname || "recording.webm", {
      type: req.file.mimetype || "audio/webm"
    });

    const transcript = await openai.audio.transcriptions.create({
      model,
      file: audioFile
    });

    return res.json({ text: transcript.text || "" });
  } catch (error) {
    const status = error && error.status ? error.status : 500;
    const message = error && error.message ? error.message : "STT request failed.";
    return res.status(status).json({ error: message });
  }
});

app.post("/api/tts", express.json({ limit: "1mb" }), async (req, res) => {
  try {
    const text = typeof req.body.text === "string" ? req.body.text.trim() : "";
    if (!text) {
      return res.status(400).json({ error: "Text is required for TTS." });
    }

    const model = typeof req.body.model === "string" && req.body.model.trim()
      ? req.body.model.trim()
      : "gpt-4o-mini-tts";
    const voice = typeof req.body.voice === "string" && req.body.voice.trim()
      ? req.body.voice.trim()
      : "alloy";

    if (isHuggingFaceTtsModel(model)) {
      const hfSpeech = await synthesizeWithHuggingFace(text, model);
      res.setHeader("Content-Type", hfSpeech.mimeType);
      return res.send(hfSpeech.audioBuffer);
    }

    if (!openai) {
      return res.status(501).json({ error: "OPENAI_API_KEY is required for TTS." });
    }

    const speech = await openai.audio.speech.create({
      model,
      voice,
      input: text,
      format: "mp3"
    });

    const audioBuffer = Buffer.from(await speech.arrayBuffer());
    res.setHeader("Content-Type", "audio/mpeg");
    return res.send(audioBuffer);
  } catch (error) {
    const status = error && error.status ? error.status : 500;
    const message = error && error.message ? error.message : "TTS request failed.";
    return res.status(status).json({ error: message });
  }
});

app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "index.html"));
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
