const messagesEl = document.getElementById("messages");
const composerEl = document.getElementById("composer");
const inputEl = document.getElementById("prompt-input");
const sidebarNewChatBtn = document.getElementById("sidebar-new-chat");
const headerNewChatBtn = document.getElementById("header-new-chat");
const resetHistoryBtn = document.getElementById("reset-history-btn");
const chatListEl = document.getElementById("chat-list");
const modelSelectEl = document.getElementById("model-select");
const attachmentInputEl = document.getElementById("attachment-input");
const attachmentListEl = document.getElementById("attachment-list");
const micBtn = document.getElementById("mic-btn");
const readLastBtn = document.getElementById("read-last-btn");
const sttModelSelectEl = document.getElementById("stt-model-select");
const ttsModelSelectEl = document.getElementById("tts-model-select");
const ttsVoiceSelectEl = document.getElementById("tts-voice-select");

const STORAGE_KEY = "chatgpt-ui-clone-state-v1";

let state = loadState();
let isSending = false;
let thinkingIndicatorEl = null;
let pendingFiles = [];
let mediaRecorder = null;
let recorderStream = null;
let audioChunks = [];
let isRecording = false;
let isSpeaking = false;
let currentAudio = null;
let currentAudioUrl = "";
let currentSpeechUtterance = null;
let resolveActivePlayback = null;
let isCreatingChat = false;
let editingChatId = "";

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function uid() {
  return `${Date.now()}-${Math.floor(Math.random() * 100000)}`;
}

function createEmptyChat(title) {
  return {
    id: uid(),
    title,
    messages: []
  };
}

function createInitialState() {
  const initialState = {
    chats: [createEmptyChat("New chat")],
    activeChatId: "",
    chatCount: 1
  };

  initialState.activeChatId = initialState.chats[0].id;
  return initialState;
}

function applyTheme() {
  document.documentElement.setAttribute("data-theme", "dark");
}

function loadState() {
  const fallback = createInitialState();

  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return fallback;
    }

    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.chats) || parsed.chats.length === 0) {
      return fallback;
    }

    const activeExists = parsed.chats.some((chat) => chat.id === parsed.activeChatId);
    return {
      chats: parsed.chats,
      activeChatId: activeExists ? parsed.activeChatId : parsed.chats[0].id,
      chatCount: Number.isInteger(parsed.chatCount) ? parsed.chatCount : parsed.chats.length
    };
  } catch (error) {
    return fallback;
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function getActiveChat() {
  return state.chats.find((chat) => chat.id === state.activeChatId) || state.chats[0];
}

function suggestChatTitleFromText(text) {
  if (typeof text !== "string") {
    return "New chat";
  }

  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned) {
    return "New chat";
  }

  const maxLen = 42;
  if (cleaned.length <= maxLen) {
    return cleaned;
  }

  return `${cleaned.slice(0, maxLen - 3).trimEnd()}...`;
}

function shouldAutoTitleChat(chat) {
  if (!chat || typeof chat.title !== "string") {
    return false;
  }

  const title = chat.title.trim().toLowerCase();
  return title === "new chat" || /^chat\s+\d+$/.test(title);
}

function setActiveToFirstChatIfNeeded() {
  if (!state.chats.some((chat) => chat.id === state.activeChatId)) {
    state.activeChatId = state.chats[0] ? state.chats[0].id : "";
  }
}

function startInlineRename(chatId) {
  editingChatId = chatId;
  renderChatList();
}

function commitInlineRename(chatId, nextTitle) {
  const chat = state.chats.find((entry) => entry.id === chatId);
  if (!chat) {
    editingChatId = "";
    renderChatList();
    return;
  }

  const cleaned = typeof nextTitle === "string" ? nextTitle.trim() : "";
  if (cleaned) {
    chat.title = cleaned;
    saveState();
  }

  editingChatId = "";
  renderChatList();
}

function deleteChatById(chatId) {
  if (state.chats.length <= 1) {
    return;
  }

  const target = state.chats.find((chat) => chat.id === chatId);
  if (!target) {
    return;
  }

  state.chats = state.chats.filter((chat) => chat.id !== chatId);
  if (editingChatId === chatId) {
    editingChatId = "";
  }
  setActiveToFirstChatIfNeeded();
  saveState();
  render();
}

function renderChatList(popChatId = "") {
  chatListEl.innerHTML = "";

  state.chats.forEach((chat) => {
    const item = document.createElement("li");
    item.className = "chat-item";
    item.dataset.chatId = chat.id;

    const isEditing = editingChatId === chat.id;
    if (isEditing) {
      item.classList.add("editing");
    }

    const titleWrap = document.createElement("div");
    titleWrap.className = "chat-item-title-wrap";

    if (isEditing) {
      const titleInput = document.createElement("input");
      titleInput.type = "text";
      titleInput.className = "chat-title-input";
      titleInput.value = chat.title;
      titleInput.maxLength = 80;

      titleInput.addEventListener("click", (event) => {
        event.stopPropagation();
      });

      titleInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          commitInlineRename(chat.id, titleInput.value);
          return;
        }

        if (event.key === "Escape") {
          event.preventDefault();
          editingChatId = "";
          renderChatList();
        }
      });

      titleInput.addEventListener("blur", () => {
        commitInlineRename(chat.id, titleInput.value);
      });

      titleWrap.appendChild(titleInput);

      // Let the input mount before focusing to avoid layout jitter.
      window.requestAnimationFrame(() => {
        titleInput.focus();
        titleInput.select();
      });
    } else {
      const titleEl = document.createElement("span");
      titleEl.className = "chat-item-title";
      titleEl.textContent = chat.title;
      titleWrap.appendChild(titleEl);
    }

    const actionsEl = document.createElement("div");
    actionsEl.className = "chat-item-actions";

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.className = "chat-action-btn icon";
    renameBtn.title = "Rename";
    renameBtn.setAttribute("aria-label", "Rename chat");
    renameBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20h4.2L19 9.2 14.8 5 4 15.8V20zm13.7-12.1 1.4-1.4a1 1 0 0 0 0-1.4l-1.3-1.3a1 1 0 0 0-1.4 0L15 5.2l2.7 2.7z"/></svg>';
    renameBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      startInlineRename(chat.id);
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "chat-action-btn icon danger";
    deleteBtn.title = "Delete";
    deleteBtn.setAttribute("aria-label", "Delete chat");
    deleteBtn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6l1 2h4v2H4V5h4l1-2zm1 6h2v9h-2V9zm4 0h2v9h-2V9zM7 9h2v9H7V9z"/></svg>';
    deleteBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteChatById(chat.id);
    });

    actionsEl.appendChild(renameBtn);
    actionsEl.appendChild(deleteBtn);
    item.appendChild(titleWrap);
    item.appendChild(actionsEl);

    if (chat.id === state.activeChatId) {
      item.classList.add("active");
    }

    if (chat.id === popChatId) {
      item.classList.add("pop-in");
    }

    item.addEventListener("click", () => {
      state.activeChatId = chat.id;
      saveState();
      render(true);
    });

    chatListEl.appendChild(item);
  });
}

async function animateMessagesClearOut() {
  if (messagesEl.children.length === 0) {
    return;
  }

  messagesEl.classList.add("clearing");
  await wait(180);
  messagesEl.classList.remove("clearing");
}

function renderMessage(message, options = {}) {
  const { stagger = false, index = 0 } = options;
  const bubble = document.createElement("div");
  bubble.className = `msg ${message.role}`;

  if (stagger) {
    bubble.classList.add("stagger");
    bubble.style.setProperty("--stagger-delay", `${Math.min(index, 12) * 36}ms`);
  }

  if (message.type === "image" && message.imageUrl) {
    const img = document.createElement("img");
    img.src = message.imageUrl;
    img.alt = "Generated image";
    img.className = "generated-image";
    bubble.appendChild(img);
  } else {
    bubble.textContent = message.text || "";
  }

  if (Array.isArray(message.attachments) && message.attachments.length > 0) {
    const attachmentsEl = document.createElement("div");
    attachmentsEl.className = "msg-attachments";

    message.attachments.forEach((file) => {
      const chip = document.createElement("span");
      chip.className = "attachment-chip";
      chip.textContent = file && file.name ? file.name : "attachment";
      attachmentsEl.appendChild(chip);
    });

    bubble.appendChild(attachmentsEl);
  }

  messagesEl.appendChild(bubble);
}

function toFileMeta(file) {
  return {
    name: file.name,
    type: file.type || "application/octet-stream",
    size: file.size || 0
  };
}

function clearPendingFiles() {
  pendingFiles = [];
  attachmentInputEl.value = "";
  renderPendingFiles();
}

function removePendingFile(index) {
  pendingFiles = pendingFiles.filter((_, idx) => idx !== index);
  renderPendingFiles();
}

function renderPendingFiles() {
  attachmentListEl.innerHTML = "";

  pendingFiles.forEach((file, index) => {
    const chip = document.createElement("span");
    chip.className = "pending-chip";
    chip.textContent = file.name;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.textContent = "x";
    removeBtn.addEventListener("click", () => removePendingFile(index));

    chip.appendChild(removeBtn);
    attachmentListEl.appendChild(chip);
  });
}

function handleAttachmentSelect() {
  const selected = Array.from(attachmentInputEl.files || []);
  pendingFiles = selected.slice(0, 6);
  renderPendingFiles();
}

function renderMessages(stagger = false) {
  const activeChat = getActiveChat();
  messagesEl.innerHTML = "";
  thinkingIndicatorEl = null;

  activeChat.messages.forEach((message, index) => {
    renderMessage(message, { stagger, index });
  });

  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showThinkingIndicator() {
  if (thinkingIndicatorEl) {
    return;
  }

  const bubble = document.createElement("div");
  bubble.className = "msg assistant thinking";
  bubble.setAttribute("aria-live", "polite");
  bubble.innerHTML =
    '<span class="thinking-label">Thinking</span><span class="thinking-dots"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>';

  messagesEl.appendChild(bubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  thinkingIndicatorEl = bubble;
}

function hideThinkingIndicator() {
  if (!thinkingIndicatorEl) {
    return;
  }

  thinkingIndicatorEl.remove();
  thinkingIndicatorEl = null;
}

function setSendingStatus(sending) {
  isSending = sending;
  inputEl.disabled = sending;
  modelSelectEl.disabled = sending;
  attachmentInputEl.disabled = sending;
  micBtn.disabled = sending;
  readLastBtn.disabled = sending;
  readLastBtn.textContent = isSpeaking ? "Stop Reading" : "Read Reply";
  const sendBtn = composerEl.querySelector(".send-btn");
  sendBtn.disabled = sending;
  sendBtn.textContent = sending ? "Sending..." : "Send";
}

function getLastAssistantText() {
  const activeChat = getActiveChat();
  const match = [...activeChat.messages]
    .reverse()
    .find((message) => message.role === "assistant" && message.type === "text" && message.text);

  return match ? match.text : "";
}

function stopRecorderStream() {
  if (!recorderStream) {
    return;
  }

  recorderStream.getTracks().forEach((track) => track.stop());
  recorderStream = null;
}

function speakWithBrowserTts(text) {
  if (!("speechSynthesis" in window) || typeof SpeechSynthesisUtterance === "undefined") {
    throw new Error("Browser TTS is not supported in this browser.");
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  utterance.volume = 1;
  currentSpeechUtterance = utterance;

  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanup = () => {
      currentSpeechUtterance = null;
      resolveActivePlayback = null;
    };

    const finish = () => {
      if (settled) {
        return;
      }

      settled = true;
      cleanup();
      resolve();
    };

    const fail = () => {
      if (settled) {
        return;
      }

      settled = true;
      cleanup();
      reject(new Error("Browser TTS playback failed."));
    };

    resolveActivePlayback = finish;
    utterance.addEventListener("end", finish, { once: true });
    utterance.addEventListener("error", fail, { once: true });
    window.speechSynthesis.speak(utterance);
  });
}

function stopReadingAloud() {
  if (currentSpeechUtterance && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    currentSpeechUtterance = null;
  }

  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }

  if (currentAudioUrl) {
    URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = "";
  }

  if (typeof resolveActivePlayback === "function") {
    resolveActivePlayback();
  }

  resolveActivePlayback = null;
  isSpeaking = false;
  readLastBtn.disabled = isSending;
  readLastBtn.textContent = "Read Reply";
}

async function transcribeRecordedAudio(blob) {
  const formData = new FormData();
  formData.append("audio", blob, "recording.webm");
  formData.append("sttModel", sttModelSelectEl.value);

  const response = await fetch("/api/stt", {
    method: "POST",
    body: formData
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "STT failed.");
  }

  const text = typeof data.text === "string" ? data.text.trim() : "";
  if (!text) {
    return;
  }

  inputEl.value = inputEl.value.trim() ? `${inputEl.value.trim()} ${text}` : text;
  inputEl.focus();
}

async function handleMicToggle() {
  if (isSending) {
    return;
  }

  if (isRecording && mediaRecorder) {
    mediaRecorder.stop();
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    addMessageToActiveChat({
      role: "assistant",
      type: "text",
      text: "Error: Microphone capture is not supported in this browser."
    });
    return;
  }

  try {
    recorderStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(recorderStream);

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener("stop", async () => {
      const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      audioChunks = [];
      isRecording = false;
      micBtn.textContent = "Mic";
      micBtn.classList.remove("recording");
      stopRecorderStream();

      try {
        await transcribeRecordedAudio(blob);
      } catch (error) {
        addMessageToActiveChat({
          role: "assistant",
          type: "text",
          text: `Error: ${error.message}`
        });
      }
    });

    mediaRecorder.start();
    isRecording = true;
    micBtn.textContent = "Stop";
    micBtn.classList.add("recording");
  } catch (error) {
    stopRecorderStream();
    addMessageToActiveChat({
      role: "assistant",
      type: "text",
      text: "Error: Could not access microphone."
    });
  }
}

async function handleReadLastReply() {
  if (isSending) {
    return;
  }

  if (isSpeaking) {
    stopReadingAloud();
    return;
  }

  const text = getLastAssistantText();
  if (!text) {
    addMessageToActiveChat({
      role: "assistant",
      type: "text",
      text: "Error: No assistant reply available to read."
    });
    return;
  }

  isSpeaking = true;
  readLastBtn.disabled = false;
  readLastBtn.textContent = "Stop Reading";

  try {
    if (ttsModelSelectEl.value === "browser-tts") {
      await speakWithBrowserTts(text);
      return;
    }

    const response = await fetch("/api/tts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text,
        model: ttsModelSelectEl.value,
        voice: ttsVoiceSelectEl.value
      })
    });

    if (!response.ok) {
      let errorMessage = "TTS failed.";
      const contentType = response.headers.get("content-type") || "";

      if (contentType.includes("application/json")) {
        const data = await response.json();
        errorMessage = data && data.error ? data.error : errorMessage;
      } else {
        const raw = await response.text();
        if (raw) {
          errorMessage = raw;
        }
      }

      throw new Error(errorMessage);
    }

    const audioBlob = await response.blob();
    currentAudioUrl = URL.createObjectURL(audioBlob);
    currentAudio = new Audio(currentAudioUrl);

    await new Promise((resolve, reject) => {
      let settled = false;

      const cleanup = () => {
        if (currentAudioUrl) {
          URL.revokeObjectURL(currentAudioUrl);
          currentAudioUrl = "";
        }
        currentAudio = null;
        resolveActivePlayback = null;
      };

      const finish = () => {
        if (settled) {
          return;
        }

        settled = true;
        cleanup();
        resolve();
      };

      const fail = () => {
        if (settled) {
          return;
        }

        settled = true;
        cleanup();
        reject(new Error("Audio playback failed."));
      };

      resolveActivePlayback = finish;
      currentAudio.addEventListener("ended", finish, { once: true });
      currentAudio.addEventListener("error", fail, { once: true });
      currentAudio.play().catch(fail);
    });
  } catch (error) {
    try {
      await speakWithBrowserTts(text);
      addMessageToActiveChat({
        role: "assistant",
        type: "text",
        text: `Notice: API TTS failed (${error.message}), browser TTS was used instead.`
      });
    } catch (fallbackError) {
      addMessageToActiveChat({
        role: "assistant",
        type: "text",
        text: `Error: ${error.message}`
      });
    }
  } finally {
    isSpeaking = false;
    readLastBtn.disabled = isSending;
    readLastBtn.textContent = "Read Reply";
  }
}

function addMessageToActiveChat(message) {
  const activeChat = getActiveChat();
  activeChat.messages.push(message);
  saveState();
  renderMessages(false);
}

async function requestAssistantReply(filesForRequest) {
  const activeChat = getActiveChat();
  let response;

  if (filesForRequest.length > 0) {
    const formData = new FormData();
    formData.append("model", modelSelectEl.value);
    formData.append("history", JSON.stringify(activeChat.messages));

    filesForRequest.forEach((file) => {
      formData.append("attachments", file);
    });

    response = await fetch("/api/message", {
      method: "POST",
      body: formData
    });
  } else {
    response = await fetch("/api/message", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: modelSelectEl.value,
        history: activeChat.messages
      })
    });
  }

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Request failed.");
  }

  if (data.type === "image" && data.imageUrl) {
    addMessageToActiveChat({
      role: "assistant",
      type: "image",
      imageUrl: data.imageUrl
    });
    return;
  }

  addMessageToActiveChat({
    role: "assistant",
    type: "text",
    text: data.text || "I could not generate a response."
  });
}

async function handleSend(event) {
  event.preventDefault();
  if (isSending) {
    return;
  }

  const activeChat = getActiveChat();
  const prompt = inputEl.value.trim();
  const filesForRequest = [...pendingFiles];

  if (!prompt && filesForRequest.length === 0) {
    return;
  }

  const userText = prompt || "Attached files";

  if (activeChat && activeChat.messages.length === 0 && shouldAutoTitleChat(activeChat)) {
    activeChat.title = suggestChatTitleFromText(userText);
    saveState();
    renderChatList();
  }

  addMessageToActiveChat({
    role: "user",
    type: "text",
    text: userText,
    attachments: filesForRequest.map(toFileMeta)
  });

  inputEl.value = "";
  clearPendingFiles();
  setSendingStatus(true);
  showThinkingIndicator();

  try {
    await requestAssistantReply(filesForRequest);
  } catch (error) {
    addMessageToActiveChat({
      role: "assistant",
      type: "text",
      text: `Error: ${error.message}`
    });
  } finally {
    hideThinkingIndicator();
    setSendingStatus(false);
    inputEl.focus();
  }
}

async function handleNewChat() {
  if (isCreatingChat) {
    return;
  }

  isCreatingChat = true;

  try {
    await animateMessagesClearOut();

    state.chatCount += 1;
    const chat = createEmptyChat("New chat");
    state.chats.unshift(chat);
    state.activeChatId = chat.id;
    saveState();
    render(false, { popChatId: chat.id });
    inputEl.focus();
  } finally {
    isCreatingChat = false;
  }
}

function handleResetHistory() {
  if (isSending) {
    return;
  }

  const confirmed = window.confirm("Reset all chat history? This cannot be undone.");
  if (!confirmed) {
    return;
  }

  localStorage.removeItem(STORAGE_KEY);
  state = createInitialState();
  saveState();
  render();
  inputEl.focus();
}

function render(staggerMessages = false, options = {}) {
  const popChatId = options && typeof options.popChatId === "string" ? options.popChatId : "";
  renderChatList(popChatId);
  renderMessages(staggerMessages);
}

composerEl.addEventListener("submit", handleSend);
sidebarNewChatBtn.addEventListener("click", handleNewChat);
headerNewChatBtn.addEventListener("click", handleNewChat);
resetHistoryBtn.addEventListener("click", handleResetHistory);
attachmentInputEl.addEventListener("change", handleAttachmentSelect);
micBtn.addEventListener("click", handleMicToggle);
readLastBtn.addEventListener("click", handleReadLastReply);

applyTheme();

render();
