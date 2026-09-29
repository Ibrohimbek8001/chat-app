let sessionId = null;
let aborter = null;
let sending = false;
const MAX_FILE_MB = 10;

const messagesEl = document.getElementById("messages");
const sourcesEl = document.getElementById("sources");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const stopBtn = document.getElementById("stopBtn");
const sessionsEl = document.getElementById("sessions");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const statsEl = document.getElementById("stats");
const charCount = document.getElementById("charCount");
const emptyState = document.getElementById("emptyState");

// --- Xavfsiz mini-markdown: avval escape, keyin format ---
function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdown(text) {
  let h = escapeHtml(text);
  // kod bloklari ```...```
  h = h.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code}</code></pre>`);
  // inline kod `...`
  h = h.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  // bold / italic
  h = h.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  h = h.replace(/(^|\s)\*([^*\n]+)\*/g, "$1<i>$2</i>");
  // ro'yxatlar: "- " bilan boshlangan qatorlar
  h = h.replace(/(?:^|\n)((?:\s*[-•] .*(?:\n|$))+)/g, (m, list) => {
    const items = list.trim().split("\n").map((l) => `<li>${l.replace(/^\s*[-•] /, "")}</li>`).join("");
    return `\n<ul>${items}</ul>`;
  });
  // qatorlar
  h = h.replace(/\n/g, "<br>");
  return h;
}

function addMsg(text, who) {
  if (emptyState) emptyState.style.display = "none";
  const d = document.createElement("div");
  d.className = "msg " + who;
  if (who === "ai") d.innerHTML = renderMarkdown(text);
  else d.textContent = text;
  messagesEl.appendChild(d);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return d;
}

function setSending(v) {
  sending = v;
  sendBtn.disabled = v;
  input.disabled = v;
  stopBtn.style.display = v ? "inline-block" : "none";
  sendBtn.style.display = v ? "none" : "inline-block";
}

async function loadSessions() {
  try {
    const r = await fetch("/api/sessions");
    if (!r.ok) return;
    const list = await r.json();
    sessionsEl.innerHTML = "";
    list.forEach((s) => {
      const li = document.createElement("li");
      li.textContent = s.title || "Yangi chat";
      li.title = s.title || "Yangi chat";
      if (s.id === sessionId) li.classList.add("active");
      li.onclick = () => openSession(s.id);
      sessionsEl.appendChild(li);
    });
  } catch {}
}

async function openSession(id) {
  if (sending) return;
  sessionId = id;
  messagesEl.innerHTML = "";
  sourcesEl.innerHTML = "";
  if (emptyState) emptyState.style.display = "none";
  try {
    const r = await fetch(`/api/sessions/${id}`);
    if (!r.ok) return;
    const data = await r.json();
    if (!data.messages.length && emptyState) {
      emptyState.style.display = "block";
    }
    data.messages.forEach((m) => addMsg(m.content, m.role === "user" ? "user" : "ai"));
  } catch {}
  loadSessions();
  loadStats();
}

async function loadStats() {
  try {
    const r = await fetch("/api/stats");
    if (!r.ok) return;
    const s = await r.json();
    statsEl.textContent = `📚 Vektorlar: ${s.vectors} | 💬 Chatlar: ${s.sessions} | RAG: ${s.rag_ready ? "ON" : "OFF (.env?)"}`;
  } catch {}
}

document.getElementById("newChat").onclick = () => {
  if (sending) return;
  sessionId = null;
  messagesEl.innerHTML = "";
  sourcesEl.innerHTML = "";
  if (emptyState) emptyState.style.display = "block";
  loadSessions();
};

if (stopBtn) {
  stopBtn.onclick = () => {
    if (aborter) aborter.abort();
  };
}

// Textarea auto-resize + counter
function autoresize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
  if (charCount) charCount.textContent = `${input.value.length}/4000`;
}
input.addEventListener("input", autoresize);
autoresize();

// Streaming yuborish — POST + fetch (URL limiti yo'q, stop bilan)
form.onsubmit = async (e) => {
  e.preventDefault();
  if (sending) return;
  const text = input.value.trim();
  if (!text) return;
  if (text.length > 4000) {
    alert("Xabar juda uzun (4000 dan oshmasin)");
    return;
  }
  input.value = "";
  autoresize();
  addMsg(text, "user");
  const aiBubble = addMsg("...", "ai");
  sourcesEl.innerHTML = "";
  setSending(true);
  aborter = new AbortController();

  let full = "";
  try {
    const r = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
      signal: aborter.signal,
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      aiBubble.innerHTML = renderMarkdown("❌ " + (err.detail || `Server xatolik: ${r.status}`));
      setSending(false);
      return;
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const p of parts) {
        const line = p.trim();
        if (!line.startsWith("data:")) continue;
        let data;
        try {
          data = JSON.parse(line.slice(5).trim());
        } catch {
          continue;
        }
        if (data.type === "sources") {
          if (data.sources.length)
            sourcesEl.innerHTML = "📎 Manba: " + data.sources.map((s) => escapeHtml(s.source)).join(", ");
          else sourcesEl.innerHTML = "📎 Manba: hujjat topilmadi (umumiy bilim)";
        } else if (data.type === "token") {
          full += data.text;
          aiBubble.innerHTML = renderMarkdown(full);
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (data.type === "error") {
          full += "\n\n❌ " + data.text;
          aiBubble.innerHTML = renderMarkdown(full);
        } else if (data.type === "done") {
          sessionId = data.session_id;
        }
      }
    }
    if (!full) aiBubble.innerHTML = renderMarkdown("❌ Bo'sh javob. Qayta urining.");
  } catch (err) {
    if (err.name === "AbortError") {
      aiBubble.innerHTML = renderMarkdown(full ? full + "\n\n⏹ To'xtatildi" : "⏹ To'xtatildi");
      // To'xtatilgan javobni saqlash uchun serverga oddiy chat orqali yozmaymiz —
      // keyingi xabar yangi kontekstdan boshlanadi
    } else {
      if (!full) aiBubble.textContent = "❌ Xatolik. Server ishlayaptimi? .env da GEMINI_API_KEY bormi?";
    }
  } finally {
    setSending(false);
    aborter = null;
    loadSessions();
    loadStats();
  }
};

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

fileInput.onchange = async () => {
  const f = fileInput.files[0];
  if (!f) return;
  const okExt = /\.(pdf|txt|md)$/i.test(f.name);
  if (!okExt) {
    uploadStatus.textContent = "❌ Faqat .pdf / .txt / .md";
    fileInput.value = "";
    return;
  }
  if (f.size > MAX_FILE_MB * 1024 * 1024) {
    uploadStatus.textContent = `❌ Fayl juda katta (>${MAX_FILE_MB}MB)`;
    fileInput.value = "";
    return;
  }
  uploadStatus.textContent = "⏳ Yuklanmoqda...";
  try {
    const fd = new FormData();
    fd.append("file", f);
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({}));
    if (r.ok) uploadStatus.textContent = `✅ ${data.source}: ${data.chunks} bo'lak Chroma ga yozildi`;
    else uploadStatus.textContent = "❌ " + (data.detail || "Xatolik");
  } catch {
    uploadStatus.textContent = "❌ Tarmoq xatolik";
  }
  fileInput.value = "";
  loadStats();
};

document.getElementById("clearDocs").onclick = async () => {
  if (!confirm("Hamma hujjat unutilsinmi?")) return;
  await fetch("/api/documents", { method: "DELETE" });
  loadStats();
  sourcesEl.innerHTML = "";
};

loadSessions();
loadStats();
