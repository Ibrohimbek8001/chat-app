let sessionId = null;
const messagesEl = document.getElementById("messages");
const sourcesEl = document.getElementById("sources");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sessionsEl = document.getElementById("sessions");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const statsEl = document.getElementById("stats");

function addMsg(text, who) {
  const d = document.createElement("div");
  d.className = "msg " + who;
  d.textContent = text;
  messagesEl.appendChild(d);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return d;
}

async function loadSessions() {
  const r = await fetch("/api/sessions");
  const list = await r.json();
  sessionsEl.innerHTML = "";
  list.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = s.title || "Yangi chat";
    if (s.id === sessionId) li.classList.add("active");
    li.onclick = () => openSession(s.id);
    sessionsEl.appendChild(li);
  });
}

async function openSession(id) {
  sessionId = id;
  messagesEl.innerHTML = "";
  sourcesEl.innerHTML = "";
  const r = await fetch(`/api/sessions/${id}`);
  const data = await r.json();
  data.messages.forEach((m) => addMsg(m.content, m.role === "user" ? "user" : "ai"));
  loadSessions();
  loadStats();
}

async function loadStats() {
  try {
    const r = await fetch("/api/stats");
    const s = await r.json();
    statsEl.textContent = `📚 Vektorlar: ${s.vectors} | 💬 Chatlar: ${s.sessions} | RAG: ${s.rag_ready ? "ON" : "OFF (.env?)"}`;
  } catch {}
}

document.getElementById("newChat").onclick = () => {
  sessionId = null;
  messagesEl.innerHTML = "";
  sourcesEl.innerHTML = "";
  loadSessions();
};

// Streaming yuborish (SSE)
form.onsubmit = async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  addMsg(text, "user");
  const aiBubble = addMsg("...", "ai");
  sourcesEl.innerHTML = "";

  const qs = new URLSearchParams({ message: text, ...(sessionId ? { session_id: sessionId } : {}) });
  const es = new EventSource(`/api/chat/stream?${qs}`);
  let full = "";
  es.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === "sources") {
      if (data.sources.length)
        sourcesEl.innerHTML = "📎 Manba: " + data.sources.map((s) => s.source).join(", ");
      else sourcesEl.innerHTML = "📎 Manba: hujjat topilmadi (umumiy bilim)";
    } else if (data.type === "token") {
      full += data.text;
      aiBubble.textContent = full;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    } else if (data.type === "done") {
      sessionId = data.session_id;
      es.close();
      loadSessions();
    }
  };
  es.onerror = () => {
    es.close();
    if (!full) aiBubble.textContent = "❌ Xatolik. Server ishlayaptimi? .env da GEMINI_API_KEY bormi?";
  };
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
  uploadStatus.textContent = "⏳ Yuklanmoqda...";
  const fd = new FormData();
  fd.append("file", f);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  const data = await r.json();
  if (r.ok) uploadStatus.textContent = `✅ ${data.source}: ${data.chunks} bo'lak Chroma ga yozildi`;
  else uploadStatus.textContent = "❌ " + (data.detail || "Xatolik");
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
