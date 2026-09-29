let sessionId = null;
let aborter = null;
let sending = false;
let openToken = 0;
// Serverdan /api/stats orqali yangilanadi (.env dagi limitlar)
const limits = { max_message_chars: 4000, max_upload_mb: 10 };

const messagesEl = document.getElementById("messages");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");
const stopBtn = document.getElementById("stopBtn");
const sessionsEl = document.getElementById("sessions");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const uploadHint = document.getElementById("uploadHint");
const statsEl = document.getElementById("stats");
const charCount = document.getElementById("charCount");
const docsEl = document.getElementById("docs");
const docsCount = document.getElementById("docsCount");
const toastEl = document.getElementById("toast");
// #messages tozalanganda DOM dan chiqib ketadi — referensni saqlab, qayta qo'yamiz
const emptyState = document.getElementById("emptyState");

// ---------- Umumiy yordamchilar ----------

let toastTimer = null;
function toast(text, kind = "error") {
  toastEl.textContent = text;
  toastEl.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (toastEl.className = "toast"), 3500);
}

async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `Server xatolik: ${r.status}`);
  return data;
}

// ---------- Xavfsiz mini-markdown: avval escape, keyin format ----------

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Kiruvchi matn allaqachon escape qilingan
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => {
    codes.push(c);
    return `\u0000${codes.length - 1}\u0000`;
  });
  s = s.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );
  s = s.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  s = s.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<i>$2</i>");
  s = s.replace(/(^|[\s(])_([^_\n]+)_(?=$|[\s).,!?:;])/g, "$1<i>$2</i>");
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[i]}</code>`);
}

const RE_FENCE = /^\s*```/;
const RE_HEADING = /^(#{1,6})\s+(.*)$/;
const RE_UL = /^\s*[-*•]\s+(.*)$/;
const RE_OL = /^\s*(\d+)[.)]\s+(.*)$/;
const RE_TABLE_ROW = /^\s*\|.*\|\s*$/;
const RE_TABLE_SEP = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/;

function isTableStart(lines, i) {
  return RE_TABLE_ROW.test(lines[i]) && i + 1 < lines.length && RE_TABLE_SEP.test(lines[i + 1]);
}

function isBlockStart(lines, i) {
  const l = lines[i];
  return RE_FENCE.test(l) || RE_HEADING.test(l) || RE_UL.test(l) || RE_OL.test(l) || isTableStart(lines, i);
}

function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    let m;
    if (RE_FENCE.test(line)) {
      const buf = [];
      i++;
      while (i < lines.length && !RE_FENCE.test(lines[i])) buf.push(lines[i++]);
      i++; // yopuvchi ``` (stream paytida hali bo'lmasligi mumkin)
      out.push(
        `<div class="code"><button type="button" class="copy">Nusxa</button><pre><code>${buf.join("\n")}</code></pre></div>`
      );
    } else if ((m = line.match(RE_HEADING))) {
      const lvl = Math.min(m[1].length + 2, 6);
      out.push(`<h${lvl}>${inline(m[2])}</h${lvl}>`);
      i++;
    } else if (isTableStart(lines, i)) {
      const row = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const head = row(line);
      i += 2;
      const body = [];
      while (i < lines.length && RE_TABLE_ROW.test(lines[i])) body.push(row(lines[i++]));
      out.push(
        `<div class="table"><table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead>` +
          `<tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`
      );
    } else if (RE_UL.test(line)) {
      const items = [];
      while (i < lines.length && (m = lines[i].match(RE_UL))) {
        items.push(`<li>${inline(m[1])}</li>`);
        i++;
      }
      out.push(`<ul>${items.join("")}</ul>`);
    } else if ((m = line.match(RE_OL))) {
      const start = m[1];
      const items = [];
      while (i < lines.length && (m = lines[i].match(RE_OL))) {
        items.push(`<li>${inline(m[2])}</li>`);
        i++;
      }
      out.push(`<ol start="${start}">${items.join("")}</ol>`);
    } else if (!line.trim()) {
      i++;
    } else {
      const para = [inline(lines[i++])];
      while (i < lines.length && lines[i].trim() && !isBlockStart(lines, i)) para.push(inline(lines[i++]));
      out.push(`<p>${para.join("<br>")}</p>`);
    }
  }
  return out.join("");
}

// ---------- Xabarlar ----------

function nearBottom() {
  return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 80;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function resetMessages(showEmpty) {
  messagesEl.replaceChildren();
  if (showEmpty) messagesEl.appendChild(emptyState);
}

function renderSources(msgEl, sources, warning) {
  msgEl.querySelectorAll(".src, .warn").forEach((n) => n.remove());
  if (warning) {
    const w = document.createElement("div");
    w.className = "warn";
    w.textContent = "⚠ " + warning;
    msgEl.appendChild(w);
  }
  if (!sources) return;
  if (!sources.length) {
    const d = document.createElement("div");
    d.className = "src none";
    d.textContent = "📎 Hujjatdan topilmadi (umumiy bilim)";
    msgEl.appendChild(d);
    return;
  }
  const names = [...new Set(sources.map((s) => s.source))];
  const det = document.createElement("details");
  det.className = "src";
  const sum = document.createElement("summary");
  sum.textContent = `📎 Manba: ${names.join(", ")} (${sources.length} parcha)`;
  det.appendChild(sum);
  sources.forEach((s) => {
    const p = document.createElement("div");
    p.className = "src-item";
    const b = document.createElement("b");
    b.textContent = s.source;
    p.append(b, document.createTextNode(": " + (s.preview || "").replace(/\s+/g, " ") + "…"));
    det.appendChild(p);
  });
  msgEl.appendChild(det);
}

function addMsg(text, who, sources) {
  emptyState.remove();
  const d = document.createElement("div");
  d.className = "msg " + who;
  const body = document.createElement("div");
  body.className = "body";
  if (who === "ai") body.innerHTML = renderMarkdown(text);
  else body.textContent = text;
  d.appendChild(body);
  if (who === "ai" && sources) renderSources(d, sources);
  messagesEl.appendChild(d);
  scrollToBottom();
  return d;
}

// Kod blokidan nusxa olish (event delegation)
messagesEl.addEventListener("click", async (e) => {
  const btn = e.target.closest(".copy");
  if (!btn) return;
  try {
    await navigator.clipboard.writeText(btn.nextElementSibling.innerText);
    btn.textContent = "✓ Nusxalandi";
    setTimeout(() => (btn.textContent = "Nusxa"), 1500);
  } catch {
    toast("Nusxa olib bo'lmadi");
  }
});

function setSending(v) {
  sending = v;
  sendBtn.disabled = v;
  input.disabled = v;
  stopBtn.hidden = !v;
  sendBtn.hidden = v;
  if (!v) input.focus();
}

// ---------- Sessiyalar ----------

async function loadSessions() {
  try {
    const list = await api("/api/sessions");
    sessionsEl.replaceChildren();
    list.forEach((s) => sessionsEl.appendChild(sessionItem(s)));
  } catch {}
}

function sessionItem(s) {
  const li = document.createElement("li");
  if (s.id === sessionId) li.classList.add("active");
  const title = document.createElement("span");
  title.className = "title";
  title.textContent = s.title || "Yangi chat";
  title.title = s.title || "Yangi chat";

  const actions = document.createElement("span");
  actions.className = "actions";
  const renameBtn = document.createElement("button");
  renameBtn.type = "button";
  renameBtn.title = "Nomini o'zgartirish";
  renameBtn.textContent = "✎";
  const delBtn = document.createElement("button");
  delBtn.type = "button";
  delBtn.title = "O'chirish";
  delBtn.textContent = "🗑";
  actions.append(renameBtn, delBtn);
  li.append(title, actions);

  li.onclick = () => openSession(s.id);
  renameBtn.onclick = (e) => {
    e.stopPropagation();
    startRename(li, title, s);
  };
  delBtn.onclick = async (e) => {
    e.stopPropagation();
    if (!confirm(`"${s.title}" chati o'chirilsinmi?`)) return;
    try {
      await api(`/api/sessions/${encodeURIComponent(s.id)}`, { method: "DELETE" });
      if (s.id === sessionId && !sending) {
        sessionId = null;
        resetMessages(true);
      }
      loadSessions();
      loadStats();
    } catch (err) {
      toast(err.message);
    }
  };
  return li;
}

function startRename(li, titleEl, s) {
  const inp = document.createElement("input");
  inp.className = "rename";
  inp.value = s.title || "";
  inp.maxLength = 100;
  titleEl.replaceWith(inp);
  inp.focus();
  inp.select();
  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    const t = inp.value.trim();
    if (save && t && t !== s.title) {
      try {
        await api(`/api/sessions/${encodeURIComponent(s.id)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: t }),
        });
      } catch (err) {
        toast(err.message);
      }
    }
    loadSessions();
  };
  inp.onclick = (e) => e.stopPropagation();
  inp.onkeydown = (e) => {
    if (e.key === "Enter") finish(true);
    else if (e.key === "Escape") finish(false);
  };
  inp.onblur = () => finish(true);
}

async function openSession(id) {
  if (sending) return;
  const token = ++openToken; // tez-tez bosilganda eski javob yangisini bosmasin
  sessionId = id;
  resetMessages(false);
  try {
    const data = await api(`/api/sessions/${encodeURIComponent(id)}`);
    if (token !== openToken) return;
    if (!data.messages.length) resetMessages(true);
    data.messages.forEach((m) =>
      m.role === "user" ? addMsg(m.content, "user") : addMsg(m.content, "ai", m.sources)
    );
  } catch (err) {
    if (token === openToken) toast(err.message);
  }
  loadSessions();
}

// ---------- Hujjatlar + statistika ----------

async function loadStats() {
  try {
    const s = await api("/api/stats");
    Object.assign(limits, s.limits || {});
    input.maxLength = limits.max_message_chars;
    uploadHint.textContent = `(max ${limits.max_upload_mb}MB)`;
    autoresize();
    statsEl.textContent = `📚 Vektorlar: ${s.vectors} | 💬 Chatlar: ${s.sessions} | RAG: ${s.rag_ready ? "ON" : "OFF (.env?)"}`;
    if (s.rag_ready) loadDocs();
  } catch {}
}

async function loadDocs() {
  try {
    const docs = await api("/api/documents");
    docsCount.textContent = `(${docs.length})`;
    docsEl.replaceChildren();
    if (!docs.length) {
      const li = document.createElement("li");
      li.className = "muted";
      li.textContent = "Hali hujjat yuklanmagan";
      docsEl.appendChild(li);
    }
    docs.forEach((d) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = `${d.source} (${d.chunks})`;
      name.title = d.source;
      const del = document.createElement("button");
      del.type = "button";
      del.title = "Hujjatni o'chirish";
      del.textContent = "×";
      del.onclick = async () => {
        if (!confirm(`"${d.source}" unutilsinmi?`)) return;
        try {
          await api(`/api/documents/${encodeURIComponent(d.source)}`, { method: "DELETE" });
          loadStats();
        } catch (err) {
          toast(err.message);
        }
      };
      li.append(name, del);
      docsEl.appendChild(li);
    });
  } catch {}
}

document.getElementById("newChat").onclick = () => {
  if (sending) return;
  openToken++;
  sessionId = null;
  resetMessages(true);
  loadSessions();
  input.focus();
};

stopBtn.onclick = () => {
  if (aborter) aborter.abort();
};

// Textarea auto-resize + counter
function autoresize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
  charCount.textContent = `${input.value.length}/${limits.max_message_chars}`;
}
input.addEventListener("input", autoresize);
autoresize();

// ---------- Streaming yuborish — POST + fetch (URL limiti yo'q, stop bilan) ----------

form.onsubmit = async (e) => {
  e.preventDefault();
  if (sending) return;
  const text = input.value.trim();
  if (!text) return;
  if (text.length > limits.max_message_chars) {
    toast(`Xabar juda uzun (${limits.max_message_chars} dan oshmasin)`);
    return;
  }
  input.value = "";
  autoresize();
  addMsg(text, "user");
  const aiMsg = addMsg("...", "ai");
  const aiBody = aiMsg.querySelector(".body");
  aiMsg.classList.add("typing");
  setSending(true);
  aborter = new AbortController();

  let full = "";
  // Har tokenda emas, kadrda bir marta render (uzun javobda O(n²) dan qochish)
  let frame = 0;
  const render = () => {
    frame = 0;
    const stick = nearBottom();
    aiBody.innerHTML = renderMarkdown(full);
    if (stick) scrollToBottom();
  };
  const scheduleRender = () => {
    if (!frame) frame = requestAnimationFrame(render);
  };
  const finalRender = (suffix) => {
    cancelAnimationFrame(frame);
    full += suffix || "";
    render();
  };

  try {
    const r = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
      signal: aborter.signal,
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      const detail = Array.isArray(err.detail) ? err.detail[0]?.msg : err.detail;
      finalRender("❌ " + (detail || `Server xatolik: ${r.status}`));
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
        if (data.type === "session") {
          sessionId = data.session_id;
        } else if (data.type === "sources") {
          renderSources(aiMsg, data.sources, data.warning);
        } else if (data.type === "token") {
          full += data.text;
          scheduleRender();
        } else if (data.type === "error") {
          finalRender("\n\n❌ " + data.text);
        }
      }
    }
    if (!full) finalRender("❌ Bo'sh javob. Qayta urining.");
    else finalRender();
  } catch (err) {
    if (err.name === "AbortError") {
      // Server qisman javobni "(javob to'xtatildi)" belgisi bilan saqlaydi
      finalRender(full ? "\n\n_(javob to'xtatildi)_" : "⏹ To'xtatildi");
    } else if (!full) {
      finalRender("❌ Xatolik. Server ishlayaptimi? .env da GEMINI_API_KEY bormi?");
    } else {
      finalRender("\n\n❌ Aloqa uzildi");
    }
  } finally {
    aiMsg.classList.remove("typing");
    setSending(false);
    aborter = null;
    loadSessions();
    loadStats();
  }
};

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    form.requestSubmit();
  }
});

// ---------- Fayl yuklash ----------

fileInput.onchange = async () => {
  const f = fileInput.files[0];
  if (!f) return;
  fileInput.value = "";
  if (!/\.(pdf|txt|md)$/i.test(f.name)) {
    uploadStatus.textContent = "❌ Faqat .pdf / .txt / .md";
    return;
  }
  if (f.size > limits.max_upload_mb * 1024 * 1024) {
    uploadStatus.textContent = `❌ Fayl juda katta (>${limits.max_upload_mb}MB)`;
    return;
  }
  uploadStatus.textContent = "⏳ Yuklanmoqda...";
  fileInput.disabled = true;
  try {
    const fd = new FormData();
    fd.append("file", f);
    const d = await api("/api/upload", { method: "POST", body: fd });
    let msg = `✅ ${d.source}: ${d.chunks} ta yangi bo'lak`;
    if (d.skipped) msg += `, ${d.skipped} ta takroriy bo'lak o'tkazib yuborildi`;
    if (d.truncated) msg += ". ⚠ Hujjat juda katta — faqat boshi yuklandi";
    uploadStatus.textContent = msg;
  } catch (err) {
    uploadStatus.textContent = "❌ " + (err.message || "Tarmoq xatolik");
  } finally {
    fileInput.disabled = false;
  }
  loadStats();
};

document.getElementById("clearDocs").onclick = async () => {
  if (!confirm("Hamma hujjat unutilsinmi?")) return;
  try {
    await api("/api/documents", { method: "DELETE" });
    toast("Hujjatlar tozalandi", "ok");
  } catch (err) {
    toast(err.message);
  }
  loadStats();
};

loadSessions();
loadStats();
