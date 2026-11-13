const $ = (id) => document.getElementById(id);
let selected = null;
let latest = null;

const PROMPT = `先读 AGENTS.md、docs/collab/CURSOR_HANDOFF.md 和 WSL skill 的 SKILL.md；
确认 uname -a 为 Linux、当前分支和未提交改动。
只从 handoff 的“下一步”继续，不重做已完成部分。
改动前说明计划；完成后跑测试并更新 handoff。
正式件默认不覆盖；使用 AI 按 references/18 记录到 AI 使用报告。`;

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function quotaInfo(q) {
  const k = String(q || "unknown").toLowerCase();
  if (k === "n/a" || k === "na") return { cls: "na", width: 0, label: "N/A" };
  if (k === "green") return { cls: "", width: 34, label: "34%" };
  if (k === "yellow") return { cls: "", width: 72, label: "72%" };
  if (k === "red") return { cls: "", width: 93, label: "93%" };
  return { cls: "unknown", width: 35, label: "MANUAL" };
}

function statusLabel(s) {
  const k = String(s || "standby").toLowerCase();
  if (k === "active") return "ACTIVE";
  if (k === "blocked") return "BLOCKED";
  if (k === "offline") return "OFFLINE";
  return "STANDBY";
}

function renderAgents(data) {
  const grid = $("agents");
  grid.innerHTML = data.agents.map((a) => {
    const q = quotaInfo(a.quota);
    const st = statusLabel(a.status);
    return `<article class="agent-card ${selected === a.id ? "selected" : ""}" data-agent="${esc(a.id)}" style="--accent:${esc(a.accent || "#38bdf8")}">
      <div class="agent-head">
        <div>
          <div class="agent-name">${esc(a.name)}</div>
          <div class="agent-runtime">${esc(a.runtime)}</div>
        </div>
        <div class="agent-role">${esc(a.role)}</div>
      </div>
      <div class="agent-task">${esc(a.task)}</div>
      <div class="agent-foot">
        <span class="status-pill ${st.toLowerCase()}">${st}</span>
        <span class="quota">QUOTA <span class="quota-bar ${q.cls}"><i style="width:${q.width}%"></i></span>${q.label}</span>
      </div>
    </article>`;
  }).join("");
  grid.querySelectorAll(".agent-card").forEach((el) => {
    el.addEventListener("click", () => {
      selected = selected === el.dataset.agent ? null : el.dataset.agent;
      renderAgents(data);
      highlightFlow();
    });
  });
}

function ageLabel(m) {
  if (m == null) return "NO HANDOFF";
  if (m < 1) return "JUST NOW";
  if (m < 60) return `${m} MIN AGO`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}H ${m % 60}M AGO`;
  return `${Math.floor(h / 24)}D AGO`;
}

function renderHandoff(data) {
  const h = data.handoff || {};
  $("owner-name").textContent = (data.owner || "unknown").toUpperCase();
  $("owner-label").textContent = `OWNER: ${(data.owner || "--").toUpperCase()}`;
  const age = h.age_minutes;
  const stale = age != null && age > 120;
  $("handoff-age").textContent = stale ? `STALE · ${ageLabel(age)}` : ageLabel(age);
  $("handoff-age").classList.toggle("stale", stale);

  const tasks = h.tasks && h.tasks.length ? h.tasks : [];
  $("tasks").innerHTML = tasks.length
    ? tasks.map((x) => `<li>${esc(x)}</li>`).join("")
    : `<li class="empty">No next actions recorded.</li>`;

  const blockers = h.blockers && h.blockers.length ? h.blockers : [];
  $("blockers").innerHTML = blockers.length
    ? blockers.map((x) => `<li>${esc(x)}</li>`).join("")
    : `<li class="empty">None recorded.</li>`;

  const own = h.ownership && h.ownership.length ? h.ownership : [];
  $("ownership").innerHTML = own.length
    ? own.map((x) => `<li>${esc(x)}</li>`).join("")
    : `<li class="empty">Not recorded.</li>`;
}

function renderHealth(data) {
  const g = data.git || {};
  const s = data.skill || {};
  const c = data.codex_sessions || {};
  $("branch").textContent = g.branch || "--";
  $("dirty").textContent = g.dirty ? "DIRTY" : "CLEAN";
  $("dirty").style.color = g.dirty ? "var(--amber)" : "var(--green)";
  $("skill-files").textContent = s.exists ? `${s.files} FILES` : "MISSING";
  $("skill-updated").textContent = s.updated ? s.updated.slice(5, 16).replace("T", " ") : "--";
  $("sessions").textContent = c.exists ? `${c.today} FILES` : "--";
  $("last-commit").textContent = g.last_commit || "--";
  $("generated").textContent = `UPDATED ${String(data.generated_at || "").slice(11, 19)}`;
  const recent = (g.recent || []).slice(0, 4);
  $("recent").innerHTML = recent.map((x) => `<span class="commit">${esc(x)}</span>`).join("");
  $("notes").innerHTML = (data.notes || []).map((x) => `<span class="note">${esc(x)}</span>`).join("");
  $("phase").textContent = `PHASE ${String(data.phase || "--").toUpperCase()}`;
}

function highlightFlow() {
  document.querySelectorAll(".flow-node, .branch-node").forEach((el) => {
    el.style.outline = selected && el.dataset.agent === selected ? "1px solid rgba(56,189,248,.55)" : "none";
    el.style.outlineOffset = "3px";
  });
}

function render(data) {
  latest = data;
  renderAgents(data);
  renderHandoff(data);
  renderHealth(data);
  highlightFlow();
}

async function load() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    render(await r.json());
  } catch (e) {
    $("generated").textContent = "API OFFLINE";
    $("generated").style.color = "var(--red)";
  }
}

function clock() {
  $("clock").textContent = new Date().toLocaleTimeString([], { hour12: false });
}

$("copy-prompt").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(PROMPT);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = PROMPT;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
  const t = $("toast");
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 1200);
});

clock();
setInterval(clock, 1000);
load();
setInterval(load, 10000);
