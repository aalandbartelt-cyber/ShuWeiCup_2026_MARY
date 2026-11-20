const $ = (id) => document.getElementById(id);
const isDesktop = Boolean(window.hudDesktop);
if (isDesktop) document.body.classList.add("desktop");
let latest = null;
let selected = null;
let pinned = false;
let hideTimer = 0;
const initialAgent = new URLSearchParams(location.search).get("agent");
let initialHandled = false;

const PROMPT = `先读 AGENTS.md、docs/collab/CURSOR_HANDOFF.md 和 WSL skill 的 SKILL.md；
确认 uname -a 为 Linux、当前分支和未提交改动。
只从 handoff 的“下一步”继续，不重做已完成部分。
改动前说明计划；完成后跑测试并更新 handoff。
正式件默认不覆盖；使用 AI 按 references/18 记录到 AI 使用报告。`;

const ICONS = {
  codex: "/icons/codex.png",
  cursor: "/icons/cursor.png",
  zcode: "/icons/zcode.png",
  dsh: "/icons/deepseek.png"
};
const INITIALS = { codex: "CX", cursor: "CU", zcode: "ZC", dsh: "DS" };
const SHORT = { codex: "Codex", cursor: "Cursor", zcode: "ZCode", dsh: "dsh" };

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function quotaInfo(q) {
  const k = String(q || "unknown").toLowerCase();
  if (k === "n/a" || k === "na") return { width: 0, label: "N/A" };
  if (k === "green") return { width: 32, label: "32%" };
  if (k === "yellow") return { width: 72, label: "72%" };
  if (k === "red") return { width: 94, label: "94%" };
  return { width: 28, label: "MANUAL" };
}

function statusLabel(s) {
  const k = String(s || "standby").toLowerCase();
  return k === "active" ? "ACTIVE" : k === "blocked" ? "BLOCKED" : k === "offline" ? "OFFLINE" : "STANDBY";
}

function setWindowOpen(open) {
  if (isDesktop) window.hudDesktop.resize(open ? 560 : 112);
}

function ageLabel(m) {
  if (m == null) return "NO HANDOFF";
  if (m < 1) return "JUST NOW";
  if (m < 60) return `${m}M AGO`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}H ${m % 60}M AGO`;
  return `${Math.floor(h / 24)}D AGO`;
}

function renderAgents(data) {
  const box = $("agents");
  box.innerHTML = (data.agents || []).map((a) => {
    const q = quotaInfo(a.quota);
    const st = statusLabel(a.status);
    return `<button class="agent ${selected === a.id ? "selected" : ""}" data-agent="${esc(a.id)}"
      style="--accent:${esc(a.accent || "#38bdf8")};--pct:${q.width}%" title="${esc(a.identity || a.name)}">
      <span class="ring"><img src="${esc(ICONS[a.id] || "")}" alt="" onerror="this.replaceWith(document.createTextNode('${esc(INITIALS[a.id] || a.name.slice(0, 2).toUpperCase())}'))"></span>
      <span class="agent-label">${esc(SHORT[a.id] || a.name)}</span>
      <span class="status-dot ${st.toLowerCase()}"></span>
    </button>`;
  }).join("");
  box.querySelectorAll(".agent").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      if (selected === el.dataset.agent && pinned) hidePop();
      else showPop(el.dataset.agent, true);
    });
    el.addEventListener("mouseenter", () => {
      clearTimeout(hideTimer);
      showPop(el.dataset.agent, false);
    });
    el.addEventListener("mouseleave", () => {
      if (!pinned) hideTimer = setTimeout(hidePop, 180);
    });
  });
}

function fillPop(data, agent) {
  const q = quotaInfo(agent.quota);
  const st = statusLabel(agent.status);
  $("pop-role").textContent = agent.role || "AGENT";
  $("pop-name").textContent = agent.name || agent.id;
  $("pop-runtime").textContent = agent.identity || agent.runtime || "";
  $("pop-status").textContent = st;
  $("pop-status").className = `pop-status ${st.toLowerCase()}`;
  $("pop-task").textContent = agent.task || "";
  $("pop-entry").textContent = agent.entry || "";
  $("pop-entry").title = agent.entry || "";
  $("pop-quota").textContent = q.label;
  $("pop-branch").textContent = data.git?.branch || "--";
  $("pop-dirty").textContent = data.git?.dirty ? "DIRTY" : "CLEAN";
  $("pop-skill").textContent = data.skill?.exists ? `${data.skill.files} FILES` : "MISSING";
  $("pop-sessions").textContent = data.codex_sessions?.exists ? `${data.codex_sessions.today} TODAY` : "--";
  $("pop-owner").textContent = (data.owner || "--").toUpperCase();

  const bus = data.bus || {};
  const mine = bus.by_agent?.[agent.id] || {};
  const latestMsg = (bus.recent || []).find((m) => m.to === agent.id || m.to === "all");
  const busBits = [];
  if (mine.unread) busBits.push(`${mine.unread} UNREAD`);
  if (mine.claimed) busBits.push(`${mine.claimed} CLAIMED`);
  if (!busBits.length) busBits.push("NO UNREAD");
  $("pop-bus").textContent = `BUS: ${busBits.join(" · ")}${latestMsg ? ` — ${latestMsg.subject}` : ""}`;
  $("pop-bus").title = latestMsg ? `${latestMsg.from} -> ${latestMsg.to}: ${latestMsg.subject}` : "No messages";

  const h = data.handoff || {};
  const tasks = h.tasks || [];
  $("tasks").innerHTML = tasks.length ? tasks.map((x) => `<li>${esc(x)}</li>`).join("") : `<li class="empty">No next actions.</li>`;
  const blockers = h.blockers || [];
  $("blockers").innerHTML = blockers.length ? blockers.map((x) => `<li>${esc(x)}</li>`).join("") : `<li class="empty">None.</li>`;
  $("generated").textContent = `UPDATED ${String(data.generated_at || "").slice(11, 19)}`;
}

function syncSelected() {
  document.querySelectorAll(".agent").forEach((el) => {
    el.classList.toggle("selected", el.dataset.agent === selected);
  });
}

function showPop(id, pin) {
  if (!latest) return;
  const agent = (latest.agents || []).find((x) => x.id === id);
  if (!agent) return;
  clearTimeout(hideTimer);
  selected = id;
  pinned = pin;
  fillPop(latest, agent);
  $("popover").hidden = false;
  setWindowOpen(true);
  syncSelected();
}

function hidePop() {
  if (pinned) return;
  selected = null;
  $("popover").hidden = true;
  setWindowOpen(false);
  syncSelected();
}

function render(data) {
  latest = data;
  $("phase").textContent = String(data.phase || "PRE").toUpperCase();
  $("owner-mini").textContent = (data.owner || "--").toUpperCase();
  const active = Number(data.bus?.active || 0);
  const badge = $("bus-badge");
  badge.hidden = active === 0;
  badge.textContent = `BUS ${active}`;
  badge.classList.toggle("hot", active > 0);
  const age = data.handoff?.age_minutes;
  const stale = age != null && age > 120;
  $("handoff-age").textContent = stale ? `STALE · ${ageLabel(age)}` : ageLabel(age);
  $("handoff-age").classList.toggle("stale", stale);
  $("generated").textContent = `UPDATED ${String(data.generated_at || "").slice(11, 19)}`;
  renderAgents(data);
  if (selected) {
    const agent = (data.agents || []).find((x) => x.id === selected);
    if (agent) fillPop(data, agent);
  }
  if (initialAgent && !initialHandled) {
    initialHandled = true;
    showPop(initialAgent, true);
  }
}

async function load() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    render(await r.json());
  } catch {
    $("generated").textContent = "API OFFLINE";
    $("generated").style.color = "var(--red)";
  }
}

function clock() {
  $("clock").textContent = new Date().toLocaleTimeString([], { hour12: false });
}

$("pop-close").addEventListener("click", () => {
  pinned = false;
  selected = null;
  $("popover").hidden = true;
  setWindowOpen(false);
  syncSelected();
});

$("win-close").addEventListener("click", () => {
  if (isDesktop) window.hudDesktop.close();
  else {
    pinned = false;
    selected = null;
    $("popover").hidden = true;
    setWindowOpen(false);
    syncSelected();
  }
});

$("popover").addEventListener("mouseenter", () => clearTimeout(hideTimer));
$("popover").addEventListener("mouseleave", () => {
  if (!pinned) hideTimer = setTimeout(hidePop, 180);
});

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

document.addEventListener("click", (e) => {
  if (!e.target.closest("#hud")) {
    pinned = false;
    selected = null;
    $("popover").hidden = true;
    setWindowOpen(false);
    syncSelected();
  }
});

clock();
setInterval(clock, 1000);
load();
setInterval(load, 10000);
