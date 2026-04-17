const state = {
  busy: false,
  lastActiveJobId: null,
};

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || "请求失败");
  }
  return payload;
}

function showToast(message, isError = false) {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.style.background = isError ? "rgba(164, 58, 44, 0.95)" : "rgba(31, 36, 48, 0.92)";
  toast.classList.add("show");
  window.clearTimeout(toast._timer);
  toast._timer = window.setTimeout(() => toast.classList.remove("show"), 2600);
}

function setButtonsDisabled(disabled) {
  document.getElementById("runSingleBtn").disabled = disabled;
  document.getElementById("runSixwayBtn").disabled = disabled;
}

function formatNumber(value) {
  return Number(value || 0).toFixed(2);
}

function formatDate(value) {
  return value || "-";
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderEnv(config) {
  const apiMode = document.getElementById("apiMode");
  const kbName = document.getElementById("kbName");
  const metadataDir = document.getElementById("metadataDir");
  const apiModeSelect = document.getElementById("apiModeSelect");
  const apiModeNote = document.getElementById("apiModeNote");
  apiMode.textContent = config.api_mode || "-";
  kbName.textContent = config.active_kb || "-";
  metadataDir.textContent = config.metadata_cache_dir || "-";

  apiModeSelect.innerHTML = "";
  (config.available_api_modes || []).forEach((mode) => {
    const option = document.createElement("option");
    option.value = mode;
    option.textContent = mode;
    option.selected = mode === config.api_mode;
    apiModeSelect.appendChild(option);
  });

  const issues = config.api_mode_issues || [];
  apiModeNote.classList.remove("ready", "warning");
  if (config.api_mode_ready) {
    apiModeNote.classList.add("ready");
    apiModeNote.textContent = `当前 ${config.api_mode} 模式的关键环境已就绪，后续页面发起的评测会直接沿用这套配置。`;
  } else if (issues.length) {
    apiModeNote.classList.add("warning");
    apiModeNote.textContent = `当前 ${config.api_mode} 模式还有配置缺口：${issues.join("，")}。如果你已经在 scripts/api_mode_exports.private.sh 里配好 key，重新应用一次模式即可。`;
  } else {
    apiModeNote.textContent = "当前模式配置检查中...";
  }

  const datasetSelect = document.getElementById("datasetPath");
  datasetSelect.innerHTML = "";
  const datasets = config.datasets || [];
  if (!datasets.length) {
    const option = document.createElement("option");
    option.value = "data/mini_dev.json";
    option.textContent = "data/mini_dev.json";
    datasetSelect.appendChild(option);
    return;
  }
  datasets.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.value;
    option.textContent = `${item.label} (${item.size})`;
    if (item.value === "data/mini_dev.json") {
      option.selected = true;
    }
    datasetSelect.appendChild(option);
  });
}

function renderJobStatus(snapshot) {
  state.busy = !!snapshot.busy;
  setButtonsDisabled(state.busy);

  const badge = document.getElementById("busyBadge");
  badge.classList.remove("busy", "error");
  badge.textContent = state.busy ? "任务运行中" : "空闲中";
  if (state.busy) {
    badge.classList.add("busy");
  }

  const summary = document.getElementById("jobSummary");
  const log = document.getElementById("jobLog");
  const job = snapshot.active_job;
  const recentJobs = snapshot.recent_jobs || [];

  if (job) {
    state.lastActiveJobId = job.id;
    const params = job.params || {};
    summary.className = "job-summary";
    summary.innerHTML = `
      <strong>${escapeHtml(job.type === "sixway" ? "六组对比任务" : "单组评测任务")}</strong><br />
      状态：${escapeHtml(job.status)}<br />
      开始时间：${escapeHtml(job.started_at)}<br />
      数据集：${escapeHtml(params.dataset_path || "-")}<br />
      ${job.type === "single" ? `模式：${escapeHtml(params.mode || "-")}<br />` : ""}
      ${job.type === "single" ? `Hints：${params.include_dataset_evidence ? "ON" : "OFF"}<br />` : ""}
      ${job.type === "single" ? `候选数：${escapeHtml(params.candidate_count || 1)}<br />` : ""}
      ${params.test_limit ? `样本上限：${escapeHtml(params.test_limit)}<br />` : "样本上限：全量<br />"}
    `;
    log.textContent = (job.logs || []).join("\n") || "任务已启动，等待日志输出...";
    return;
  }

  const latest = recentJobs[0];
  if (latest) {
    const hasError = !!latest.error;
    if (hasError) {
      badge.textContent = "上次任务失败";
      badge.classList.add("error");
    }
    const latestSummary = latest.summary || {};
    summary.className = "job-summary";
    summary.innerHTML = `
      <strong>最近一次任务：${escapeHtml(latest.type === "sixway" ? "六组对比" : "单组评测")}</strong><br />
      状态：${escapeHtml(latest.status)}<br />
      开始时间：${escapeHtml(latest.started_at)}<br />
      结束时间：${escapeHtml(latest.ended_at || "-")}<br />
      ${latestSummary.accuracy !== undefined ? `EX：${formatNumber(latestSummary.accuracy)}%<br />` : ""}
      ${latestSummary.execution_success_rate !== undefined ? `Exec：${formatNumber(latestSummary.execution_success_rate)}%<br />` : ""}
      ${latest.error ? `错误：${escapeHtml(latest.error)}` : "可以直接继续发起新的任务。"}
    `;
    log.textContent = "当前没有正在运行的任务。";
    return;
  }

  summary.className = "job-summary empty-state";
  summary.textContent = "当前没有正在运行的任务。";
  log.textContent = "等待任务启动...";
}

function computeMetricScale(rows, key) {
  const values = rows.map((row) => Number(row[key] || 0));
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  let range = maxValue - minValue;

  if (range < 0.01) {
    range = 1;
  }

  const padding = Math.max(range * 0.2, key === "execution_success_rate" ? 0.08 : 1.5);
  const lowerBound = Math.max(0, minValue - padding);
  const upperBound = Math.min(100, maxValue + padding);
  const safeRange = Math.max(upperBound - lowerBound, 0.1);
  const middle = lowerBound + safeRange / 2;

  return {
    min: lowerBound,
    max: upperBound,
    mid: middle,
    toHeight(value) {
      const normalized = (Number(value || 0) - lowerBound) / safeRange;
      return 18 + Math.max(0, Math.min(1, normalized)) * 78;
    },
  };
}

function setAxisTicks(prefix, scale) {
  document.getElementById(`${prefix}TickTop`).textContent = `${formatNumber(scale.max)}%`;
  document.getElementById(`${prefix}TickMid`).textContent = `${formatNumber(scale.mid)}%`;
  document.getElementById(`${prefix}TickBottom`).textContent = `${formatNumber(scale.min)}%`;
  document.getElementById(`${prefix}RangeNote`).textContent = `局部纵轴：${formatNumber(scale.min)}% - ${formatNumber(scale.max)}%`;
}

function renderMetricChart(containerId, rows, key, barClass, prefix) {
  const chart = document.getElementById(containerId);
  chart.innerHTML = "";

  if (!rows.length) {
    chart.innerHTML = `<div class="empty-state">暂无数据。</div>`;
    return;
  }

  const scale = computeMetricScale(rows, key);
  setAxisTicks(prefix, scale);

  rows.forEach((row) => {
    const value = Number(row[key] || 0);
    const group = document.createElement("div");
    group.className = "bar-group";
    group.innerHTML = `
      <div class="bar-column">
        <div class="bar-value">${formatNumber(value)}%</div>
        <div class="bar ${barClass}" style="height:${scale.toHeight(value)}%"></div>
      </div>
      <div class="bar-note" title="${escapeHtml(row.label)}">${escapeHtml(row.label)}</div>
    `;
    chart.appendChild(group);
  });
}

function renderComparison(payload) {
  const title = document.getElementById("comparisonTitle");
  const source = document.getElementById("comparisonSource");
  const tbody = document.querySelector("#comparisonTable tbody");
  const rows = payload.rows || [];

  title.textContent = payload.title || "最近结果对比";
  source.textContent = payload.source === "latest_sixway" ? "来自最近一次 sixway 汇总" : "来自 history 最近结果";

  tbody.innerHTML = "";

  if (!rows.length) {
    document.getElementById("accuracyChart").innerHTML = `<div class="empty-state">暂无可展示的对比结果。</div>`;
    document.getElementById("executionChart").innerHTML = `<div class="empty-state">暂无可展示的对比结果。</div>`;
    ["accuracy", "execution"].forEach((prefix) => {
      document.getElementById(`${prefix}TickTop`).textContent = "-";
      document.getElementById(`${prefix}TickMid`).textContent = "-";
      document.getElementById(`${prefix}TickBottom`).textContent = "-";
      document.getElementById(`${prefix}RangeNote`).textContent = "-";
    });
    return;
  }

  renderMetricChart("accuracyChart", rows, "accuracy", "ex", "accuracy");
  renderMetricChart("executionChart", rows, "execution_success_rate", "exec", "execution");

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(row.label)}</td>
      <td class="metric-cell">${formatNumber(row.accuracy)}</td>
      <td>${formatNumber(row.execution_success_rate)}</td>
      <td>${row.correct}/${row.total}</td>
    `;
    tbody.appendChild(tr);
  });
}

function createFileButton(label, path) {
  if (!path) {
    return '<span class="muted">-</span>';
  }
  return `<button class="mini-btn" data-file-path="${escapeHtml(path)}">${label}</button>`;
}

function renderHistory(payload) {
  const tbody = document.querySelector("#historyTable tbody");
  tbody.innerHTML = "";
  const rows = payload.rows || [];
  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="6" class="muted">还没有历史记录。</td>`;
    tbody.appendChild(tr);
    return;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(row.label)}</td>
      <td class="metric-cell">${formatNumber(row.accuracy)}</td>
      <td>${formatNumber(row.execution_success_rate)}</td>
      <td>${escapeHtml(row.dataset_path || "-")}</td>
      <td>${escapeHtml(formatDate(row.ended_at || row.started_at))}</td>
      <td>
        <div class="table-actions">
          ${createFileButton("摘要", row.summary_path)}
          ${createFileButton("明细", row.result_path)}
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadFile(path) {
  try {
    const payload = await apiFetch(`/api/file?path=${encodeURIComponent(path)}`);
    document.getElementById("viewerPath").textContent = payload.path || path;
    const viewer = document.getElementById("fileViewer");
    viewer.textContent =
      payload.content_type === "json"
        ? JSON.stringify(payload.content, null, 2)
        : payload.content || "";
  } catch (error) {
    showToast(error.message, true);
  }
}

async function refreshConfig() {
  const payload = await apiFetch("/api/config");
  renderEnv(payload);
}

async function applyApiMode() {
  const button = document.getElementById("applyApiModeBtn");
  const mode = document.getElementById("apiModeSelect").value;
  try {
    button.disabled = true;
    const payload = await apiFetch("/api/config/api-mode", {
      method: "POST",
      body: JSON.stringify({ mode }),
    });
    showToast(payload.message || `已切换到 ${mode}`);
    await refreshConfig();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
}

async function refreshStatus() {
  const snapshot = await apiFetch("/api/status");
  renderJobStatus(snapshot);
}

async function refreshComparison() {
  const payload = await apiFetch("/api/comparison");
  renderComparison(payload);
}

async function refreshHistory() {
  const payload = await apiFetch("/api/history?limit=12");
  renderHistory(payload);
}

function collectFormPayload() {
  return {
    dataset_path: document.getElementById("datasetPath").value,
    mode: document.getElementById("mode").value,
    test_limit: document.getElementById("testLimit").value || null,
    candidate_count: document.getElementById("candidateCount").value || 1,
    include_dataset_evidence: document.getElementById("useHints").checked,
  };
}

async function runSingle(event) {
  event.preventDefault();
  try {
    const payload = collectFormPayload();
    await apiFetch("/api/run/single", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    showToast("单组评测已经启动。");
    await refreshStatus();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function runSixway() {
  try {
    const formPayload = collectFormPayload();
    await apiFetch("/api/run/sixway", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: formPayload.dataset_path,
        test_limit: formPayload.test_limit,
      }),
    });
    showToast("六组对比已经启动。");
    await refreshStatus();
  } catch (error) {
    showToast(error.message, true);
  }
}

function bindEvents() {
  document.getElementById("runForm").addEventListener("submit", runSingle);
  document.getElementById("runSixwayBtn").addEventListener("click", runSixway);
  document.getElementById("applyApiModeBtn").addEventListener("click", applyApiMode);
  document.getElementById("historyTable").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-file-path]");
    if (!button) {
      return;
    }
    await loadFile(button.dataset.filePath);
  });
}

async function boot() {
  bindEvents();
  try {
    await refreshConfig();
    await Promise.all([refreshStatus(), refreshComparison(), refreshHistory()]);
  } catch (error) {
    showToast(error.message, true);
  }
  window.setInterval(refreshStatus, 2000);
  window.setInterval(refreshComparison, 8000);
  window.setInterval(refreshHistory, 8000);
}

document.addEventListener("DOMContentLoaded", boot);
