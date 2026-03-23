/* ── ICT Bot Dashboard — app.js ─────────────────────────────────────────── */

const API = '';   // same origin; empty = relative URLs
let equityChart = null;
let ws = null;
let pollTimer = null;

// ── Toast notifications ───────────────────────────────────────────────────
function toast(msg, type = 'info', duration = 4000) {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ── API helper ────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    const txt = await r.text().catch(() => r.statusText);
    throw new Error(txt || r.statusText);
  }
  return r.json();
}

// ── Format helpers ────────────────────────────────────────────────────────
const fmt  = (n, d = 2) => n == null ? '—' : Number(n).toFixed(d);
const fmtK = (n) => n == null ? '—' :
  Math.abs(n) >= 1e6 ? (n / 1e6).toFixed(2) + 'M' :
  Math.abs(n) >= 1e3 ? (n / 1e3).toFixed(1) + 'K' : Number(n).toFixed(2);
const colorClass = (n) => n > 0 ? 'text-green' : n < 0 ? 'text-red' : '';
const sign = (n) => n > 0 ? '+' : '';

// ── Mode badge ────────────────────────────────────────────────────────────
function updateModeBadge(mode) {
  const btn = el('mode-btn');
  btn.textContent = mode.toUpperCase();
  btn.className = `mode-btn badge ${mode === 'live' ? 'badge-live' : 'badge-paper'}`;
  // Update balance card sub-text
  const sub = el('stat-balance-sub');
  if (sub) sub.textContent = mode === 'live' ? 'Live broker account' : 'Paper trading account';
}

// ── Toggle trading mode ───────────────────────────────────────────────────
async function toggleMode() {
  const current = el('mode-btn').textContent.trim().toLowerCase();
  const newMode  = current === 'paper' ? 'live' : 'paper';

  if (newMode === 'live') {
    const ok = confirm(
      '⚠  Switch to LIVE mode?\n\n' +
      'Real orders will be placed on the broker.\n' +
      'Make sure your API keys have trading permissions and your .env is configured.'
    );
    if (!ok) return;
  }

  try {
    const data = await apiFetch(`/api/mode?mode=${newMode}`, { method: 'POST' });
    updateModeBadge(data.mode);
    toast(
      `Switched to ${data.mode.toUpperCase()} mode`,
      data.mode === 'live' ? 'error' : 'success',
    );
    await refreshAll();
  } catch (e) {
    toast('Mode switch failed: ' + e.message, 'error');
  }
}

// ── Build equity chart ────────────────────────────────────────────────────
function buildChart(history) {
  const ctx = document.getElementById('equityChart').getContext('2d');
  const labels = history.map(p => {
    const d = new Date(p.t);
    return d.getHours().toString().padStart(2,'0') + ':' +
           d.getMinutes().toString().padStart(2,'0');
  });
  const values = history.map(p => p.v);
  const isUp   = values.length < 2 || values[values.length - 1] >= values[0];
  const color  = isUp ? '#3fb950' : '#f85149';

  if (equityChart) {
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = values;
    equityChart.data.datasets[0].borderColor = color;
    equityChart.data.datasets[0].backgroundColor =
      isUp ? 'rgba(63,185,80,.08)' : 'rgba(248,81,73,.08)';
    equityChart.update('none');
    return;
  }

  equityChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Equity',
        data: values,
        borderColor: color,
        backgroundColor: isUp ? 'rgba(63,185,80,.08)' : 'rgba(248,81,73,.08)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ' $' + ctx.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2 }),
          },
        },
      },
      scales: {
        x: {
          grid: { color: '#30363d', drawTicks: false },
          ticks: { color: '#8b949e', maxTicksLimit: 8, font: { size: 11 } },
        },
        y: {
          grid: { color: '#30363d', drawTicks: false },
          ticks: {
            color: '#8b949e',
            font: { size: 11 },
            callback: v => '$' + v.toLocaleString(),
          },
        },
      },
    },
  });
}

// ── Update stat cards ─────────────────────────────────────────────────────
function updateStats(balance, trades, btResult) {
  el('stat-balance').textContent = '$' + fmtK(balance.balance);
  el('stat-balance').className   = 'stat-value ' + (balance.mode === 'live' ? 'text-green' : colorClass(balance.pnl));

  el('stat-pnl').textContent =
    balance.mode === 'live' ? 'Live' :
    (balance.pnl >= 0 ? '+' : '') + fmt(balance.pnl_pct, 1) + '%';
  el('stat-pnl').className   = 'stat-value ' + colorClass(balance.pnl);
  el('stat-pnl-abs').textContent =
    balance.mode === 'live' ? 'Real broker balance' :
    sign(balance.pnl) + '$' + fmtK(Math.abs(balance.pnl));

  const open = Array.isArray(trades) ? trades.filter(t => !t.exit_price) : [];
  el('stat-open').textContent = open.length.toString();

  if (btResult) {
    el('stat-winrate').textContent = fmt(btResult.win_rate_pct, 1) + '%';
    el('stat-winrate').className   = 'stat-value ' + (btResult.win_rate_pct > 50 ? 'text-green' : 'text-yellow');
    el('stat-maxdd').textContent   = fmt(btResult.max_drawdown_pct, 2) + '%';
    el('stat-maxdd').className     = 'stat-value ' + (btResult.max_drawdown_pct > 15 ? 'text-red' : 'text-yellow');
  }
}

// ── Signal checklist ──────────────────────────────────────────────────────
const CHECKS = [
  { key: 'structure', label: 'Market Structure (BOS/CHoCH)' },
  { key: 'sweep',     label: 'Liquidity Sweep' },
  { key: 'ob',        label: 'Order Block Found' },
  { key: 'in_ob',     label: 'Price inside OB Zone' },
  { key: 'fvg',       label: 'Fair Value Gap (FVG)' },
];

function renderSignal(signal) {
  const dirEl  = el('signal-direction');
  const entEl  = el('sig-entry');
  const slEl   = el('sig-sl');
  const tpEl   = el('sig-tp');
  const rsEl   = el('signal-reason');
  const clEl   = el('checklist');

  if (!signal) {
    dirEl.innerHTML = '<span class="signal-dir-badge dir-none">Scanning…</span>';
    entEl.textContent = '—';
    slEl.textContent  = '—';
    tpEl.textContent  = '—';
    rsEl.textContent  = 'No active setup.';
    clEl.innerHTML = CHECKS.map(c => checkItem(c.label, 'wait')).join('');
    return;
  }

  const dir = signal.direction === 'long' ? 'LONG ▲' : 'SHORT ▼';
  const cls = signal.direction === 'long' ? 'dir-long' : 'dir-short';
  dirEl.innerHTML = `<span class="signal-dir-badge ${cls}">${dir}</span>`;

  entEl.textContent = fmt(signal.entry);
  slEl.textContent  = fmt(signal.stop_loss);
  tpEl.textContent  = fmt(signal.take_profit);
  rsEl.textContent  = signal.reason || '';

  clEl.innerHTML = CHECKS.map(c => checkItem(c.label, 'pass')).join('');
}

function checkItem(label, state) {
  const icon = state === 'pass' ? '✓' : state === 'fail' ? '✗' : '·';
  return `
    <div class="check-item">
      <div class="check-icon ${state}">${icon}</div>
      <span class="check-label ${state}">${label}</span>
    </div>`;
}

// ── Backtest results ──────────────────────────────────────────────────────
function renderBacktestResults(m) {
  if (!m || m.error) {
    el('bt-result-grid').innerHTML =
      `<div class="bt-metric" style="grid-column:1/-1">
         <div class="bt-metric-label">Status</div>
         <div class="bt-metric-val text-red">${m ? m.error : 'Not run yet'}</div>
       </div>`;
    return;
  }

  const metrics = [
    { label: 'Win Rate',       val: fmt(m.win_rate_pct, 1) + '%',
      cls: m.win_rate_pct > 50 ? 'text-green' : 'text-yellow' },
    { label: 'Profit Factor',  val: fmt(m.profit_factor, 2),
      cls: m.profit_factor > 1 ? 'text-green' : 'text-red' },
    { label: 'Max Drawdown',   val: fmt(m.max_drawdown_pct, 2) + '%',
      cls: m.max_drawdown_pct > 15 ? 'text-red' : 'text-yellow' },
    { label: 'Expectancy',     val: (m.expectancy_r > 0 ? '+' : '') + fmt(m.expectancy_r, 3) + 'R',
      cls: m.expectancy_r > 0 ? 'text-green' : 'text-red' },
    { label: 'Net Return',     val: (m.net_profit_pct > 0 ? '+' : '') + fmt(m.net_profit_pct, 2) + '%',
      cls: m.net_profit_pct > 0 ? 'text-green' : 'text-red' },
    { label: 'Sharpe',         val: fmt(m.sharpe_per_trade, 2), cls: '' },
    { label: 'Consec. Losses', val: m.max_consec_losses, cls: '' },
    { label: 'Trades',         val: m.total_trades, cls: '' },
  ];

  el('bt-result-grid').innerHTML = metrics.map(m_ =>
    `<div class="bt-metric">
       <div class="bt-metric-label">${m_.label}</div>
       <div class="bt-metric-val ${m_.cls}">${m_.val}</div>
     </div>`
  ).join('');

  const extra = document.createElement('div');
  extra.style.cssText = 'padding: 0 16px 10px; font-size: 12px; color: var(--muted);';
  extra.innerHTML =
    `Signal rate: <b style="color:var(--text)">${fmt(m.signal_rate_pct,1)}%</b> &nbsp;|&nbsp; ` +
    `Break-even WR: <b style="color:var(--text)">${fmt(m.breakeven_wr_pct,1)}%</b>` +
    (m.symbol ? ` &nbsp;|&nbsp; Symbol: <b style="color:var(--blue)">${m.symbol}</b>` : '') +
    (m.broker  ? ` &nbsp;|&nbsp; Broker: <b style="color:var(--purple)">${m.broker}</b>` : '');
  const grid = el('bt-result-grid');
  const old = grid.nextElementSibling;
  if (old && old.classList.contains('bt-extra')) old.remove();
  extra.classList.add('bt-extra');
  grid.after(extra);
}

// ── Trade log table ────────────────────────────────────────────────────────
function renderTrades(trades, mode) {
  const tbody = el('trades-tbody');
  if (!trades || !trades.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state">No ${mode === 'live' ? 'live' : 'paper'} trades yet</td></tr>`;
    return;
  }

  const rows = [...trades].slice(0, 100).map(t => {
    const dirCls = t.direction === 'long' ? 'dir-long-txt' : 'dir-short-txt';
    const pnlCls = t.pnl > 0 ? 'text-green' : t.pnl < 0 ? 'text-red' : '';

    let outCls, outLbl;
    if (mode === 'live') {
      outCls = t.outcome === 'placed' ? 'outcome-tp' : 'outcome-sl';
      outLbl = t.outcome === 'cancelled' ? '✗ Cancelled' : '✓ Placed';
    } else {
      outCls = t.outcome === 'tp_hit' ? 'outcome-tp' : t.outcome === 'sl_hit' ? 'outcome-sl' : 'outcome-to';
      outLbl = t.outcome === 'tp_hit' ? 'TP ✓' : t.outcome === 'sl_hit' ? 'SL ✗' : '—';
    }

    return `<tr>
      <td>${t.order_id || '—'}</td>
      <td class="${dirCls}">${(t.direction || '').toUpperCase()}</td>
      <td>${fmt(t.entry)}</td>
      <td>${fmt(t.exit_price)}</td>
      <td class="${pnlCls}">${t.pnl != null ? sign(t.pnl) + '$' + fmt(Math.abs(t.pnl), 2) : '—'}</td>
      <td>${t.quantity || '—'}</td>
      <td class="${outCls}">${outLbl}</td>
    </tr>`;
  }).join('');
  tbody.innerHTML = rows;
}

// ── ML confidence bar in signal panel ────────────────────────────────────
function renderMLConfidence(ml) {
  const row = el('ml-confidence-row');
  if (!ml) { row.style.display = 'none'; return; }
  row.style.display = '';

  const pct   = Math.round(ml.confidence * 100);
  const bar   = el('ml-confidence-bar');
  const pctEl = el('ml-confidence-pct');
  const vrd   = el('ml-confidence-verdict');

  bar.style.width = pct + '%';
  bar.style.background = pct >= 70 ? 'var(--green)' : pct >= 60 ? 'var(--yellow)' : 'var(--red)';
  pctEl.textContent = pct + '%';
  pctEl.className = 'ml-confidence-pct ' + (pct >= 60 ? 'text-green' : 'text-red');

  if (ml.approved) {
    vrd.textContent = '✓ Approved';
    vrd.className = 'ml-confidence-verdict text-green';
  } else {
    vrd.textContent = '✗ Blocked';
    vrd.className = 'ml-confidence-verdict text-red';
  }
}

// ── ML Status panel ───────────────────────────────────────────────────────
function renderMLStatus(s) {
  if (!s || s.status === 'not_loaded') {
    el('ml-status-line').textContent = 'Model not loaded yet. Click Train to initialise.';
    return;
  }

  const nameMap = { random_forest: 'Random Forest', xgboost: 'XGBoost', none: 'None' };
  el('ml-model-name').textContent = nameMap[s.model_name] || s.model_name || '—';
  el('ml-version').textContent    = s.version ? 'v' + s.version : '—';
  el('ml-f1').textContent         = s.f1   != null ? (s.f1   * 100).toFixed(1) + '%' : '—';
  el('ml-auc').textContent        = s.roc_auc != null ? (s.roc_auc * 100).toFixed(1) + '%' : '—';
  el('ml-ntrain').textContent     = s.n_train ? s.n_train.toLocaleString() + ' samples' : '—';

  const perf = s.performance || {};
  el('ml-live-acc').textContent = perf.accuracy != null
    ? (perf.accuracy * 100).toFixed(1) + '%  (n=' + (perf.n || 0) + ')'
    : '—';

  // Colour F1
  const f1El = el('ml-f1');
  const f1v  = s.f1 != null ? s.f1 * 100 : null;
  f1El.className = 'ml-metric-val ' + (f1v == null ? '' : f1v >= 60 ? 'text-green' : f1v >= 50 ? 'text-yellow' : 'text-red');

  const ts = s.trained_at ? new Date(s.trained_at).toLocaleString() : '';
  el('ml-status-line').textContent = s.status === 'ready'
    ? `Model ready · threshold ${(s.threshold * 100).toFixed(0)}%${ts ? '  ·  trained ' + ts : ''}`
    : 'No model — click Train';

  // Sync slider to server threshold
  if (s.threshold) {
    const slider = el('ml-threshold-input');
    const tVal   = Math.round(s.threshold * 100);
    slider.value = tVal;
    el('ml-threshold-val').textContent = (tVal / 100).toFixed(2);
  }
}

function renderFeatureImportances(metrics) {
  const listEl = el('ml-features-list');
  if (!metrics) { listEl.innerHTML = '<div class="empty-state">Train a model to see features</div>'; return; }
  const imps = metrics.feature_importances || [];
  if (!imps.length) { listEl.innerHTML = '<div class="empty-state">No feature data</div>'; return; }

  const top = imps.slice(0, 10);
  const maxImp = top[0]?.importance || 1;
  listEl.innerHTML = top.map(f => {
    const pct = Math.round((f.importance / maxImp) * 100);
    return `
      <div class="ml-feat-row">
        <div class="ml-feat-name">${f.name}</div>
        <div class="ml-feat-bar-wrap">
          <div class="ml-feat-bar" style="width:${pct}%"></div>
        </div>
        <div class="ml-feat-val">${(f.importance * 100).toFixed(2)}%</div>
      </div>`;
  }).join('');
}

async function doMLTrain() {
  const btn  = el('ml-train-btn');
  const spin = el('ml-spin');
  btn.disabled = true;
  spin.classList.add('show');
  toast('Training model — this may take 10–30 seconds…', 'info', 8000);
  try {
    const result = await apiFetch('/api/ml/train', { method: 'POST' });
    toast(
      result.error
        ? 'Training failed: ' + result.error
        : `Model v${result.version || '?'} trained  F1=${result.new_f1 != null ? (result.new_f1*100).toFixed(1)+'%' : '?'}`,
      result.error ? 'error' : 'success',
    );
    await refreshMLStatus();
  } catch (e) {
    toast('Train failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    spin.classList.remove('show');
  }
}

async function refreshMLStatus() {
  try {
    const s = await apiFetch('/api/ml/status');
    renderMLStatus(s);
    // Render feature importances from current metrics
    if (s.status === 'ready') {
      try {
        const vers = await apiFetch('/api/ml/versions');
        const latest = (vers.versions || []).slice(-1)[0];
        if (latest) renderFeatureImportances(latest.metrics);
      } catch (_) {}
    }
  } catch (_) {}
}

// ── Scan button ──────────────────────────────────────────────────────────
async function doScan() {
  const broker = el('broker-select').value;
  const symbol = (el('symbol-input').value || '').trim();
  const btn    = el('scan-btn');
  const spin   = el('scan-spin');
  btn.disabled = true;
  spin.classList.add('show');

  try {
    const params = new URLSearchParams({ broker_name: broker });
    if (symbol) params.set('symbol', symbol);
    const data = await apiFetch(`/api/scan?${params}`, { method: 'POST' });

    if (data.signal) {
      renderSignal(data.signal);
      renderMLConfidence(data.ml || data.signal?.ml);
      const mlTag  = data.ml ? ` [AI ${Math.round(data.ml.confidence * 100)}%]` : '';
      const modeLabel = data.mode === 'live' ? ' [LIVE ORDER]' : ' [Paper]';
      toast(
        `Signal: ${data.signal.direction.toUpperCase()} @ ${data.signal.entry}${mlTag}${modeLabel}`,
        data.ml && !data.ml.approved ? 'error' : 'success',
      );
    } else if (data.message && data.message.includes('blocked')) {
      toast('ML blocked the trade: ' + data.message, 'error');
      if (data.signal) renderSignal(data.signal);
      renderMLConfidence(data.ml);
    } else {
      renderSignal(null);
      renderMLConfidence(null);
      toast('No ICT setup found on current data.', 'info');
    }
    await refreshAll();
  } catch (e) {
    toast('Scan failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    spin.classList.remove('show');
  }
}

// ── Run backtest ──────────────────────────────────────────────────────────
async function doBacktest() {
  const btn    = el('bt-run-btn');
  const spin   = el('bt-spin');
  const status = el('bt-status');
  btn.disabled = true;
  spin.classList.add('show');
  status.textContent = 'Running…';

  const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;

  try {
    let data;
    if (activeTab === 'real') {
      const broker   = el('bt-real-broker').value;
      const symbol   = el('bt-real-symbol').value;
      const interval = el('bt-real-interval').value;
      const rr       = el('bt-rr').value;
      const risk     = el('bt-risk').value;
      data = await apiFetch(
        `/api/backtest/real?broker_name=${encodeURIComponent(broker)}&symbol=${encodeURIComponent(symbol)}&interval=${interval}&rr=${rr}&risk=${risk}`,
        { method: 'POST' },
      );
    } else {
      const trials   = el('bt-trials').value;
      const win_rate = el('bt-winrate').value;
      const rr       = el('bt-rr').value;
      const risk     = el('bt-risk').value;
      data = await apiFetch('/api/backtest/synthetic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trials: +trials, win_rate: +win_rate / 100, rr: +rr, risk: +risk }),
      });
    }

    renderBacktestResults(data);
    const [bal, tradesResp] = await Promise.all([
      apiFetch('/api/balance'),
      apiFetch('/api/trades'),
    ]);
    updateStats(bal, tradesResp.trades || [], data);
    status.textContent = data.error
      ? '⚠ ' + data.error
      : `Done · ${data.total_trades || 0} trades`;
    if (!data.error) toast('Backtest complete!', 'success');
    else toast(data.error, 'error');
  } catch (e) {
    status.textContent = '⚠ ' + e.message;
    toast('Backtest failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    spin.classList.remove('show');
  }
}

// ── Tab switching ─────────────────────────────────────────────────────────
function setTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  const realOpts = el('bt-real-opts');
  realOpts.classList.toggle('show', tab === 'real');
  el('bt-trials-row').style.display  = tab === 'synthetic' ? '' : 'none';
  el('bt-winrate-row').style.display = tab === 'synthetic' ? '' : 'none';
}

// ── Refresh all data ──────────────────────────────────────────────────────
async function refreshAll() {
  try {
    const [status, balance, positions, trades, equity, lastBt] = await Promise.all([
      apiFetch('/api/status'),
      apiFetch('/api/balance'),
      apiFetch('/api/positions'),
      apiFetch('/api/trades'),
      apiFetch('/api/equity'),
      apiFetch('/api/backtest/last'),
    ]);

    // Sync mode badge and broker select to server state
    updateModeBadge(status.mode);
    const brokerSel = el('broker-select');
    if (status.broker && brokerSel.value !== status.broker) {
      brokerSel.value = status.broker;
    }

    updateStats(balance, trades.trades || [], lastBt.results);
    buildChart(equity.history || []);
    renderTrades(trades.trades || [], trades.mode || status.mode);
    el('stat-open').textContent = (positions.positions || []).length;

    if (lastBt.results) renderBacktestResults(lastBt.results);
  } catch (e) {
    console.error('Refresh error:', e);
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);
    if (msg.event === 'snapshot') {
      buildChart(msg.data.equity || []);
      if (msg.data.signal) renderSignal(msg.data.signal);
      if (msg.data.mode)   updateModeBadge(msg.data.mode);
    } else if (msg.event === 'signal') {
      renderSignal(msg.data);
      refreshAll();
    } else if (msg.event === 'balance') {
      refreshAll();
    } else if (msg.event === 'backtest_done') {
      renderBacktestResults(msg.data);
    } else if (msg.event === 'mode_changed') {
      updateModeBadge(msg.data.mode);
    } else if (msg.event === 'training_progress') {
      renderTrainingProgress(msg.data);
    } else if (msg.event === 'training_complete') {
      renderTrainingProgress(msg.data);
      toast('Training complete! Model has been updated.', 'success');
      setTrainingUI(false);
      refreshMLStatus();
    }
  };

  ws.onclose = () => {
    setTimeout(connectWS, 5000);
  };
}

// ── Training loop controls ────────────────────────────────────────────────
let trainingPollTimer = null;

function setTrainingUI(running) {
  el('train-start-btn').disabled = running;
  el('train-stop-btn').disabled  = !running;
  if (running) {
    el('train-loop-spin').classList.add('show');
    el('train-status-val').textContent = 'Running';
    el('train-status-val').className = 'train-stat-val text-green';
  } else {
    el('train-loop-spin').classList.remove('show');
    el('train-status-val').textContent = 'Idle';
    el('train-status-val').className = 'train-stat-val';
  }
}

function renderTrainingProgress(s) {
  if (!s) return;
  setTrainingUI(s.running);

  el('train-epoch').textContent   = s.epoch || 0;
  el('train-windows').textContent = (s.total_windows || 0).toLocaleString();
  el('train-signals').textContent = (s.total_signals || 0).toLocaleString();
  el('train-trades').textContent  = (s.total_trades || 0).toLocaleString();
  el('train-retrains').textContent = s.retrains_done || 0;

  // Win rate
  const wr = s.win_rate;
  el('train-winrate').textContent = wr != null ? wr + '%' : '—';
  el('train-winrate').className = 'train-stat-val ' + (wr > 50 ? 'text-green' : wr > 40 ? 'text-yellow' : wr != null ? 'text-red' : '');

  // F1
  const f1 = s.last_f1;
  el('train-f1').textContent = f1 ? (f1 * 100).toFixed(1) + '%' : '—';
  el('train-f1').className = 'train-stat-val ' + (f1 > 0.6 ? 'text-green' : f1 > 0.5 ? 'text-yellow' : f1 ? 'text-red' : '');

  // Elapsed
  el('train-elapsed').textContent = s.elapsed_hours != null
    ? s.elapsed_hours.toFixed(1) + 'h / ' + (s.duration_hours || 0) + 'h'
    : '—';

  // Progress bar
  const pct = s.duration_hours > 0 && s.elapsed_hours != null
    ? Math.min(100, (s.elapsed_hours / s.duration_hours) * 100)
    : 0;
  el('training-progress-bar').style.width = pct + '%';

  // Error
  const errEl = el('train-error');
  if (s.last_error) {
    errEl.style.display = '';
    errEl.textContent = '⚠ ' + s.last_error;
  } else {
    errEl.style.display = 'none';
  }
}

async function startTraining() {
  const duration = el('train-duration').value;
  const symbol   = el('train-symbol').value;
  const interval = el('train-interval').value;
  const fetchSec = el('train-fetch-sec').value;
  const retrain  = el('train-retrain-every').value;

  el('train-start-btn').disabled = true;
  el('train-loop-spin').classList.add('show');

  try {
    const params = new URLSearchParams({
      duration_hours: duration,
      symbol: symbol,
      interval: interval,
      fetch_interval_sec: fetchSec,
      retrain_every: retrain,
      broker_name: 'delta',
    });
    const data = await apiFetch(`/api/training/start?${params}`, { method: 'POST' });
    toast(`Training started (${duration}h, ${symbol}, ${interval})`, 'success');
    setTrainingUI(true);
    if (data.status) renderTrainingProgress(data.status);
    // Start polling training status
    trainingPollTimer = setInterval(pollTrainingStatus, 10_000);
  } catch (e) {
    toast('Failed to start training: ' + e.message, 'error');
    el('train-start-btn').disabled = false;
    el('train-loop-spin').classList.remove('show');
  }
}

async function stopTraining() {
  try {
    await apiFetch('/api/training/stop', { method: 'POST' });
    toast('Training stopping…', 'info');
    setTrainingUI(false);
    if (trainingPollTimer) { clearInterval(trainingPollTimer); trainingPollTimer = null; }
  } catch (e) {
    toast('Stop failed: ' + e.message, 'error');
  }
}

async function pollTrainingStatus() {
  try {
    const s = await apiFetch('/api/training/status');
    renderTrainingProgress(s);
    if (!s.running && trainingPollTimer) {
      clearInterval(trainingPollTimer);
      trainingPollTimer = null;
    }
  } catch (_) {}
}

// ── Mistake Analysis ──────────────────────────────────────────────────────
async function loadMistakeReport() {
  const spin    = el('mistake-spin');
  const content = el('mistake-content');
  spin.classList.add('show');

  try {
    const data = await apiFetch('/api/ml/mistakes');

    if (data.error) {
      content.innerHTML = `<div class="empty-state">${data.error}</div>`;
      return;
    }

    let html = '';

    // Top insights
    if (data.top_insights && data.top_insights.length) {
      html += '<div class="mistake-section">';
      html += '<div class="mistake-section-title">🎯 Key Insights</div>';
      data.top_insights.forEach(insight => {
        html += `<div class="mistake-insight">${insight}</div>`;
      });
      html += '</div>';
    }

    // Summary stats
    if (data.summary) {
      const s = data.summary;
      html += '<div class="mistake-section">';
      html += '<div class="mistake-section-title">📈 Summary</div>';
      html += '<div class="mistake-stats">';
      html += `<span>Total: <b>${s.total_trades}</b></span>`;
      html += `<span>Wins: <b class="text-green">${s.wins}</b></span>`;
      html += `<span>Losses: <b class="text-red">${s.losses}</b></span>`;
      html += `<span>Win Rate: <b class="${s.win_rate > 0.5 ? 'text-green' : 'text-red'}">${(s.win_rate*100).toFixed(1)}%</b></span>`;
      html += '</div></div>';
    }

    // Feature insights
    if (data.feature_insights && data.feature_insights.length) {
      html += '<div class="mistake-section">';
      html += '<div class="mistake-section-title">🔬 Feature Analysis</div>';
      data.feature_insights.forEach(fi => {
        const worse = fi.high_loss_rate > fi.low_loss_rate ? 'high' : 'low';
        const worseRate = worse === 'high' ? fi.high_loss_rate : fi.low_loss_rate;
        html += `<div class="mistake-feature">`;
        html += `  <span class="mistake-feat-name">${fi.feature}</span>`;
        html += `  <span class="mistake-feat-effect">effect: ${fi.effect_size.toFixed(2)}</span>`;
        html += `  <span class="mistake-feat-rate text-red">${(worseRate*100).toFixed(0)}% loss rate when ${worse}</span>`;
        html += `</div>`;
      });
      html += '</div>';
    }

    // Direction analysis
    if (data.direction && Object.keys(data.direction).length) {
      html += '<div class="mistake-section">';
      html += '<div class="mistake-section-title">↕ Direction Analysis</div>';
      for (const [dir, info] of Object.entries(data.direction)) {
        const wrCls = info.win_rate > 0.5 ? 'text-green' : 'text-red';
        html += `<div class="mistake-dir"><b>${dir.toUpperCase()}</b>: `;
        html += `<span class="${wrCls}">${(info.win_rate*100).toFixed(0)}%</span> win rate `;
        html += `(${info.trades} trades)</div>`;
      }
      html += '</div>';
    }

    // Streaks
    if (data.streaks && data.streaks.max_loss_streak) {
      html += '<div class="mistake-section">';
      html += '<div class="mistake-section-title">📉 Streaks</div>';
      html += `<div class="mistake-stats">`;
      html += `<span>Max Loss Streak: <b class="text-red">${data.streaks.max_loss_streak}</b></span>`;
      html += `<span>Avg Loss Streak: <b>${data.streaks.avg_loss_streak}</b></span>`;
      html += `<span>Max Win Streak: <b class="text-green">${data.streaks.max_win_streak}</b></span>`;
      html += `</div></div>`;
    }

    content.innerHTML = html || '<div class="empty-state">No mistake patterns found yet</div>';
  } catch (e) {
    content.innerHTML = `<div class="empty-state">Failed to load: ${e.message}</div>`;
  } finally {
    spin.classList.remove('show');
  }
}

// ── DOM helpers ───────────────────────────────────────────────────────────
function el(id) { return document.getElementById(id); }

// ── Init ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  // Buttons
  el('scan-btn').addEventListener('click', doScan);
  el('bt-run-btn').addEventListener('click', doBacktest);
  el('mode-btn').addEventListener('click', toggleMode);

  // Tabs
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.addEventListener('click', () => setTab(b.dataset.tab))
  );

  // Broker change → update symbol + notify server
  el('broker-select').addEventListener('change', async () => {
    const broker = el('broker-select').value;
    el('symbol-input').value = broker === 'upstox' ? 'NSE_FO|NIFTY25MARFUT' : 'BTCUSD';
    try { await apiFetch(`/api/broker?name=${broker}`, { method: 'POST' }); } catch (_) {}
    await refreshAll();
  });

  // Real-data backtest broker change → update symbol
  el('bt-real-broker').addEventListener('change', () => {
    const broker = el('bt-real-broker').value;
    el('bt-real-symbol').value = broker === 'upstox' ? 'NSE_INDEX|Nifty 50' : 'BTCUSD';
  });

  // ML — train button
  el('ml-train-btn').addEventListener('click', doMLTrain);

  // ML — threshold slider live display
  el('ml-threshold-input').addEventListener('input', () => {
    const v = el('ml-threshold-input').value;
    el('ml-threshold-val').textContent = (v / 100).toFixed(2);
  });

  // ML — set threshold
  el('ml-threshold-set').addEventListener('click', async () => {
    const v = parseFloat(el('ml-threshold-input').value) / 100;
    try {
      await apiFetch(`/api/ml/threshold?value=${v}`, { method: 'POST' });
      toast(`Threshold set to ${(v * 100).toFixed(0)}%`, 'success');
      await refreshMLStatus();
    } catch (e) {
      toast('Failed: ' + e.message, 'error');
    }
  });

  // Training loop controls
  el('train-start-btn').addEventListener('click', startTraining);
  el('train-stop-btn').addEventListener('click', stopTraining);
  el('mistake-refresh-btn').addEventListener('click', loadMistakeReport);

  // Initial render placeholders
  renderSignal(null);
  renderBacktestResults(null);
  renderTrades([], 'paper');

  // Load all data (this will sync mode badge from server)
  await refreshAll();

  // Load ML status
  await refreshMLStatus();

  // Check if training is already running
  pollTrainingStatus();

  // Load mistake analysis
  loadMistakeReport();

  // Connect WebSocket
  try { connectWS(); } catch (e) { /* WS optional */ }

  // Poll every 30s
  pollTimer = setInterval(refreshAll, 30_000);
  // Poll ML status every 60s
  setInterval(refreshMLStatus, 60_000);

  // Load last signal
  try {
    const sig = await apiFetch('/api/signal');
    if (sig.signal) {
      renderSignal(sig.signal);
      if (sig.signal.ml) renderMLConfidence(sig.signal.ml);
    }
  } catch (_) {}
});
