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
  // Balance
  el('stat-balance').textContent = '$' + fmtK(balance.balance);
  el('stat-balance').className   = 'stat-value ' + colorClass(balance.pnl);

  // PnL
  el('stat-pnl').textContent =
    (balance.pnl >= 0 ? '+' : '') + fmt(balance.pnl_pct, 1) + '%';
  el('stat-pnl').className   = 'stat-value ' + colorClass(balance.pnl);
  el('stat-pnl-abs').textContent =
    sign(balance.pnl) + '$' + fmtK(Math.abs(balance.pnl));

  // Open trades
  const open = trades.filter ? trades.filter(t => !t.exit_price) : [];
  el('stat-open').textContent = (typeof trades === 'number' ? trades : open.length).toString();

  // Backtest metrics (win rate + max DD)
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
    // All checks "waiting"
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

  // Mark all checks as passing (signal only generates when all pass)
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
    { label: 'Trades',         val: m.total_trades,
      cls: '' },
  ];

  el('bt-result-grid').innerHTML = metrics.map(m_ =>
    `<div class="bt-metric">
       <div class="bt-metric-label">${m_.label}</div>
       <div class="bt-metric-val ${m_.cls}">${m_.val}</div>
     </div>`
  ).join('');

  // extra row: signal rate + breakeven
  const extra = document.createElement('div');
  extra.style.cssText = 'padding: 0 16px 10px; font-size: 12px; color: var(--muted);';
  extra.innerHTML =
    `Signal rate: <b style="color:var(--text)">${fmt(m.signal_rate_pct,1)}%</b> &nbsp;|&nbsp; ` +
    `Break-even WR: <b style="color:var(--text)">${fmt(m.breakeven_wr_pct,1)}%</b> &nbsp;|&nbsp; ` +
    (m.symbol ? `Symbol: <b style="color:var(--blue)">${m.symbol}</b> &nbsp;|&nbsp; ` : '') +
    (m.broker  ? `Broker: <b style="color:var(--purple)">${m.broker}</b>` : '');
  const grid = el('bt-result-grid');
  // remove old extra if exists
  const old = grid.nextElementSibling;
  if (old && old.classList.contains('bt-extra')) old.remove();
  extra.classList.add('bt-extra');
  grid.after(extra);
}

// ── Trade log table ────────────────────────────────────────────────────────
function renderTrades(trades) {
  const tbody = el('trades-tbody');
  if (!trades || !trades.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No trades yet</td></tr>';
    return;
  }

  const rows = [...trades].reverse().slice(0, 100).map(t => {
    const dirCls = t.direction === 'long' ? 'dir-long-txt' : 'dir-short-txt';
    const pnlCls = t.pnl > 0 ? 'text-green' : t.pnl < 0 ? 'text-red' : '';
    const outCls = t.outcome === 'tp_hit' ? 'outcome-tp' : t.outcome === 'sl_hit' ? 'outcome-sl' : 'outcome-to';
    const outLbl = t.outcome === 'tp_hit' ? 'TP ✓' : t.outcome === 'sl_hit' ? 'SL ✗' : '—';
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

// ── Scan button ──────────────────────────────────────────────────────────
async function doScan() {
  const broker = el('broker-select').value;
  const btn    = el('scan-btn');
  const spin   = el('scan-spin');
  btn.disabled = true;
  spin.classList.add('show');

  try {
    const broker_q = encodeURIComponent(broker);
    const data = await apiFetch(`/api/scan?broker=${broker_q}`, { method: 'POST' });
    if (data.signal) {
      renderSignal(data.signal);
      toast(`Signal: ${data.signal.direction.toUpperCase()} @ ${data.signal.entry}`, 'success');
    } else {
      renderSignal(null);
      toast('No ICT setup found on current data.', 'info');
    }
    // refresh balance + trades
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
        body: JSON.stringify({ trials: +trials, win_rate: +win_rate, rr: +rr, risk: +risk }),
      });
    }

    renderBacktestResults(data);
    updateStats(
      await apiFetch('/api/balance'),
      (await apiFetch('/api/trades')).trades,
      data,
    );
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
  el('bt-trials-row').style.display = tab === 'synthetic' ? '' : 'none';
  el('bt-winrate-row').style.display = tab === 'synthetic' ? '' : 'none';
}

// ── Refresh all data ──────────────────────────────────────────────────────
async function refreshAll() {
  try {
    const [balance, positions, trades, equity, lastBt] = await Promise.all([
      apiFetch('/api/balance'),
      apiFetch('/api/positions'),
      apiFetch('/api/trades'),
      apiFetch('/api/equity'),
      apiFetch('/api/backtest/last'),
    ]);

    updateStats(balance, trades.trades || [], lastBt.results);
    buildChart(equity.history || []);
    renderTrades(trades.trades || []);
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
      // initial data burst
      buildChart(msg.data.equity || []);
      if (msg.data.signal) renderSignal(msg.data.signal);
    } else if (msg.event === 'signal') {
      renderSignal(msg.data);
      refreshAll();
    } else if (msg.event === 'balance') {
      refreshAll();
    } else if (msg.event === 'backtest_done') {
      renderBacktestResults(msg.data);
    }
  };

  ws.onclose = () => {
    setTimeout(connectWS, 5000);   // auto-reconnect
  };
}

// ── DOM helpers ───────────────────────────────────────────────────────────
function el(id) { return document.getElementById(id); }

// ── Init ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  // Wire buttons
  el('scan-btn').addEventListener('click', doScan);
  el('bt-run-btn').addEventListener('click', doBacktest);

  document.querySelectorAll('.tab-btn').forEach(b =>
    b.addEventListener('click', () => setTab(b.dataset.tab))
  );

  // Initial render placeholders
  renderSignal(null);
  renderBacktestResults(null);
  renderTrades([]);

  // Load data
  await refreshAll();

  // Connect WebSocket
  try { connectWS(); } catch (e) { /* WS optional */ }

  // Poll every 30s
  pollTimer = setInterval(refreshAll, 30_000);

  // Load last signal
  try {
    const sig = await apiFetch('/api/signal');
    if (sig.signal) renderSignal(sig.signal);
  } catch (_) {}
});
