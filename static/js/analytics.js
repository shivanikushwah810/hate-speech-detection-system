/* ═══════════════════════════════════════════════════════════════════════
   analytics.js — small helper library shared by the 3 new pages.
   Charts are plain HTML/CSS (no Chart.js / CDN), so they work offline.
   Every dynamic string goes through HS.esc() to prevent HTML injection.
   ═══════════════════════════════════════════════════════════════════════ */
const HS = (() => {
  const CLASS_COLORS = { 'Hate Speech': '#ff1744', 'Offensive': '#ff9100', 'Normal': '#00e676' };
  const CLASS_ORDER = ['Hate Speech', 'Offensive', 'Normal'];
  const SERIES = ['#6c63ff', '#00bcd4', '#ff9100', '#ff6584', '#00e676'];

  const esc = s => String(s ?? '').replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (x, d = 1) => (Number(x) * 100).toFixed(d) + '%';
  const f4  = x => Number(x).toFixed(4);

  /** fetch wrapper: never throws; returns {ok,status,data,error} */
  async function api(url, opts) {
    try {
      const r = await fetch(url, opts);
      let data = null;
      try { data = await r.json(); } catch (_) { /* non-JSON body */ }
      return { ok: r.ok, status: r.status, data, error: r.ok ? null : (data && data.error) || `HTTP ${r.status}` };
    } catch (e) {
      return { ok: false, status: 0, data: null, error: 'Could not reach the server: ' + e.message };
    }
  }

  function banner(type, html) { return `<div class="banner ${type}">${html}</div>`; }

  /** after inserting bars into the DOM, animate them to their target width */
  function animateBars(root) {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      (root || document).querySelectorAll('[data-w]').forEach(el => { el.style.width = el.dataset.w + '%'; });
    }));
  }

  const sortedNames = models => Object.keys(models).sort((a, b) => (models[a].order ?? 0) - (models[b].order ?? 0));

  /* ── metric tiles ─────────────────────────────────────────────────── */
  function metricTiles(m) {
    const tile = (label, val, sub, primary) =>
      `<div class="metric-tile${primary ? ' primary' : ''}"><div class="label">${label}</div>
       <div class="value">${pct(val, 2)}</div>${sub ? `<div class="sub">${sub}</div>` : ''}</div>`;
    return `<div class="grid-auto">
      ${tile('Accuracy', m.accuracy, `on ${m.n_test.toLocaleString()} test texts`)}
      ${tile('Precision (macro)', m.macro_precision, 'weighted: ' + pct(m.weighted_precision, 2))}
      ${tile('Recall (macro)', m.macro_recall, 'weighted: ' + pct(m.weighted_recall, 2))}
      ${tile('Macro F1-score', m.macro_f1, 'every class counts equally', true)}
      ${tile('Weighted F1-score', m.weighted_f1, 'weighted by class size')}
    </div>`;
  }

  /* ── confusion matrix (counts + row %) ────────────────────────────── */
  function confusionMatrix(cm, labels) {
    const head = labels.map(l => `<th>${esc(l)}</th>`).join('');
    const rows = cm.map((row, i) => {
      const total = row.reduce((a, b) => a + b, 0) || 1;
      const cells = row.map((v, j) => {
        const share = v / total;
        const color = i === j ? '0,230,118' : '255,23,68';
        const alpha = (0.08 + share * (i === j ? 0.55 : 0.9)).toFixed(2);
        return `<td style="background:rgba(${color},${alpha})" title="${esc(labels[i])} predicted as ${esc(labels[j])}">
                  <span class="n">${v.toLocaleString()}</span><span class="pct">${pct(share, 1)}</span></td>`;
      }).join('');
      return `<tr><th class="row-head">${esc(labels[i])}</th>${cells}</tr>`;
    }).join('');
    return `<div class="cm-axis">Predicted label →</div>
      <div class="cm-wrap"><table class="cm"><tr><th></th>${head}</tr>${rows}</table></div>
      <div class="cm-axis">↑ True label (rows) · % = share of that true class</div>`;
  }

  /* ── per-class precision / recall / F1 ────────────────────────────── */
  function perClassBars(pc) {
    const metrics = [['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1-score']];
    const legend = `<div class="legend">${metrics.map((m, i) => `<span><i style="background:${SERIES[i]}"></i>${m[1]}</span>`).join('')}</div>`;
    const groups = CLASS_ORDER.filter(c => pc[c]).map(cls => {
      const bars = metrics.map((m, i) => hbar(m[1], pc[cls][m[0]], SERIES[i])).join('');
      return `<div class="chart-group"><div class="group-title">${esc(cls)} · ${pc[cls].support.toLocaleString()} test texts</div>${bars}</div>`;
    }).join('');
    return legend + groups;
  }

  function hbar(label, value, color, valueText) {
    const w = Math.max(0, Math.min(100, value * 100));
    return `<div class="hbar"><div class="top"><span>${esc(label)}</span><span>${valueText ?? pct(value, 2)}</span></div>
      <div class="track"><div class="fill" style="background:${color}" data-w="${w.toFixed(2)}"></div></div></div>`;
  }

  /* ── comparison: grouped bars (one group per metric) ──────────────── */
  function groupedBars(models, metricDefs) {
    const names = sortedNames(models);
    const legend = `<div class="legend">${names.map((n, i) =>
      `<span><i style="background:${SERIES[i % SERIES.length]}"></i>${esc(n)}</span>`).join('')}</div>`;
    const groups = metricDefs.map(md => {
      const bars = names.map((n, i) => hbar(n, models[n].metrics[md.key], SERIES[i % SERIES.length])).join('');
      return `<div class="chart-group"><div class="group-title">${esc(md.label)}</div>${bars}</div>`;
    }).join('');
    return legend + groups;
  }

  /** training-time bars on a LOG scale (times differ by 1000x) */
  function timeBars(models, field) {
    const names = sortedNames(models);
    const vals = names.map(n => models[n][field]);
    const max = Math.max(...vals, 0.001);
    const scale = t => 100 * Math.log10(1 + t * 1000) / Math.log10(1 + max * 1000);
    return names.map((n, i) => {
      const t = models[n][field];
      const txt = t < 1 ? (t * 1000).toFixed(0) + ' ms' : t.toFixed(2) + ' s';
      return `<div class="hbar"><div class="top"><span>${esc(n)}</span><span>${txt}</span></div>
        <div class="track"><div class="fill" style="background:${SERIES[i % SERIES.length]}" data-w="${Math.max(1.5, scale(t)).toFixed(2)}"></div></div></div>`;
    }).join('');
  }

  /* ── XAI: diverging bar chart of feature contributions ────────────── */
  function divergingBars(view) {
    const rows = [...view.positive, ...view.negative].sort((a, b) => b.contribution - a.contribution);
    if (!rows.length) {
      return `<div class="empty-state"><div class="big">🤷</div>No word in this text had a non-zero weight for this view.</div>`;
    }
    const maxAbs = Math.max(...rows.map(r => Math.abs(r.contribution)), 1e-9);
    return rows.map(r => {
      const w = (Math.abs(r.contribution) / maxAbs * 100).toFixed(1);
      const pos = r.contribution > 0;
      const bar = `<div class="dv-bar ${pos ? 'pos' : 'neg'}" data-w="${w}"></div>`;
      return `<div class="dv-row" title="weight ${r.weight} × TF-IDF ${r.tfidf}">
        <div class="dv-label">${esc(r.feature)}</div>
        <div class="dv-track"><div class="dv-half left">${pos ? '' : bar}</div><div class="dv-axis"></div><div class="dv-half">${pos ? bar : ''}</div></div>
        <div class="dv-val ${pos ? 'pos' : 'neg'}">${pos ? '+' : ''}${r.contribution.toFixed(3)}</div></div>`;
    }).join('');
  }

  /** highlighted words of the cleaned text for one view */
  function tokenChips(tokens, viewKey) {
    const vals = tokens.filter(t => t.known).map(t => Math.abs(t.values[viewKey] || 0));
    const maxAbs = Math.max(...vals, 1e-9);
    return tokens.map(t => {
      if (!t.known) return `<span class="tok unknown" title="not in the model vocabulary — ignored">${esc(t.token)}</span>`;
      const v = t.values[viewKey] || 0;
      if (v === 0) return `<span class="tok neutral" title="no effect in this view">${esc(t.token)}</span>`;
      const a = (0.15 + 0.6 * Math.abs(v) / maxAbs).toFixed(2);
      const rgb = v > 0 ? '108,99,255' : '255,101,132';
      return `<span class="tok" style="background:rgba(${rgb},${a});border-color:rgba(${rgb},0.9)"
               title="${v > 0 ? 'pushes toward' : 'pushes away'} (${v > 0 ? '+' : ''}${v.toFixed(3)})">${esc(t.token)}</span>`;
    }).join(' ');
  }

  /* ── background job polling ───────────────────────────────────────── */
  function pollJob(name, onUpdate) {
    return new Promise(resolve => {
      const tick = async () => {
        const r = await api(`/api/jobs/${name}/status`);
        if (!r.ok) { onUpdate({ state: 'failed', log: r.error }); return resolve('failed'); }
        onUpdate(r.data);
        if (r.data.state === 'running') return setTimeout(tick, 1500);
        resolve(r.data.state);
      };
      tick();
    });
  }


  /** Run a background job from a button: spinner + live log + clear failure state.
   *  ui = {btn, spinner, log, runningText}. Resolves 'finished' | 'failed'. */
  async function runJob(name, ui) {
    const label = ui.btn.textContent;
    ui.btn.disabled = true;
    ui.btn.textContent = ui.runningText || 'Running…';
    ui.log.classList.remove('hidden');
    ui.log.textContent = 'Starting…';
    const r = await api(`/api/jobs/${name}/start`, { method: 'POST' });
    if (!r.ok) {
      ui.log.textContent = r.error;
      ui.btn.disabled = false; ui.btn.textContent = label;
      return 'failed';
    }
    ui.spinner.classList.remove('hidden');
    const final = await pollJob(name, s => { ui.log.textContent = s.log || '…running'; ui.log.scrollTop = ui.log.scrollHeight; });
    ui.spinner.classList.add('hidden');
    ui.btn.disabled = false; ui.btn.textContent = label;
    return final;
  }

  /** <option> list for the "past reports" drop-down */
  function historyOptions(list) {
    const opts = ['<option value="">Latest report</option>'];
    (list || []).forEach(r => opts.push(`<option value="${esc(r.name)}">${esc((r.generated_at || r.name).replace('T', ' ').slice(0, 19))}${r.best_model ? ' · best: ' + esc(r.best_model) : ''}</option>`));
    return opts.join('');
  }

  /** what the dataset validation step found */
  function datasetValidation(i) {
    if (!i) return '';
    const raw = i.class_distribution_raw || {};
    const rawTxt = CLASS_ORDER.filter(k => raw[k] !== undefined).map(k => `${esc(k)}: ${raw[k].toLocaleString()}`).join(' · ');
    return `<div class="how-to-read"><ul>
      <li>Columns used: text = <code class="code-inline">${esc(i.text_column)}</code>, label = <code class="code-inline">${esc(i.label_column)}</code>.
          Labels: 0 → Hate Speech, 1 → Offensive, 2 → Normal (“neither”).</li>
      <li>${i.rows_read.toLocaleString()} rows read · ${(i.rows_skipped_empty_text || 0).toLocaleString()} empty texts and
          ${(i.rows_skipped_missing_label || 0).toLocaleString()} missing labels skipped.</li>
      <li>Raw class counts — ${rawTxt}</li>
      <li>${i.duplicates_or_empty_removed.toLocaleString()} duplicate texts removed
          (${(i.duplicate_label_conflicts || 0).toLocaleString()} had conflicting labels; first copy kept) →
          <b>${i.rows_used_after_dedup.toLocaleString()}</b> unique texts.</li></ul></div>`;
  }

  function datasetSummary(d) {
    if (!d) return '';
    const dist = CLASS_ORDER.filter(k => (d.class_distribution_test || {})[k] !== undefined)
      .map(k => [k, d.class_distribution_test[k]]).map(([k, v]) => `${esc(k)}: ${v.toLocaleString()}`).join(' · ');
    return `<div class="how-to-read">
      <b>${esc(d.file)}</b> (SHA-256 ${esc(d.sha256_prefix)}) ·
      ${d.rows_used_after_dedup.toLocaleString()} unique labelled texts after removing
      ${d.duplicates_or_empty_removed.toLocaleString()} duplicate/empty rows<br>
      Stratified split, seed ${d.random_state}: <b>${d.n_train.toLocaleString()} train</b> /
      <b>${d.n_test.toLocaleString()} test</b> (${Math.round(d.test_size * 100)}% test)<br>
      Test class counts — ${dist}</div>`;
  }

  return { esc, pct, f4, api, banner, animateBars, metricTiles, confusionMatrix, perClassBars,
           hbar, groupedBars, timeBars, divergingBars, tokenChips, pollJob, datasetSummary,
           runJob, historyOptions, datasetValidation,
           CLASS_COLORS, CLASS_ORDER, SERIES, sortedNames };
})();
