import { CLASSES_META, BENCHMARK_MODELS, NOVELTY_EXPERIMENTS, SAMPLE_COMPLAINTS } from './data.js';
import { classifyComplaint } from './classifier.js';

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  renderPresetChips();
  renderLeaderboard();
  renderNoveltyExperiments();
  bindClassifierEvents();
  
  // Run initial demo classification with the first sample
  loadSample(0);
});

// Theme handling
function initTheme() {
  const toggleBtn = document.getElementById('theme-toggle');
  const savedTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);

  toggleBtn.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
  });
}

function updateThemeIcon(theme) {
  const icon = document.getElementById('theme-icon');
  if (icon) {
    icon.innerHTML = theme === 'dark' 
      ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`
      : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
  }
}

// Preset chips
function renderPresetChips() {
  const container = document.getElementById('preset-chips');
  if (!container) return;

  container.innerHTML = SAMPLE_COMPLAINTS.map((sample, idx) => `
    <button class="chip" data-index="${idx}">
      ${sample.label.split('/')[0].trim()}
    </button>
  `).join('');

  container.querySelectorAll('.chip').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const idx = parseInt(e.target.getAttribute('data-index'), 10);
      loadSample(idx);
    });
  });
}

function loadSample(idx) {
  const sample = SAMPLE_COMPLAINTS[idx];
  if (!sample) return;

  const textarea = document.getElementById('complaint-text');
  textarea.value = sample.text;
  runInference();
}

// Classifier interactions
function bindClassifierEvents() {
  const runBtn = document.getElementById('btn-classify');
  const clearBtn = document.getElementById('btn-clear');
  const randomBtn = document.getElementById('btn-random');
  const modelSelect = document.getElementById('model-select');
  const textarea = document.getElementById('complaint-text');

  if (runBtn) {
    runBtn.addEventListener('click', () => runInference());
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      textarea.value = '';
      textarea.focus();
    });
  }

  if (randomBtn) {
    randomBtn.addEventListener('click', () => {
      const randIdx = Math.floor(Math.random() * SAMPLE_COMPLAINTS.length);
      loadSample(randIdx);
    });
  }

  if (modelSelect) {
    modelSelect.addEventListener('change', () => {
      if (textarea.value.trim().length > 0) {
        runInference();
      }
    });
  }
}

async function runInference() {
  const textarea = document.getElementById('complaint-text');
  const text = textarea.value.trim();
  if (!text) return;

  const modelType = document.getElementById('model-select').value;
  const runBtn = document.getElementById('btn-classify');
  
  if (runBtn) {
    runBtn.innerHTML = `
      <svg class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="12" y1="2" x2="12" y2="6"></line>
        <line x1="12" y1="18" x2="12" y2="22"></line>
        <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
        <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
        <line x1="2" y1="12" x2="6" y2="12"></line>
        <line x1="18" y1="12" x2="22" y2="12"></line>
        <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line>
        <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
      </svg>
      Analyzing...
    `;
    runBtn.disabled = true;
  }

  try {
    const result = await classifyComplaint(text, modelType);
    renderResults(result);
  } catch (err) {
    alert(err.message);
  } finally {
    if (runBtn) {
      runBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
        Classify Complaint
      `;
      runBtn.disabled = false;
    }
  }
}

function renderResults(result) {
  const container = document.getElementById('results-container');
  if (!container) return;

  const { topClass, allDistributions, modelName, latencyMs, tokenHighlights } = result;

  container.innerHTML = `
    <div class="result-box">
      <!-- Winner Card -->
      <div class="winner-card" style="border-color: ${topClass.color}40; background: ${topClass.color}10;">
        <div class="winner-info">
          <span class="winner-subtitle">Predicted Category (Top-1)</span>
          <h3 class="winner-name">${topClass.className}</h3>
          <span class="metric-sub">Model: ${modelName} • Latency: ${latencyMs} ms</span>
        </div>
        <div class="winner-score-badge">
          <div class="winner-pct" style="color: ${topClass.color};">${topClass.percentage}%</div>
          <div class="winner-conf">Confidence</div>
        </div>
      </div>

      <!-- Probability Distribution -->
      <div class="dist-list">
        ${allDistributions.map(item => `
          <div class="dist-item">
            <div class="dist-meta">
              <span class="dist-name">${item.className}</span>
              <span class="dist-pct">${item.percentage}%</span>
            </div>
            <div class="dist-bar-bg">
              <div class="dist-bar-fill" style="width: ${item.percentage}%; background-color: ${item.color};"></div>
            </div>
          </div>
        `).join('')}
      </div>

      <!-- Lexical Highlights -->
      ${tokenHighlights && tokenHighlights.length > 0 ? `
        <div class="tokens-wrap">
          <div class="tokens-title">Key Lexical / Attention Signals</div>
          <div class="token-tags">
            ${tokenHighlights.map(t => `<span class="token-tag">${t}</span>`).join('')}
          </div>
        </div>
      ` : ''}
    </div>
  `;
}

// Leaderboard Table
function renderLeaderboard() {
  const tableBody = document.getElementById('benchmark-tbody');
  if (!tableBody) return;

  tableBody.innerHTML = BENCHMARK_MODELS.map((m, idx) => {
    let rankBadgeClass = '';
    if (idx === 0) rankBadgeClass = 'rank-1';
    else if (idx === 1) rankBadgeClass = 'rank-2';
    else if (idx === 2) rankBadgeClass = 'rank-3';

    return `
      <tr class="${m.isEnsemble ? 'highlight-row' : ''}">
        <td>
          <div class="model-cell">
            <span class="rank-badge ${rankBadgeClass}">${idx + 1}</span>
            <div>
              <div>${m.name} ${m.isEnsemble ? '<span class="paradigm-tag" style="background: rgba(16,185,129,0.2); color: #10b981; font-weight:700;">+2 BONUS ENSEMBLE</span>' : ''}</div>
              <small class="text-muted" style="font-size:0.75rem; color: var(--text-muted);">${m.config}</small>
            </div>
          </div>
        </td>
        <td><span class="paradigm-tag">${m.paradigm}</span></td>
        <td><strong class="f1-badge">${m.macroF1.toFixed(4)}</strong></td>
        <td>${(m.accuracy * 100).toFixed(2)}%</td>
        <td>${(m.weightedF1 * 100).toFixed(2)}%</td>
        <td><span style="font-family:var(--font-mono); font-size:0.85rem;">${m.trainTime}</span></td>
        <td><span style="font-family:var(--font-mono); font-size:0.85rem;">${m.inferTime}</span></td>
      </tr>
    `;
  }).join('');
}

// Novelty & Imbalance Experiments
function renderNoveltyExperiments() {
  const container = document.getElementById('novelty-tbody');
  if (!container) return;

  container.innerHTML = NOVELTY_EXPERIMENTS.map(row => `
    <tr>
      <td><strong>${row.model}</strong></td>
      <td><span class="paradigm-tag">${row.strategy}</span></td>
      <td><strong style="color: var(--accent-emerald); font-family: var(--font-mono);">${row.macroF1.toFixed(4)}</strong></td>
      <td>${(row.acc * 100).toFixed(2)}%</td>
      <td><span style="font-family: var(--font-mono);">${row.min4F1.toFixed(4)}</span></td>
      <td><span style="font-size: 0.85rem; color: var(--text-secondary);">${row.finding}</span></td>
    </tr>
  `).join('');
}
