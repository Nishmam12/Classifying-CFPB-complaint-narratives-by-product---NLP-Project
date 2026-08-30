import { CLASSES_META, BENCHMARK_MODELS, NOVELTY_EXPERIMENTS, SAMPLE_COMPLAINTS } from './data.js';
import { classifyComplaint } from './classifier.js';

document.addEventListener('DOMContentLoaded', () => {
  renderPresetPills();
  renderLeaderboard();
  renderNoveltyExperiments();
  bindClassifierControls();
  bindSylvaFrameInteraction();

  // Load initial demo complaint
  loadSample(0);
});

function bindSylvaFrameInteraction() {
  const iframe = document.getElementById('sylva-frame');
  if (!iframe) return;

  iframe.addEventListener('load', () => {
    try {
      const doc = iframe.contentDocument || iframe.contentWindow?.document;
      if (!doc) return;

      // When "Explore the work" or "Enter" is clicked inside the Sylva Hero, scroll down
      const exploreBtn = doc.querySelector('.liquid-button--explore');
      const enterLink = doc.querySelector('.dock-item--enter');
      const scrollCue = doc.querySelector('.scroll');

      const scrollToDashboard = (e) => {
        if (e) e.preventDefault();
        const target = document.getElementById('playground');
        if (target) {
          target.scrollIntoView({ behavior: 'smooth' });
        }
      };

      if (exploreBtn) exploreBtn.addEventListener('click', scrollToDashboard);
      if (enterLink) enterLink.addEventListener('click', scrollToDashboard);
      if (scrollCue) scrollCue.addEventListener('click', scrollToDashboard);
    } catch (err) {
      console.warn("Cross-origin frame boundary check:", err);
    }
  });
}

function renderPresetPills() {
  const container = document.getElementById('preset-pills-container');
  if (!container) return;

  container.innerHTML = SAMPLE_COMPLAINTS.map((sample, idx) => `
    <button class="chip-btn" data-index="${idx}">
      ${sample.label.split('/')[0].trim()}
    </button>
  `).join('');

  container.querySelectorAll('.chip-btn').forEach(btn => {
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
  if (textarea) {
    textarea.value = sample.text;
    runInference();
  }
}

function bindClassifierControls() {
  const runBtn = document.getElementById('btn-classify');
  const clearBtn = document.getElementById('btn-clear');
  const randomBtn = document.getElementById('btn-random');
  const modelSelect = document.getElementById('model-select');

  if (runBtn) {
    runBtn.addEventListener('click', () => runInference());
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      const textarea = document.getElementById('complaint-text');
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
    modelSelect.addEventListener('change', () => runInference());
  }
}

async function runInference() {
  const textarea = document.getElementById('complaint-text');
  const text = textarea ? textarea.value.trim() : '';
  if (!text) return;

  const modelSelect = document.getElementById('model-select');
  const modelType = modelSelect ? modelSelect.value : 'ensemble';

  const runBtn = document.getElementById('btn-classify');
  if (runBtn) {
    runBtn.textContent = 'Analyzing...';
    runBtn.disabled = true;
  }

  try {
    const result = await classifyComplaint(text, modelType);
    renderResults(result);
  } catch (err) {
    console.error("Classification error:", err);
  } finally {
    if (runBtn) {
      runBtn.textContent = 'Classify Complaint';
      runBtn.disabled = false;
    }
  }
}

function renderResults(result) {
  const container = document.getElementById('results-container');
  if (!container) return;

  const { topClass, allDistributions, modelName, latencyMs, tokenHighlights } = result;

  container.innerHTML = `
    <div style="display:flex; flex-direction:column; justify-content:space-between; height:100%;">
      <!-- Winner Card -->
      <div class="winner-box" style="border-color:${topClass.color}45; background:${topClass.color}12;">
        <div>
          <span class="winner-subtitle" style="color:${topClass.color};">Top-1 Prediction</span>
          <h4 class="winner-class-name">${topClass.className}</h4>
          <span style="font-size:0.78rem; color:var(--text-muted);">
            Architecture: <strong style="color:var(--text-pure);">${modelName}</strong> • Latency: <strong style="font-family:var(--font-mono); color:var(--emerald-green);">${latencyMs} ms</strong>
          </span>
        </div>
        <div>
          <div class="winner-confidence" style="color:${topClass.color};">${topClass.percentage}%</div>
          <span style="font-family:var(--font-mono); font-size:0.7rem; color:var(--text-dim); text-transform:uppercase;">Confidence</span>
        </div>
      </div>

      <!-- Probability Distribution Bars -->
      <div class="distribution-stack">
        ${allDistributions.map(item => `
          <div class="dist-bar-item">
            <div class="dist-labels">
              <span>${item.className}</span>
              <span style="font-family:var(--font-mono); font-weight:600; color:var(--text-pure);">${item.percentage}%</span>
            </div>
            <div class="dist-track">
              <div class="dist-fill" style="width:${item.percentage}%; background-color:${item.color};"></div>
            </div>
          </div>
        `).join('')}
      </div>

      <!-- Lexical Attention Signals -->
      ${tokenHighlights && tokenHighlights.length > 0 ? `
        <div class="tokens-container">
          <div class="tokens-header">Key Lexical Signals &amp; Attention Tokens:</div>
          <div class="tokens-list">
            ${tokenHighlights.map(t => `<span class="token-chip">${t}</span>`).join('')}
          </div>
        </div>
      ` : ''}
    </div>
  `;
}

function renderLeaderboard() {
  const tbody = document.getElementById('leaderboard-tbody');
  if (!tbody) return;

  tbody.innerHTML = BENCHMARK_MODELS.map((m, idx) => {
    let rankBadge = `<span class="rank-circle">${idx + 1}</span>`;
    if (idx === 0) rankBadge = `<span class="rank-circle rank-1">1</span>`;
    else if (idx === 1) rankBadge = `<span class="rank-circle rank-2">2</span>`;
    else if (idx === 2) rankBadge = `<span class="rank-circle rank-3">3</span>`;

    return `
      <tr class="${m.isEnsemble ? 'highlight-top' : ''}">
        <td>
          <div style="display:flex; align-items:center;">
            ${rankBadge}
            <div>
              <div style="font-weight:600; color:var(--text-pure); display:flex; align-items:center; gap:6px;">
                ${m.name}
                ${m.isEnsemble ? '<span class="bonus-pill" style="padding:2px 6px; font-size:0.7rem;">PRO +2</span>' : ''}
              </div>
              <small style="font-size:0.75rem; color:var(--text-dim);">${m.config}</small>
            </div>
          </div>
        </td>
        <td><span style="font-size:0.78rem; padding:3px 8px; border-radius:var(--radius-pill); background:rgba(255,255,255,0.05); color:var(--text-muted);">${m.paradigm}</span></td>
        <td><strong style="font-family:var(--font-mono); color:var(--emerald-green);">${m.macroF1.toFixed(4)}</strong></td>
        <td>${(m.accuracy * 100).toFixed(2)}%</td>
        <td>${(m.weightedF1 * 100).toFixed(2)}%</td>
        <td><span style="font-family:var(--font-mono); font-size:0.8rem;">${m.trainTime}</span></td>
        <td><span style="font-family:var(--font-mono); font-size:0.8rem;">${m.inferTime}</span></td>
      </tr>
    `;
  }).join('');
}

function renderNoveltyExperiments() {
  const tbody = document.getElementById('novelty-tbody');
  if (!tbody) return;

  tbody.innerHTML = NOVELTY_EXPERIMENTS.map(row => `
    <tr>
      <td><strong style="color:var(--text-pure);">${row.model}</strong></td>
      <td><span style="font-size:0.78rem; padding:3px 8px; border-radius:var(--radius-pill); background:rgba(255,255,255,0.05); color:var(--text-muted);">${row.strategy}</span></td>
      <td><strong style="font-family:var(--font-mono); color:var(--amber-gold);">${row.macroF1.toFixed(4)}</strong></td>
      <td>${(row.acc * 100).toFixed(2)}%</td>
      <td><span style="font-family:var(--font-mono); font-size:0.8rem;">${row.min4F1.toFixed(4)}</span></td>
      <td><span style="font-size:0.85rem; color:var(--text-muted);">${row.finding}</span></td>
    </tr>
  `).join('');
}
