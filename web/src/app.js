import { CLASSES_META, BENCHMARK_MODELS, NOVELTY_EXPERIMENTS, SAMPLE_COMPLAINTS, GALLERY_FIGURES } from './data.js';
import { classifyComplaint } from './classifier.js';

let currentSampleIndex = 0;
let currentLeaderboardFilter = 'all';
let currentNoveltyFilter = 'all';
let currentBertWeight = 0.75;

document.addEventListener('DOMContentLoaded', () => {
  renderPresetPills();
  renderLeaderboard();
  renderNoveltyExperiments();
  renderGallery();
  bindClassifierControls();
  bindEnsembleSlider();
  bindFilterTabs();
  bindLightbox();
  bindTextareaCounter();
  bindScrollSpy();

  // Load initial demo complaint
  loadSample(0);
});

/* =========================================================================
   Preset Buttons
   ========================================================================= */
function renderPresetPills() {
  const container = document.getElementById('preset-pills-container');
  if (!container) return;

  container.innerHTML = SAMPLE_COMPLAINTS.map((sample, idx) => `
    <button class="chip-btn ${idx === currentSampleIndex ? 'active' : ''}" data-index="${idx}">
      ${CLASSES_META[sample.classId]?.icon || '📄'} ${sample.label.split('/')[0].trim()}
    </button>
  `).join('');

  container.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const target = e.currentTarget;
      const idx = parseInt(target.getAttribute('data-index'), 10);
      container.querySelectorAll('.chip-btn').forEach(b => b.classList.remove('active'));
      target.classList.add('active');
      loadSample(idx);
    });
  });
}

function loadSample(idx) {
  currentSampleIndex = idx;
  const sample = SAMPLE_COMPLAINTS[idx];
  if (!sample) return;

  const textarea = document.getElementById('complaint-text');
  if (textarea) {
    textarea.value = sample.text;
    updateTextareaCounter(sample.text);
    runInference();
  }
}

/* =========================================================================
   Textarea Counter & Input Controls
   ========================================================================= */
function bindTextareaCounter() {
  const textarea = document.getElementById('complaint-text');
  if (!textarea) return;

  textarea.addEventListener('input', () => {
    updateTextareaCounter(textarea.value);
  });
}

function updateTextareaCounter(text) {
  const counter = document.getElementById('textarea-counter');
  if (!counter) return;

  const trimmed = text.trim();
  const wordCount = trimmed ? trimmed.split(/\s+/).length : 0;
  const charCount = text.length;

  counter.textContent = `${wordCount} words • ${charCount} chars`;
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
      if (textarea) {
        textarea.value = '';
        updateTextareaCounter('');
        textarea.focus();
      }
      const presetPills = document.querySelectorAll('#preset-pills-container .chip-btn');
      presetPills.forEach(p => p.classList.remove('active'));
    });
  }

  if (randomBtn) {
    randomBtn.addEventListener('click', () => {
      let randIdx = Math.floor(Math.random() * SAMPLE_COMPLAINTS.length);
      if (randIdx === currentSampleIndex) {
        randIdx = (randIdx + 1) % SAMPLE_COMPLAINTS.length;
      }
      const presetPills = document.querySelectorAll('#preset-pills-container .chip-btn');
      presetPills.forEach(p => p.classList.remove('active'));
      const activePill = document.querySelector(`#preset-pills-container .chip-btn[data-index="${randIdx}"]`);
      if (activePill) activePill.classList.add('active');
      loadSample(randIdx);
    });
  }

  if (modelSelect) {
    modelSelect.addEventListener('change', () => runInference());
  }
}

/* =========================================================================
   Inference Execution & Result Rendering
   ========================================================================= */
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
    const result = await classifyComplaint(text, modelType, { bertWeight: currentBertWeight });
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
    <div class="results-wrapper">
      <!-- Winner Card -->
      <div class="winner-box" style="border-color:${topClass.color}50; background:${topClass.color}15;">
        <div>
          <span class="winner-subtitle" style="color:${topClass.color};">
            <span>${topClass.icon || '🏆'}</span> Top-1 Predicted Product Category
          </span>
          <h4 class="winner-class-name">${topClass.className}</h4>
          <span style="font-size:0.82rem; color:var(--text-muted);">
            Architecture: <strong style="color:var(--text-pure);">${modelName}</strong> • Inference: <strong style="font-family:var(--font-mono); color:var(--emerald-green);">${latencyMs} ms</strong>
          </span>
        </div>
        <div>
          <div class="winner-confidence" style="color:${topClass.color};">${topClass.percentage}%</div>
          <span style="font-family:var(--font-mono); font-size:0.72rem; color:var(--text-dim); text-transform:uppercase; display:block; text-align:right;">Confidence</span>
        </div>
      </div>

      <!-- Probability Distribution Bars -->
      <div class="distribution-stack">
        ${allDistributions.map(item => `
          <div class="dist-bar-item">
            <div class="dist-labels">
              <span class="dist-class-name">
                <span>${item.icon || '•'}</span> ${item.className}
              </span>
              <span style="font-family:var(--font-mono); font-weight:600; color:var(--text-pure);">${item.percentage}%</span>
            </div>
            <div class="dist-track">
              <div class="dist-fill" style="width:${item.percentage}%; background-color:${item.color};"></div>
            </div>
          </div>
        `).join('')}
      </div>

      <!-- Key Attention Tokens -->
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

/* =========================================================================
   Leaderboard Rendering & Filtering
   ========================================================================= */
function renderLeaderboard() {
  const tbody = document.getElementById('leaderboard-tbody');
  if (!tbody) return;

  const filtered = currentLeaderboardFilter === 'all'
    ? BENCHMARK_MODELS
    : BENCHMARK_MODELS.filter(m => {
        const p = m.paradigm.toLowerCase();
        const f = currentLeaderboardFilter.toLowerCase();
        if (f.includes('transformer') || f.includes('slm') || f.includes('encoder') || f.includes('decoder')) {
          return p.includes('transformer') || p.includes('encoder') || p.includes('decoder') || p.includes('slm');
        }
        if (f.includes('recurrent')) {
          return p.includes('recurrent');
        }
        if (f.includes('classical')) {
          return p.includes('classical');
        }
        if (f.includes('ensemble')) {
          return p.includes('ensemble');
        }
        return p === f;
      });

  tbody.innerHTML = filtered.map((m, idx) => {
    let rankBadge = `<span class="rank-circle">${idx + 1}</span>`;
    if (m.isEnsemble) rankBadge = `<span class="rank-circle rank-1">1</span>`;
    else if (m.isBestSingle) rankBadge = `<span class="rank-circle rank-2">2</span>`;
    else if (m.isSLM) rankBadge = `<span class="rank-circle rank-3">3</span>`;
    else if (idx === 2) rankBadge = `<span class="rank-circle rank-3">3</span>`;

    const f1Pct = (m.macroF1 * 100).toFixed(1);
    const accPct = (m.accuracy * 100).toFixed(1);

    return `
      <tr class="${m.isEnsemble ? 'highlight-top' : (m.isSLM ? 'highlight-slm' : '')}">
        <td>
          <div style="display:flex; align-items:center;">
            ${rankBadge}
            <div>
              <div style="font-weight:700; color:var(--text-pure); display:flex; align-items:center; gap:6px; flex-wrap:wrap;">
                ${m.name}
                ${m.isEnsemble ? '<span class="bonus-pill" style="padding:2px 8px; font-size:0.7rem;">PRO +2</span>' : ''}
                ${m.isBestSingle ? '<span class="badge-tag" style="background:rgba(59,130,246,0.2); color:#60a5fa; border:1px solid rgba(59,130,246,0.4);">BEST ENCODER</span>' : ''}
                ${m.isSLM ? '<span class="badge-tag" style="background:rgba(245,158,11,0.2); color:#fbbf24; border:1px solid rgba(245,158,11,0.4);">⭐ 1.54B LoRA SLM</span>' : ''}
              </div>
              <small style="font-size:0.75rem; color:var(--text-dim); display:block; margin-top:2px;">${m.config}</small>
            </div>
          </div>
        </td>
        <td>
          <span style="font-size:0.78rem; padding:4px 10px; border-radius:var(--radius-pill); background:rgba(255,255,255,0.06); color:var(--text-muted); font-weight:500;">
            ${m.paradigm}
          </span>
        </td>
        <td>
          <div class="metric-bar-cell">
            <strong style="font-family:var(--font-mono); color:var(--emerald-green); font-size:0.95rem;">${m.macroF1.toFixed(4)}</strong>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" style="width:${f1Pct}%; background:var(--emerald-green);"></div>
            </div>
          </div>
        </td>
        <td>
          <div class="metric-bar-cell">
            <span style="font-family:var(--font-mono); color:var(--text-pure); font-size:0.88rem;">${(m.accuracy * 100).toFixed(2)}%</span>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" style="width:${accPct}%; background:var(--blue-accent);"></div>
            </div>
          </div>
        </td>
        <td><span style="font-family:var(--font-mono); font-size:0.88rem;">${(m.weightedF1 * 100).toFixed(2)}%</span></td>
        <td><span style="font-family:var(--font-mono); font-size:0.82rem; color:var(--text-dim);">${m.trainTime}</span></td>
        <td><span style="font-family:var(--font-mono); font-size:0.82rem; color:var(--emerald-green); font-weight:600;">${m.inferTime}</span></td>
      </tr>
    `;
  }).join('');
}

/* =========================================================================
   Ablation Studies Rendering & Filtering
   ========================================================================= */
function renderNoveltyExperiments() {
  const tbody = document.getElementById('novelty-tbody');
  if (!tbody) return;

  const filtered = currentNoveltyFilter === 'all'
    ? NOVELTY_EXPERIMENTS
    : NOVELTY_EXPERIMENTS.filter(r => r.model.toLowerCase() === currentNoveltyFilter.toLowerCase());

  tbody.innerHTML = filtered.map(row => `
    <tr>
      <td><strong style="color:var(--text-pure); font-weight:700;">${row.model}</strong></td>
      <td>
        <span style="font-size:0.78rem; padding:3px 8px; border-radius:var(--radius-pill); background:rgba(255,255,255,0.05); color:var(--text-muted);">
          ${row.paradigm}
        </span>
      </td>
      <td>
        <span style="font-size:0.85rem; font-weight:600; color:var(--text-pure);">
          ${row.strategy}
        </span>
      </td>
      <td><strong style="font-family:var(--font-mono); color:var(--amber-gold);">${row.macroF1.toFixed(4)}</strong></td>
      <td><span style="font-family:var(--font-mono); font-size:0.88rem;">${(row.acc * 100).toFixed(2)}%</span></td>
      <td><span style="font-family:var(--font-mono); font-size:0.88rem; font-weight:600; color:var(--emerald-green);">${row.min4F1.toFixed(4)}</span></td>
      <td><span style="font-size:0.85rem; color:var(--text-muted); line-height:1.5;">${row.finding}</span></td>
    </tr>
  `).join('');
}

/* =========================================================================
   Visual Diagnostic Gallery & Lightbox Modal
   ========================================================================= */
function renderGallery() {
  const container = document.getElementById('gallery-grid-container');
  if (!container) return;

  container.innerHTML = GALLERY_FIGURES.map((fig, idx) => `
    <div class="gallery-item" data-index="${idx}">
      <div class="gallery-img-wrapper">
        <img src="${fig.src}" alt="${fig.title}" loading="lazy" />
        <div class="gallery-zoom-overlay">
          <span class="zoom-badge">🔍 Click to Expand</span>
        </div>
      </div>
      <div class="gallery-meta">
        <span class="gallery-meta-tag">${fig.category}</span>
        <h4 class="gallery-meta-title">${fig.title}</h4>
        <p class="gallery-meta-desc">${fig.description}</p>
      </div>
    </div>
  `).join('');

  container.querySelectorAll('.gallery-item').forEach(item => {
    item.addEventListener('click', (e) => {
      const idx = parseInt(e.currentTarget.getAttribute('data-index'), 10);
      openLightbox(idx);
    });
  });
}

function bindLightbox() {
  const modal = document.getElementById('lightbox-modal');
  const closeBtn = document.getElementById('lightbox-close-btn');

  if (!modal) return;

  if (closeBtn) {
    closeBtn.addEventListener('click', () => closeLightbox());
  }

  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      closeLightbox();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('open')) {
      closeLightbox();
    }
  });
}

function openLightbox(idx) {
  const fig = GALLERY_FIGURES[idx];
  if (!fig) return;

  const modal = document.getElementById('lightbox-modal');
  const img = document.getElementById('lightbox-img');
  const title = document.getElementById('lightbox-title');
  const desc = document.getElementById('lightbox-desc');

  if (modal && img && title && desc) {
    img.src = fig.src;
    img.alt = fig.title;
    title.textContent = fig.title;
    desc.textContent = fig.description;
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
  }
}

function closeLightbox() {
  const modal = document.getElementById('lightbox-modal');
  if (modal) {
    modal.classList.remove('open');
    document.body.style.overflow = '';
  }
}

/* =========================================================================
   Ensemble Blend Slider
   ========================================================================= */
function bindEnsembleSlider() {
  const slider = document.getElementById('blend-slider');
  const badge = document.getElementById('blend-badge');

  if (!slider) return;

  slider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    currentBertWeight = val / 100;
    const lrWeight = 100 - val;

    if (badge) {
      badge.textContent = `Current Blend: ${val}% BERT + ${lrWeight}% LR`;
    }

    const modelSelect = document.getElementById('model-select');
    if (modelSelect && modelSelect.value === 'ensemble') {
      runInference();
    }
  });
}

/* =========================================================================
   Filter Tabs
   ========================================================================= */
function bindFilterTabs() {
  // Leaderboard filters
  const lbFilters = document.querySelectorAll('#leaderboard-filters .filter-tab-btn');
  lbFilters.forEach(btn => {
    btn.addEventListener('click', (e) => {
      lbFilters.forEach(b => b.classList.remove('active'));
      e.currentTarget.classList.add('active');
      currentLeaderboardFilter = e.currentTarget.getAttribute('data-filter');
      renderLeaderboard();
    });
  });

  // Novelty / Ablation filters
  const noveltyFilters = document.querySelectorAll('#novelty-filters .filter-tab-btn');
  noveltyFilters.forEach(btn => {
    btn.addEventListener('click', (e) => {
      noveltyFilters.forEach(b => b.classList.remove('active'));
      e.currentTarget.classList.add('active');
      currentNoveltyFilter = e.currentTarget.getAttribute('data-filter');
      renderNoveltyExperiments();
    });
  });
}

/* =========================================================================
   Scroll Spy for Active Navigation
   ========================================================================= */
function bindScrollSpy() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
      const sectionTop = section.offsetTop - 120;
      if (window.scrollY >= sectionTop) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });
}
