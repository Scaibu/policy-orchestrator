// Shared diagram navigation manifest and runtime utilities
const DIAGRAM_NAV = [
  { id: "master",        label: "Master Blueprint",      file: "master.html",        section: "1.0",   tag: "7-Phase Lifecycle" },
  { id: "phase-1",       label: "Phase 1 · Curation",    file: "phase-1.html",       section: "1.1.1", tag: "Data & Ingestion" },
  { id: "phase-2",       label: "Phase 2 · Micro-Arch",  file: "phase-2.html",       section: "1.1.2", tag: "Block Dynamics" },
  { id: "phase-3",       label: "Phase 3 · 5D Scaling",  file: "phase-3.html",       section: "1.1.3", tag: "Distributed Mesh" },
  { id: "phase-4",       label: "Phase 4 · Loss & Opt",  file: "phase-4.html",       section: "1.1.4", tag: "Loss & Recovery" },
  { id: "phase-5",       label: "Phase 5 · Alignment",   file: "phase-5.html",       section: "1.1.5", tag: "RLHF & Reasoning" },
  { id: "phase-6",       label: "Phase 6 · PEFT",        file: "phase-6.html",       section: "1.1.6", tag: "PEFT & Pruning" },
  { id: "phase-7",       label: "Phase 7 · Serving",     file: "phase-7.html",       section: "1.1.7", tag: "Paged & Speculative" },
  { id: "control-plane", label: "Control Plane",         file: "control-plane.html", section: "1.2",   tag: "Orchestration" },
  { id: "data-plane",    label: "Data Plane",            file: "data-plane.html",    section: "1.3",   tag: "Tensor Topology" },
  { id: "e2e",           label: "E2E Data Flow",         file: "e2e.html",           section: "4.1",   tag: "Transmission Map" },
  { id: "er-schema",     label: "ER Schema",             file: "er-schema.html",     section: "4.3",   tag: "Data Models & JSON" }
];

function initDiagramPage({ id, prevFile, nextFile }) {
  // Live clock
  const clockEl = document.getElementById("clock");
  if (clockEl) {
    const updateClock = () => {
      const now = new Date();
      clockEl.textContent = now.toTimeString().split(" ")[0] + " UTC";
    };
    updateClock();
    setInterval(updateClock, 1000);
  }

  // Keyboard navigation
  window.addEventListener("keydown", (e) => {
    // Avoid interfering if focus is in an input/textarea
    if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;

    if (e.key === "ArrowLeft" && prevFile) {
      window.location.href = prevFile;
    } else if (e.key === "ArrowRight" && nextFile) {
      window.location.href = nextFile;
    } else if (e.key === "Escape") {
      window.location.href = "../index.html#sec-diagrams";
    }
  });
}

function showToast(message) {
  let toast = document.getElementById("diag-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "diag-toast";
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toast._timeout);
  toast._timeout = setTimeout(() => {
    toast.classList.remove("show");
  }, 2200);
}

function copyMermaid(sourceId) {
  const el = document.getElementById(sourceId);
  if (!el) return;
  const text = el.textContent.trim();
  navigator.clipboard.writeText(text).then(() => {
    showToast("✓ Mermaid syntax copied to clipboard");
  }).catch(() => {
    showToast("Error copying to clipboard");
  });
}

function toggleSource(sourceId) {
  const el = document.getElementById(sourceId);
  if (!el) return;
  if (el.style.display === "none" || !el.style.display) {
    el.style.display = "block";
  } else {
    el.style.display = "none";
  }
}

function zoomDiagram(btn, factor) {
  const frame = btn.closest('.diagram-frame') || btn.closest('.mermaid-wrap') || document;
  const svg = frame.querySelector('.mermaid svg') || frame.querySelector('svg');
  if (!svg) return;
  let currentZoom = parseFloat(svg.getAttribute('data-zoom') || '1');
  currentZoom = Math.min(Math.max(currentZoom * factor, 0.6), 4.0);
  svg.setAttribute('data-zoom', currentZoom);
  svg.style.transform = `scale(${currentZoom})`;
  svg.style.transformOrigin = 'top center';
  svg.style.transition = 'transform 0.15s ease-out';
  if (currentZoom > 1.05) {
    svg.style.minWidth = `${100 * currentZoom}%`;
  } else {
    svg.style.minWidth = '';
  }
  showToast(`🔍 Zoom: ${Math.round(currentZoom * 100)}%`);
}

function resetZoom(btn) {
  const frame = btn.closest('.diagram-frame') || btn.closest('.mermaid-wrap') || document;
  const svg = frame.querySelector('.mermaid svg') || frame.querySelector('svg');
  if (!svg) return;
  svg.setAttribute('data-zoom', '1');
  svg.style.transform = '';
  svg.style.minWidth = '';
  showToast('⟲ Zoom reset to 100%');
}

function toggleFullscreen(btn) {
  const frame = btn.closest('.diagram-frame') || btn.closest('.mermaid-wrap');
  if (!frame) return;
  if (!document.fullscreenElement) {
    if (frame.requestFullscreen) {
      frame.requestFullscreen();
    } else if (frame.webkitRequestFullscreen) {
      frame.webkitRequestFullscreen();
    }
    showToast('⛶ Fullscreen mode (Press Esc to exit)');
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    }
  }
}
