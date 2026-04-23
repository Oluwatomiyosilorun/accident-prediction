// ── Dark Mode Toggle ─────────────────────────────────────────────────────────
(function () {
  const html   = document.documentElement;
  const toggle = document.getElementById('themeToggle');
  const icon   = document.getElementById('themeIcon');

  // Apply saved preference immediately (before paint)
  const saved = localStorage.getItem('theme') || 'light';
  html.setAttribute('data-bs-theme', saved);
  updateIcon(saved);

  if (toggle) {
    toggle.addEventListener('click', function () {
      const current = html.getAttribute('data-bs-theme');
      const next    = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-bs-theme', next);
      localStorage.setItem('theme', next);
      updateIcon(next);
    });
  }

  function updateIcon(theme) {
    if (!icon) return;
    if (theme === 'dark') {
      icon.className = 'bi bi-sun-fill';
      if (toggle) toggle.setAttribute('title', 'Switch to light mode');
    } else {
      icon.className = 'bi bi-moon-fill';
      if (toggle) toggle.setAttribute('title', 'Switch to dark mode');
    }
  }
})();
