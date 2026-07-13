/* ── HDI Predictor · main.js ───────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {

  /* ── Animated score counter ───────────────────────────────────── */
  const scoreEl = document.getElementById('scoreNum');
  if (scoreEl) {
    const target = parseFloat(scoreEl.dataset.target) || 0;
    let start = null;
    const duration = 1400;

    function animateCount(timestamp) {
      if (!start) start = timestamp;
      const progress = Math.min((timestamp - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 4);
      scoreEl.textContent = (target * eased).toFixed(4);
      if (progress < 1) requestAnimationFrame(animateCount);
      else scoreEl.textContent = target.toFixed(4);
    }
    requestAnimationFrame(animateCount);
  }

  /* ── Gauge & mini-bar animation on load ───────────────────────── */
  setTimeout(() => {
    document.querySelectorAll('.gauge-fill').forEach(el => {
      const w = getComputedStyle(el).getPropertyValue('--target-width');
      el.style.width = w;
    });
    document.querySelectorAll('.mini-fill').forEach(el => {
      const w = getComputedStyle(el).getPropertyValue('--w');
      el.style.width = w;
    });
  }, 200);

  /* ── Form submission loading state ───────────────────────────── */
  const form = document.getElementById('predictForm');
  const btn  = document.getElementById('submitBtn');
  if (form && btn) {
    form.addEventListener('submit', () => {
      btn.disabled = true;
      btn.querySelector('.btn-text').textContent = 'Predicting…';
      btn.querySelector('.btn-icon').textContent = '⏳';
    });
  }

  /* ── Tooltip title fallback ───────────────────────────────────── */
  document.querySelectorAll('.tooltip').forEach(el => {
    el.setAttribute('aria-label', el.getAttribute('title') || '');
  });

  /* ── Smooth reveal on scroll ──────────────────────────────────── */
  if ('IntersectionObserver' in window) {
    const cards = document.querySelectorAll('.step-card, .viz-card, .result-card');
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.style.opacity = '1';
          e.target.style.transform = 'translateY(0)';
          obs.unobserve(e.target);
        }
      });
    }, { threshold: 0.08 });

    cards.forEach(card => {
      card.style.opacity = '0';
      card.style.transform = 'translateY(18px)';
      card.style.transition = 'opacity .45s ease, transform .45s ease';
      obs.observe(card);
    });
  }

});
