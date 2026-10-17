/**
 * AIL-RDD Scroll Effects & Viewport Observer
 * Smooth scroll-triggered fade animations, floating nav elevation, and scrollspy.
 * Strictly zero emojis.
 */

export function initScrollEffects() {
  const floatingNav = document.querySelector('.floating-nav');
  const navLinks = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('section[id]');

  // Handle sticky floating nav elevation on scroll
  const handleScroll = () => {
    const scrollY = window.scrollY || window.pageYOffset;
    if (scrollY > 24) {
      floatingNav?.classList.add('scrolled');
    } else {
      floatingNav?.classList.remove('scrolled');
    }

    // Scrollspy section detection
    let currentSectionId = '';
    sections.forEach((section) => {
      const sectionTop = section.offsetTop - 120;
      const sectionHeight = section.offsetHeight;
      if (scrollY >= sectionTop && scrollY < sectionTop + sectionHeight) {
        currentSectionId = section.getAttribute('id');
      }
    });

    if (currentSectionId) {
      navLinks.forEach((link) => {
        const href = link.getAttribute('href');
        const isActive = href === `#${currentSectionId}`;
        link.classList.toggle('active', isActive);
      });
    }
  };

  window.addEventListener('scroll', handleScroll, { passive: true });
  handleScroll();

  // Smooth click navigation
  navLinks.forEach((link) => {
    link.addEventListener('click', (e) => {
      const href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        const target = document.querySelector(href);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth' });
          navLinks.forEach((l) => l.classList.remove('active'));
          link.classList.add('active');
        }
      }
    });
  });

  // IntersectionObserver for elements with .fade-reveal
  const revealElements = document.querySelectorAll('.fade-reveal');

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.12,
        rootMargin: '0px 0px -40px 0px',
      }
    );

    revealElements.forEach((el) => observer.observe(el));
  } else {
    // Fallback if IntersectionObserver is unsupported
    revealElements.forEach((el) => el.classList.add('revealed'));
  }
}
