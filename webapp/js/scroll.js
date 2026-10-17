/**
 * AIL-RDD Scroll Effects & Viewport Observer
 * Smooth scroll-triggered fade animations and scroll masks.
 * Strictly zero emojis.
 */

export function initScrollEffects() {
  const floatingNav = document.querySelector('.floating-nav');

  if (floatingNav) {
    const handleScroll = () => {
      const scrollY = window.scrollY || window.pageYOffset;
      if (scrollY > 24) {
        floatingNav.classList.add('scrolled');
      } else {
        floatingNav.classList.remove('scrolled');
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
  }

  // IntersectionObserver for elements with .fade-reveal
  const revealElements = document.querySelectorAll('.fade-reveal');

  if (revealElements.length > 0) {
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
          threshold: 0.1,
          rootMargin: '0px 0px -20px 0px',
        }
      );

      revealElements.forEach((el) => observer.observe(el));
    } else {
      revealElements.forEach((el) => el.classList.add('revealed'));
    }
  }
}
