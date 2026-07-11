// Custom JavaScript for CHAGEEPT
(function() {
    // Update favicon - remove any existing icon links first (Chainlit injects its own),
    // then add ours with a cache-busting query param so browsers that already cached
    // Chainlit's default favicon are forced to re-fetch.
    document.querySelectorAll("link[rel~='icon'], link[rel='apple-touch-icon']").forEach((el) => el.remove());

    const iconLinks = [
        { rel: 'icon', type: 'image/x-icon', href: '/public/favicon.ico?v=3' },
        { rel: 'icon', type: 'image/png', sizes: '32x32', href: '/public/favicon-32x32.png?v=3' },
        { rel: 'icon', type: 'image/png', sizes: '16x16', href: '/public/favicon-16x16.png?v=3' },
        { rel: 'apple-touch-icon', sizes: '180x180', href: '/public/apple-touch-icon.png?v=3' },
    ];
    iconLinks.forEach(({ rel, type, sizes, href }) => {
        const link = document.createElement('link');
        link.rel = rel;
        if (type) link.type = type;
        if (sizes) link.sizes = sizes;
        link.href = href;
        document.head.appendChild(link);
    });

    // Add CHAGEE logo to welcome screen
    function addWelcomeLogo() {
        // Wait for the DOM to be ready
        setTimeout(() => {
            // Find the starters container or main welcome area
            const startersContainer = document.querySelector('[class*="starters"]') || 
                                    document.querySelector('.MuiGrid-root') ||
                                    document.querySelector('main > div');
            
            if (startersContainer && !document.getElementById('chagee-welcome-logo')) {
                // Create logo element
                const logoContainer = document.createElement('div');
                logoContainer.id = 'chagee-welcome-logo';
                logoContainer.style.textAlign = 'center';
                logoContainer.style.marginBottom = '40px';
                
                const logo = document.createElement('img');
                logo.src = '/public/chageelong.png';
                logo.alt = 'CHAGEE';
                logo.style.maxWidth = '280px';
                logo.style.height = 'auto';
                logo.style.display = 'block';
                logo.style.margin = '0 auto';
                
                logoContainer.appendChild(logo);
                
                // Insert before starters
                startersContainer.parentNode.insertBefore(logoContainer, startersContainer);
            }
        }, 500);
    }

    // Run on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', addWelcomeLogo);
    } else {
        addWelcomeLogo();
    }

    // Re-run when navigating (for chat history)
    window.addEventListener('popstate', addWelcomeLogo);
})();
