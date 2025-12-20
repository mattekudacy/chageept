// Custom JavaScript for CHAGEEPT
(function() {
    // Update favicon
    const link = document.querySelector("link[rel~='icon']") || document.createElement('link');
    link.type = 'image/png';
    link.rel = 'icon';
    link.href = '/public/favicon.png';
    document.head.appendChild(link);

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
