const toolbarSelector = '.select-toolbar, .point-cloud-group-toolbar';
const stackOffsetProperty = '--bottom-toolbar-stack-offset';
const toolbarGap = 8;

/**
 * Keeps every visible toolbar anchored at the bottom of the canvas in a
 * vertical stack. Toolbars are created by several independent tools, so the
 * stack is driven by their shared DOM classes rather than by tool state.
 */
const setupBottomToolbarStack = (container: HTMLElement) => {
    const resizeObserver = new ResizeObserver(() => scheduleLayout());
    let observedToolbars = new Set<HTMLElement>();
    let animationFrame = 0;

    const getToolbars = () => Array.from(container.querySelectorAll<HTMLElement>(toolbarSelector));

    const syncObservedToolbars = (toolbars: HTMLElement[]) => {
        const nextToolbars = new Set(toolbars);

        for (const toolbar of observedToolbars) {
            if (!nextToolbars.has(toolbar)) {
                resizeObserver.unobserve(toolbar);
                toolbar.style.removeProperty(stackOffsetProperty);
            }
        }

        for (const toolbar of nextToolbars) {
            if (!observedToolbars.has(toolbar)) {
                resizeObserver.observe(toolbar);
            }
        }

        observedToolbars = nextToolbars;
    };

    const layout = () => {
        animationFrame = 0;

        const toolbars = getToolbars();
        syncObservedToolbars(toolbars);

        let stackOffset = 0;
        for (const toolbar of toolbars) {
            toolbar.style.setProperty(stackOffsetProperty, `${stackOffset}px`);

            const style = getComputedStyle(toolbar);
            if (style.display !== 'none' && style.visibility !== 'hidden') {
                stackOffset += Math.ceil(toolbar.getBoundingClientRect().height) + toolbarGap;
            }
        }
    };

    function scheduleLayout() {
        if (animationFrame === 0) {
            animationFrame = requestAnimationFrame(layout);
        }
    }

    const mutationObserver = new MutationObserver((mutations) => {
        const affectsToolbarLayout = mutations.some((mutation) => {
            return mutation.type === 'childList' ||
                (mutation.target instanceof HTMLElement && mutation.target.matches(toolbarSelector));
        });

        if (affectsToolbarLayout) {
            scheduleLayout();
        }
    });

    mutationObserver.observe(container, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['class']
    });

    scheduleLayout();
};

export { setupBottomToolbarStack };
