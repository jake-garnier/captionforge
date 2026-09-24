/**
 * Patreon Action Recorder
 *
 * Injected into the Patreon page to capture user interactions.
 * Events are queued in window.__recordedActions for Python to poll.
 */

(function() {
    // Prevent double initialization
    if (window.__actionRecorderInitialized) {
        console.log('[ActionRecorder] Already initialized');
        return;
    }
    window.__actionRecorderInitialized = true;

    // Event queue that Python will poll
    window.__recordedActions = [];

    // Recording state
    window.__recordingActive = false;
    window.__recordingStartTime = null;

    /**
     * Generate a unique selector for an element
     * Priority: data attributes > id > aria-label > class path
     */
    function getSelector(element) {
        if (!element || element === document.body || element === document.documentElement) {
            return 'body';
        }

        // Try data-tag attribute (Patreon uses these)
        if (element.dataset && element.dataset.tag) {
            return `[data-tag="${element.dataset.tag}"]`;
        }

        // Try data-testid (common in React apps)
        if (element.dataset && element.dataset.testid) {
            return `[data-testid="${element.dataset.testid}"]`;
        }

        // Try ID
        if (element.id) {
            return `#${element.id}`;
        }

        // Try aria-label
        const ariaLabel = element.getAttribute('aria-label');
        if (ariaLabel) {
            const tagName = element.tagName.toLowerCase();
            return `${tagName}[aria-label="${ariaLabel}"]`;
        }

        // Try name attribute for inputs
        if (element.name) {
            return `[name="${element.name}"]`;
        }

        // Try role + text content for buttons
        const role = element.getAttribute('role');
        if (role === 'button' || element.tagName === 'BUTTON') {
            const text = element.textContent.trim().substring(0, 50);
            if (text) {
                return `button:contains("${text}")`;
            }
        }

        // Build path with classes
        const tagName = element.tagName.toLowerCase();
        const classes = Array.from(element.classList)
            .filter(c => !c.match(/^(css-|sc-|emotion)/)) // Skip generated class names
            .slice(0, 3)
            .join('.');

        if (classes) {
            // Check if this selector is unique
            const selector = `${tagName}.${classes}`;
            if (document.querySelectorAll(selector).length === 1) {
                return selector;
            }
        }

        // Fall back to nth-child path
        const parent = element.parentElement;
        if (parent) {
            const siblings = Array.from(parent.children);
            const index = siblings.indexOf(element) + 1;
            const parentSelector = getSelector(parent);
            return `${parentSelector} > ${tagName}:nth-child(${index})`;
        }

        return tagName;
    }

    /**
     * Get element coordinates relative to viewport
     */
    function getCoordinates(element) {
        const rect = element.getBoundingClientRect();
        return {
            x: Math.round(rect.left + rect.width / 2),
            y: Math.round(rect.top + rect.height / 2)
        };
    }

    /**
     * Get timestamp relative to recording start
     */
    function getTimestamp() {
        if (!window.__recordingStartTime) {
            return 0;
        }
        return Date.now() - window.__recordingStartTime;
    }

    /**
     * Queue an action for Python to retrieve
     */
    function queueAction(action) {
        if (!window.__recordingActive) {
            return;
        }

        action.timestamp = getTimestamp();
        window.__recordedActions.push(action);

        console.log('[ActionRecorder] Captured:', action.type, action);
    }

    /**
     * Handle click events
     */
    function handleClick(event) {
        const element = event.target;

        // Skip if it's a file input (handled separately)
        if (element.tagName === 'INPUT' && element.type === 'file') {
            queueAction({
                type: 'file_upload',
                selector: getSelector(element),
                coordinates: getCoordinates(element),
                element_tag: element.tagName.toLowerCase()
            });
            return;
        }

        queueAction({
            type: 'click',
            selector: getSelector(element),
            coordinates: getCoordinates(element),
            element_text: element.textContent ? element.textContent.trim().substring(0, 100) : null,
            element_tag: element.tagName.toLowerCase()
        });
    }

    /**
     * Handle input/change events for text fields
     */
    function handleInput(event) {
        const element = event.target;

        // Only capture actual text inputs
        if (!['INPUT', 'TEXTAREA'].includes(element.tagName) &&
            !element.isContentEditable) {
            return;
        }

        // Get the current value
        const value = element.isContentEditable
            ? element.textContent
            : element.value;

        // Debounce: update last action if it's the same element
        const lastAction = window.__recordedActions[window.__recordedActions.length - 1];
        if (lastAction &&
            lastAction.type === 'type' &&
            lastAction.selector === getSelector(element)) {
            lastAction.text = value;
            lastAction.timestamp = getTimestamp();
            return;
        }

        queueAction({
            type: 'type',
            selector: getSelector(element),
            text: value,
            element_tag: element.tagName.toLowerCase(),
            is_contenteditable: element.isContentEditable
        });
    }

    /**
     * Handle keyboard events (for special keys)
     */
    function handleKeydown(event) {
        // Only capture special keys
        const specialKeys = ['Enter', 'Tab', 'Escape', 'Backspace', 'Delete'];
        if (!specialKeys.includes(event.key)) {
            return;
        }

        queueAction({
            type: 'keypress',
            key: event.key,
            selector: getSelector(event.target),
            with_shift: event.shiftKey,
            with_ctrl: event.ctrlKey,
            with_alt: event.altKey
        });
    }

    /**
     * Handle scroll events
     */
    let scrollTimeout = null;
    function handleScroll(event) {
        // Debounce scroll events
        if (scrollTimeout) {
            clearTimeout(scrollTimeout);
        }

        scrollTimeout = setTimeout(() => {
            queueAction({
                type: 'scroll',
                scroll_x: window.scrollX,
                scroll_y: window.scrollY
            });
        }, 250);
    }

    /**
     * Track navigation (page changes)
     */
    let lastUrl = window.location.href;
    function checkNavigation() {
        if (window.location.href !== lastUrl) {
            queueAction({
                type: 'navigate',
                url: window.location.href,
                previous_url: lastUrl
            });
            lastUrl = window.location.href;
        }
    }

    // Check for URL changes periodically (for SPA navigation)
    setInterval(checkNavigation, 500);

    /**
     * Start recording
     */
    window.__startRecording = function() {
        window.__recordedActions = [];
        window.__recordingStartTime = Date.now();
        window.__recordingActive = true;

        // Record initial state
        queueAction({
            type: 'navigate',
            url: window.location.href
        });

        console.log('[ActionRecorder] Recording started');
        return { status: 'started', timestamp: window.__recordingStartTime };
    };

    /**
     * Stop recording
     */
    window.__stopRecording = function() {
        window.__recordingActive = false;
        const actions = window.__recordedActions.slice();

        console.log('[ActionRecorder] Recording stopped. Captured', actions.length, 'actions');
        return {
            status: 'stopped',
            action_count: actions.length,
            actions: actions
        };
    };

    /**
     * Get recorded actions (for polling)
     */
    window.__getRecordedActions = function(clear = false) {
        const actions = window.__recordedActions.slice();
        if (clear) {
            window.__recordedActions = [];
        }
        return actions;
    };

    /**
     * Check if recording is active
     */
    window.__isRecording = function() {
        return window.__recordingActive;
    };

    // Attach event listeners
    document.addEventListener('click', handleClick, true);
    document.addEventListener('input', handleInput, true);
    document.addEventListener('change', handleInput, true);
    document.addEventListener('keydown', handleKeydown, true);
    window.addEventListener('scroll', handleScroll, true);

    // Listen for popstate (browser back/forward)
    window.addEventListener('popstate', function() {
        setTimeout(checkNavigation, 100);
    });

    console.log('[ActionRecorder] Initialized and ready');
})();
