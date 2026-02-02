/**
 * app.js - DND Video Pipeline - frontend entry point.
 *
 * This file is intentionally slim: it imports the feature modules, wires up
 * the shared `switchView` utility, and kicks off initialisation.
 *
 * Module layout:
 *   modules/state.js       - shared state object + showToast helper
 *   modules/settings.js	- settings modal (API key management)
 *   modules/upload.js      - drag-drop file selection and upload
 *   modules/transcript.js  - transcript display and speaker-mapping form
 *   modules/websocket.js   - real-time progress via WebSocket
 *   modules/pipeline.js	- phase-by-phase pipeline controller
 *   modules/history.js     - past-session sidebar
 */

import { initSettings }					from './modules/settings.js';
import { initUpload }                      from './modules/upload.js';
import { initHistory }                     from './modules/history.js';
import { switchToSession as _switchToSess, handleNewSession } from './modules/pipeline.js';

// ---------------------------------------------------------------------------
// Bootstrap - wait for the DOM to be ready before touching any elements
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {

	/**
     * Named view sections that can be shown/hidden.
     * Keys must match the `id` attributes of the <section> elements in
     * index.html (minus the "-view" suffix) plus the history sidebar.
     */
	const views = {
		upload:   document.getElementById('upload-view'),
		pipeline: document.getElementById('pipeline-view'),
	};

	/**
     * Transition to a named view.
     *
     * The brief delay between removing 'active' and adding it to the new view
     * gives CSS transitions time to fire (opacity/translateY fade).
     *
     * @param {'upload'|'pipeline'} viewName - Target view.
     */
	function switchView(viewName) {
		// Fade out all views
		Object.values(views).forEach(v => {
			if (v) {
				v.classList.remove('active');
				setTimeout(() => v.classList.add('hidden'), 50);
			}
		});

		// Fade in the target view after a brief pause
		setTimeout(() => {
			const target = views[viewName];
			if (target) {
				target.classList.remove('hidden');
				void target.offsetWidth; // force layout reflow to re-trigger CSS transition
				target.classList.add('active');
			}
		}, 60);
	}

	/**
     * Switch the pipeline view to a specific active session by jobId.
     * Bound here so it closes over `switchView`.
     *
     * @param {string} jobId - The session UUID to switch to.
     */
	function switchToSession(jobId) {
		return _switchToSess(jobId, switchView);
	}

	// ---------------------------------------------------------------------------
	// Initialise feature modules
	// ---------------------------------------------------------------------------
	initSettings();
	initUpload(switchView);
	initHistory(switchView, switchToSession);

	const newSessionBtn = document.getElementById('new-session-btn');
	if (newSessionBtn) newSessionBtn.onclick = () => handleNewSession(switchView);
});
