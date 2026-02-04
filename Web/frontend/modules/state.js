/**
 * modules/state.js - Shared application state and toast utility.
 *
 * The `state` object is imported by every other module so they all read and
 * write the same values.  Using a single exported object means mutations are
 * visible across modules without any pub/sub machinery.
 *
 * `showToast` is placed here because it has no UI dependencies and is called
 * from both the settings and upload modules.
 */

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/**
 * Central state store for the current pipeline session.
 *
 * @property {File|null}   file     - The audio file the user selected.
 * @property {string|null} jobId	- UUID assigned by the server after upload.
 * @property {string[]}	speakers - Speaker IDs detected during transcription.
 * @property {boolean}     autoRun  - When true, each stage advances automatically.
 */
export const state = {
	file: null,
	jobId: null,
	speakers: [],
	autoRun: false,
	transcriber: 'assemblyai',
	llm: 'anthropic',
	videoGen: 'luma',
};

// ---------------------------------------------------------------------------
// Active sessions registry - tracks all in-progress or recently-started jobs
// so users can switch between them while pipelines run in the background.
// ---------------------------------------------------------------------------

/**
 * Map of all known active sessions keyed by jobId.
 *
 * @type {Object.<string, {filename: string, status: string}>}
 */
export const sessions = {};

/**
 * Register a newly-created session.
 *
 * @param {string} jobId	- UUID returned by the server.
 * @param {string} filename - Original audio filename.
 */
export function registerSession(jobId, filename) {
	sessions[jobId] = { filename, name: filename, status: 'processing' };
}

/**
 * Rename an active session (changes the display name in the sidebar).
 *
 * @param {string} jobId - The session to rename.
 * @param {string} name  - New display name.
 */
export function renameSession(jobId, name) {
	if (sessions[jobId]) sessions[jobId].name = name;
}

/**
 * Update the status of a tracked session.
 *
 * @param {string} jobId  - The session to update.
 * @param {string} status - New status string (e.g. 'processing', 'completed', 'error').
 */
export function updateSessionStatus(jobId, status) {
	if (sessions[jobId]) sessions[jobId].status = status;
}

// ---------------------------------------------------------------------------
// Toast notification helper
// ---------------------------------------------------------------------------

/** Reference to the toast container element - resolved once on first use. */
let _toastContainer = null;

/**
 * Display a temporary notification toast in the bottom-right corner.
 *
 * @param {string} message  - Text to display.
 * @param {'info'|'success'|'error'} [type='info'] - Controls border/text colour.
 */
export function showToast(message, type = 'info') {
	// Lazily resolve the container so this function can be called before DOM ready
	if (!_toastContainer) {
		_toastContainer = document.getElementById('toast-container');
	}
	if (!_toastContainer) return;

	const toast = document.createElement('div');
	toast.className = `toast toast-${type}`;
	toast.textContent = message;
	_toastContainer.appendChild(toast);

	// Trigger CSS transition: invisible -> visible
	setTimeout(() => toast.classList.add('show'), 10);

	// Remove toast after 3 seconds (wait for fade-out transition)
	setTimeout(() => {
		toast.classList.remove('show');
		setTimeout(() => toast.remove(), 300);
	}, 3000);
}
