/**
 * modules/history.js - Past-session history sidebar.
 *
 * Fetches the list of previous pipeline runs from /api/history and renders
 * clickable entries in the sidebar.  Clicking an entry either:
 *  - Shows the finished video directly (results view), or
 *  - Loads the transcript so the user can redo the speaker mapping.
 *
 * For sessions that have multiple provider combinations (e.g., Luma + Replicate),
 * run badges are shown so the user can click into a specific past run.
 * Renders two sections in the sidebar:
 *
 *  1. Active Sessions - jobs started this browser session that are still
 *     running (or recently finished).  Clicking one calls `switchToSession`
 *     so the pipeline view switches to that job without interrupting it.
 *
 *  2. Past Sessions - completed/historical runs fetched from /api/history.
 *     Clicking an entry either shows the finished video or loads the
 *     transcript for speaker-mapping.
 */

import { state, sessions } from './state.js';

// Stored reference to the switchView callback for use in delete handlers.
let _switchView = null;

/** Stored reference to the switchToSession callback wired up in app.js. */
let _switchToSession = null;

/**
 * Initialise the history sidebar: load initial data and wire the refresh button.
 * Must be called once after the DOM is ready.
 *
 * @param {function} switchView      - View-switching callback.
 * @param {function} switchToSession - Callback to switch the pipeline to an active job.
 */
export function initHistory(switchView, switchToSession) {
	_switchToSession = switchToSession;
	_switchView = switchView;

	const refreshBtn = document.getElementById('refresh-history-btn');
	if (refreshBtn) {
		refreshBtn.addEventListener('click', () => {
			renderActiveSessions();
			fetchHistory(switchView);
		});
	}

	// Load history immediately on page open
	renderActiveSessions();
	fetchHistory(switchView);
}

// ---------------------------------------------------------------------------
// Active sessions (in-memory, current browser session)
// ---------------------------------------------------------------------------

/**
 * Re-render the "Active Sessions" section of the sidebar.
 * Called whenever a session's status changes (stage complete, error, etc.).
 */
export function renderActiveSessions() {
	const container = document.getElementById('active-sessions-list');
	if (!container) return;

	const entries = Object.entries(sessions);

	if (entries.length === 0) {
		container.innerHTML = '<p class="sidebar-empty">No active sessions.</p>';
		return;
	}

	container.innerHTML = '';
	entries.forEach(([jobId, session]) => {
		const el = document.createElement('div');
		el.className = 'history-item active-session-item';
		if (state.jobId === jobId) el.classList.add('active');

		const statusClass = session.status === 'completed' ? 'badge-done'
                          : session.status === 'error'     ? 'badge-error'
                          : 'badge-running';
		const statusLabel = session.status === 'completed' ? 'Done'
                          : session.status === 'error'     ? 'Error'
                          : 'Running';

		el.innerHTML = `
			<div class="history-item-header">
				<h3 class="active-session-name">${_truncate(session.name || session.filename, 22)}</h3>
				<button class="delete-session-btn" data-job-id="${jobId}" title="Delete session" aria-label="Delete session">&#x2715;</button>
			</div>
			<div class="history-item-meta">
				<span class="meta-badge ${statusClass}">${statusLabel}</span>
			</div>
		`;

		el.addEventListener('click', (e) => {
			if (e.target.closest('.delete-session-btn')) {
				e.stopPropagation();
				_deleteActiveSession(jobId, el);
				return;
			}
			if (_switchToSession) _switchToSession(jobId);
		});

		container.appendChild(el);
	});
}

// ---------------------------------------------------------------------------
// Fetch & render past sessions
// ---------------------------------------------------------------------------

/**
 * Fetch past sessions from the server and render them in the sidebar list.
 *
 * @param {function} switchView - View-switching callback (passed through to click handlers).
 */
export async function fetchHistory(switchView) {
	const historyList = document.getElementById('history-list');
	if (!historyList) return;

	// Show skeleton loader while waiting
	historyList.innerHTML = `
		<div class="skeleton-loader small">
			<div class="skeleton-bar"></div>
			<div class="skeleton-bar"></div>
		</div>`;

	try {
		const res = await fetch('/api/history');
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();
		renderHistory(data.sessions, historyList, switchView);
	} catch (err) {
		console.error('History fetch error:', err);
		historyList.innerHTML = '<p style="color:#ff7b72; padding:1rem;">Failed to load history.</p>';
	}
}

/**
 * Render a list of session objects into the sidebar history list element.
 *
 * @param {object[]}	sessions	- Array of session metadata from the API.
 * @param {HTMLElement} historyList - The container element to populate.
 * @param {function}	switchView  - View-switching callback for click handlers.
 */
/** Format an ISO timestamp string into a human-readable label. */
function _formatTimestamp(iso) {
	if (!iso) return '';
	try {
		const d = new Date(iso);
		return d.toLocaleString(undefined, {
			year: 'numeric', month: 'short', day: 'numeric',
			hour: '2-digit', minute: '2-digit',
		});
	} catch (_) {
		return iso;
	}
}

function renderHistory(sessions, historyList, switchView) {
	historyList.innerHTML = '';

	if (!sessions || sessions.length === 0) {
		historyList.innerHTML = '<p style="color:#8b949e; padding:1rem;">No past sessions found.</p>';
		return;
	}

	sessions.forEach(session => {
		const el = document.createElement('div');
		el.className = 'history-item';

		// Build status badge HTML
		let badges = '';
		if (session.has_transcript) badges += '<span class="meta-badge">Transcript</span>';
		if (session.video_url)       badges += '<span class="meta-badge video">Video</span>';

		// Build run-variant badges (one per completed Stage-3 provider combination)
		let runBadgesHtml = '';
		if (session.runs && session.runs.length > 1) {
			const badgeItems = session.runs.map(run => {
				const label = run.video_gen || run.key;
				const hasVideo = run.has_video ? ' has-video' : '';
				return `<span class="run-badge${hasVideo}" data-run-key="${run.key}" title="${run.transcriber} / ${run.llm} / ${run.video_gen}">${label}</span>`;
			}).join('');
			runBadgesHtml = `<div class="run-badges">${badgeItems}</div>`;
		}

		const tsHtml = session.created_at
			? `<p class="history-timestamp">${_formatTimestamp(session.created_at)}</p>`
			: '';

		el.innerHTML = `
			<div class="history-item-header">
				<h3 class="history-item-name">${session.name}</h3>
				<button class="delete-session-btn" data-session-id="${session.id}" title="Delete session" aria-label="Delete session">&#x2715;</button>
			</div>
			${tsHtml}
			<div class="history-item-meta">${badges}</div>
			${runBadgesHtml}
		`;

		el.addEventListener('click', (e) => {
			// Delete button
			if (e.target.closest('.delete-session-btn')) {
				e.stopPropagation();
				_deleteSession(session.id, el);
				return;
			}
			// Run badge
			const runBadge = e.target.closest('.run-badge');
			if (runBadge) {
				e.stopPropagation();
				const runKey = runBadge.dataset.runKey;
				loadHistorySessionWithKey(session, runKey, el, switchView);
				return;
			}
			loadHistorySession(session, el, switchView);
		});
		historyList.appendChild(el);
	});
}

async function _deleteActiveSession(jobId, element) {
	try {
		const res = await fetch(`/api/job/${jobId}`, { method: 'DELETE' });
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		// Remove from the in-memory sessions registry
		delete sessions[jobId];
		element.remove();
		// If this was the currently-viewed session, go back to the upload view
		if (state.jobId === jobId) {
			state.jobId = null;
			if (_switchView) _switchView('upload');
		}
	} catch (err) {
		console.error('Delete active session failed:', err);
		alert('Failed to delete session.');
	}
}

async function _deleteSession(sessionId, element) {
	try {
		const res = await fetch(`/api/session/${sessionId}`, { method: 'DELETE' });
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		element.remove();
	} catch (err) {
		console.error('Delete session failed:', err);
		alert('Failed to delete session.');
	}
}

// ---------------------------------------------------------------------------
// Load a history session
// ---------------------------------------------------------------------------

/**
 * Handle a click on a history sidebar item.
 *
 * If the session has a video, jump straight to the results view.
 * If it only has a transcript, load it into the mapping view so the user
 * can assign new character names and re-run the pipeline.
 *
 * @param {object}      session   - Session metadata from the API.
 * @param {HTMLElement} element   - The clicked sidebar item (for active styling).
 * @param {function}	switchView - View-switching callback.
 */
async function loadHistorySession(session, element, switchView) {
	// Highlight the selected item and clear any previous active item in the past-sessions list only
	document.querySelectorAll('#history-list .history-item').forEach(el => el.classList.remove('active'));
	if (element) element.classList.add('active');

	try {
		const res = await fetch(`/api/resume/${session.id}`, { method: 'POST' });
		if (!res.ok) throw new Error(`Resume failed: HTTP ${res.status}`);
		const data = await res.json();

		if (_switchToSession) {
			await _switchToSession(data.job_id);
			const nameInput = document.getElementById('session-name-input');
			if (nameInput) nameInput.value = session.name || '';
		}
	} catch (err) {
		console.error('Error loading history session:', err);
		alert('Error loading this session.');
	}
}

/**
 * Load a specific provider-combination run from a history session.
 *
 * Parses the run key (e.g. "deepgram__claude__luma") into provider components,
 * calls the resume endpoint with those provider params, then loads the run's
 * video or transcript view.
 *
 * @param {object}      session   - Session metadata from the API.
 * @param {string}      runKey	- Provider combination key (e.g. "assembly__claude__luma").
 * @param {HTMLElement} element   - The clicked sidebar item (for active styling).
 * @param {function}	switchView - View-switching callback.
 */
async function loadHistorySessionWithKey(session, runKey, element, switchView) {
	document.querySelectorAll('#history-list .history-item').forEach(el => el.classList.remove('active'));
	if (element) element.classList.add('active');

	// Parse key into provider components: "transcriber__llm__video_gen"
	const parts = runKey.split('__');
	const params = new URLSearchParams();
	if (parts[0]) params.set('transcriber', parts[0]);
	if (parts[1]) params.set('llm', parts[1]);
	if (parts[2]) params.set('video_gen', parts[2]);

	try {
		const res = await fetch(`/api/resume/${session.id}?${params}`, { method: 'POST' });
		if (!res.ok) throw new Error(`Resume failed: HTTP ${res.status}`);
		const data = await res.json();

		state.jobId = data.job_id;

		if (_switchToSession) {
			await _switchToSession(data.job_id);
			const nameInput = document.getElementById('session-name-input');
			if (nameInput) nameInput.value = session.name || '';
		}
	} catch (err) {
		console.error('Error resuming session with key:', err);
		alert('Error loading this provider combination.');
	}
}
// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function _truncate(str, max) {
	if (!str) return 'Unnamed session';
	return str.length > max ? str.slice(0, max - 1) + '...' : str;
}
