/**
 * modules/pipeline-timers.js - Phase elapsed-time tracking utilities.
 *
 * Owns the _phaseTimers state dict and all functions that start, stop, freeze,
 * and display per-phase timers.  Also owns sub-phase timers (individual steps
 * within a phase) and a live pipeline total timer that pauses between stages.
 *
 * Also exports the time-formatting helpers used by pipeline-render.js
 * (formatTime) and pipeline.js (formatElapsed).
 */

// ---------------------------------------------------------------------------
// Phase timers - track actual processing time per stage
// ---------------------------------------------------------------------------
// Each entry: { startedAt: epoch ms, intervalId: id, frozenMs: number|null }
const _phaseTimers = {};

// ---------------------------------------------------------------------------
// Sub-phase timers - track individual steps within a stage
// ---------------------------------------------------------------------------
// Each entry: { startedAt: epoch ms, intervalId: id, frozenMs: number|null, cached: bool }
const _subPhaseTimers = {};

// ---------------------------------------------------------------------------
// Pipeline total timer - live interval that pauses between stages
// ---------------------------------------------------------------------------
let _totalTimerInterval = null;

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

/** Clear all phase timers (call when resetting the pipeline UI for a new session). */
export function resetPhaseTimers() {
	for (const n of [1, 2, 3, 4]) {
		const timer = _phaseTimers[n];
		if (timer?.intervalId) clearInterval(timer.intervalId);
		delete _phaseTimers[n];
	}
	_stopPipelineTotalInterval();
	resetSubPhaseTimers();
}

/** Clear all sub-phase timers. */
export function resetSubPhaseTimers() {
	for (const key of Object.keys(_subPhaseTimers)) {
		const t = _subPhaseTimers[key];
		if (t?.intervalId) clearInterval(t.intervalId);
		// Hide the DOM row
		const row = document.getElementById(`sub-timer-${key}`);
		if (row) {
			row.classList.add("hidden");
			row.classList.remove("running", "cached");
		}
		const valEl = document.getElementById(`sub-timer-${key}-val`);
		if (valEl) valEl.textContent = "-";
		delete _subPhaseTimers[key];
	}
}

/** Start the live counting timer for a phase. */
export function startPhaseTimer(n, startEpoch) {
	if (_phaseTimers[n]?.intervalId) return; // already running
	const startedAt = startEpoch || Date.now();
	const timerEl = document.getElementById(`phase-${n}-timer`);
	const update = () => {
		const elapsed = Date.now() - startedAt;
		if (timerEl) timerEl.textContent = formatElapsed(elapsed);
	};
	update();
	const intervalId = setInterval(update, 500);
	_phaseTimers[n] = { startedAt, intervalId, frozenMs: null };
	_startPipelineTotalInterval();
}

/** Stop the live timer and freeze it at the final elapsed time. */
export function stopPhaseTimer(n, endEpoch) {
	const timer = _phaseTimers[n];
	if (!timer) return;
	if (timer.intervalId) {
		clearInterval(timer.intervalId);
		timer.intervalId = null;
	}
	const frozenMs =
		timer.frozenMs ?? (endEpoch || Date.now()) - timer.startedAt;
	timer.frozenMs = frozenMs;
	const timerEl = document.getElementById(`phase-${n}-timer`);
	if (timerEl) timerEl.textContent = formatElapsed(frozenMs);
	// Stop total interval if no phase is currently running
	const anyRunning = [1, 2, 3, 4].some(
		(i) => _phaseTimers[i]?.intervalId != null,
	);
	if (!anyRunning) _stopPipelineTotalInterval();
	updateTotalTime();
}

/** Show a frozen timer for a phase that was already completed (e.g. on page hydration). */
export function freezePhaseTimer(n, elapsedMs) {
	if (_phaseTimers[n]?.intervalId) clearInterval(_phaseTimers[n].intervalId);
	_phaseTimers[n] = {
		startedAt: null,
		intervalId: null,
		frozenMs: elapsedMs,
	};
	const timerEl = document.getElementById(`phase-${n}-timer`);
	if (timerEl) timerEl.textContent = formatElapsed(elapsedMs);
	updateTotalTime();
}

// ---------------------------------------------------------------------------
// Sub-phase timer functions
// ---------------------------------------------------------------------------

/**
 * Start a live sub-phase timer.
 * @param {string} key   - e.g. "1-convert", "2-speakers"
 * @param {string} label - Display label (unused if DOM row already has it)
 */
export function startSubPhaseTimer(key, label) {
	// Don't restart an already-running timer (prevents reset from repeated progress messages)
	if (_subPhaseTimers[key]?.intervalId) {
		return;
	}
	const startedAt = Date.now();
	const valEl = document.getElementById(`sub-timer-${key}-val`);
	const row = document.getElementById(`sub-timer-${key}`);

	if (row) {
		row.classList.remove("hidden", "cached");
		row.classList.add("running");
	}

	const update = () => {
		const elapsed = Date.now() - startedAt;
		if (valEl) valEl.textContent = formatElapsed(elapsed);
	};
	update();
	const intervalId = setInterval(update, 500);
	_subPhaseTimers[key] = {
		startedAt,
		intervalId,
		frozenMs: null,
		cached: false,
	};
}

/**
 * Freeze a running sub-phase timer at the current elapsed time.
 * @param {string} key - e.g. "1-convert"
 */
export function stopSubPhaseTimer(key) {
	const timer = _subPhaseTimers[key];
	if (!timer) return;
	if (timer.intervalId) {
		clearInterval(timer.intervalId);
		timer.intervalId = null;
	}
	const frozenMs = timer.frozenMs ?? Date.now() - timer.startedAt;
	timer.frozenMs = frozenMs;
	const valEl = document.getElementById(`sub-timer-${key}-val`);
	if (valEl) valEl.textContent = formatElapsed(frozenMs);
	const row = document.getElementById(`sub-timer-${key}`);
	if (row) row.classList.remove("running");
}

/**
 * Show a frozen sub-phase timer for a step that already completed (e.g. on session hydration).
 * @param {string} key       - e.g. "1-convert"
 * @param {number} elapsedMs - Pre-recorded elapsed time in milliseconds.
 */
export function freezeSubPhaseTimer(key, elapsedMs) {
	if (_subPhaseTimers[key]?.intervalId) {
		clearInterval(_subPhaseTimers[key].intervalId);
	}
	_subPhaseTimers[key] = {
		startedAt: null,
		intervalId: null,
		frozenMs: elapsedMs,
		cached: false,
	};
	const valEl = document.getElementById(`sub-timer-${key}-val`);
	if (valEl) valEl.textContent = formatElapsed(elapsedMs);
	const row = document.getElementById(`sub-timer-${key}`);
	if (row) {
		row.classList.remove("hidden", "running", "cached");
	}
}

/**
 * Show a sub-phase as instantly "cached" (no meaningful elapsed time).
 * @param {string} key   - e.g. "1-convert"
 */
export function markSubPhaseCached(key) {
	if (_subPhaseTimers[key]?.intervalId) {
		clearInterval(_subPhaseTimers[key].intervalId);
	}
	_subPhaseTimers[key] = {
		startedAt: null,
		intervalId: null,
		frozenMs: 0,
		cached: true,
	};
	const valEl = document.getElementById(`sub-timer-${key}-val`);
	if (valEl) valEl.textContent = "cached";
	const row = document.getElementById(`sub-timer-${key}`);
	if (row) {
		row.classList.remove("hidden", "running");
		row.classList.add("cached");
	}
}

// ---------------------------------------------------------------------------
// Pipeline total timer
// ---------------------------------------------------------------------------

/** Recalculate and display the total pipeline time from all frozen + live phase timers. */
export function updateTotalTime() {
	let totalMs = 0;
	let hasData = false;
	for (const n of [1, 2, 3, 4]) {
		const t = _phaseTimers[n];
		if (t?.frozenMs != null) {
			totalMs += t.frozenMs;
			hasData = true;
		} else if (t?.startedAt != null && t?.intervalId != null) {
			// Phase is currently running add live elapsed
			totalMs += Date.now() - t.startedAt;
			hasData = true;
		}
	}
	const el = document.getElementById("pipeline-total-time");
	if (!el) return;
	if (hasData) {
		el.textContent = `Pipeline: ${formatElapsed(totalMs)}`;
		el.classList.remove("hidden");
	} else {
		el.classList.add("hidden");
	}
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

function _startPipelineTotalInterval() {
	if (_totalTimerInterval) return;
	_totalTimerInterval = setInterval(updateTotalTime, 500);
}

function _stopPipelineTotalInterval() {
	if (_totalTimerInterval) {
		clearInterval(_totalTimerInterval);
		_totalTimerInterval = null;
	}
	updateTotalTime(); // final frozen snapshot
}

// ---------------------------------------------------------------------------
// Formatting helpers (also used by pipeline-render.js and pipeline.js)
// ---------------------------------------------------------------------------

/** Format elapsed milliseconds as "1m 23s" or "45s". */
export function formatElapsed(ms) {
	const totalSec = Math.floor(ms / 1000);
	const m = Math.floor(totalSec / 60);
	const s = totalSec % 60;
	return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

/** Format a timestamp in seconds as "m:ss". */
export function formatTime(seconds) {
	if (seconds == null) return "?";
	const m = Math.floor(seconds / 60);
	const s = Math.floor(seconds % 60);
	return `${m}:${String(s).padStart(2, "0")}`;
}
