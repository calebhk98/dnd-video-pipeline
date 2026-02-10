/**
 * modules/pipeline-phase.js - Phase panel UI state helpers.
 *
 * Updates badge, status text, progress bar, and detail text for each of the
 * four pipeline phase panels.  Delegates timer management to pipeline-timers.js.
 */

import { startPhaseTimer, stopPhaseTimer } from './pipeline-timers.js';

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

/** Reveal the phase panel for stage n (removes 'hidden' class). */
export function showPhase(n) {
	document.getElementById(`phase-${n}`)?.classList.remove('hidden');
}

/** Mark a phase as actively running: set badge, status text, progress bar, and start timer. */
export function setPhaseRunning(n) {
	const badge  = document.getElementById(`phase-${n}-badge`);
	const status = document.getElementById(`phase-${n}-status`);
	const bar	= document.getElementById(`phase-${n}-bar`);

	badge?.classList.remove('done', 'error');
	badge?.classList.add('running');
	if (status) status.textContent = 'Running...';
	if (bar) bar.style.width = '5%';

	// Start the live elapsed timer for this phase
	startPhaseTimer(n);

	// Disable the next stage's action button while this stage runs
	const nextBtnIds = ['', 'run-stage2-btn', 'run-stage3-btn', 'run-stage4-btn'];
	const nextBtn = document.getElementById(nextBtnIds[n]);
	if (nextBtn) nextBtn.disabled = true;
}

/** Update the detail text and progress bar width for the running phase. */
export function updatePhaseProgress(n, detail, percent) {
	const detailEl = document.getElementById(`phase-${n}-detail`);
	const bar      = document.getElementById(`phase-${n}-bar`);
	if (detailEl && detail) detailEl.textContent = detail;
	if (bar && percent != null) bar.style.width = `${percent}%`;
}

/** Mark a phase as complete: update badge, fill bar to 100%, freeze timer. */
export function setPhaseComplete(n, message) {
	const badge	= document.getElementById(`phase-${n}-badge`);
	const status   = document.getElementById(`phase-${n}-status`);
	const bar      = document.getElementById(`phase-${n}-bar`);
	const progress = document.getElementById(`phase-${n}-progress`);

	badge?.classList.remove('running', 'error');
	badge?.classList.add('done');
	if (status) status.textContent = message || 'Complete';
	if (bar) bar.style.width = '100%';
	if (progress) setTimeout(() => progress.classList.add('hidden'), 800);

	// Freeze the elapsed timer for this phase
	stopPhaseTimer(n);

	// Re-enable the next stage's action button now that this stage is done
	const nextBtnIds = ['', 'run-stage2-btn', 'run-stage3-btn', 'run-stage4-btn'];
	const nextBtn = document.getElementById(nextBtnIds[n]);
	if (nextBtn) nextBtn.disabled = false;
}

/** Mark a phase as errored and hide all subsequent phase panels. */
export function setPhaseError(n, message) {
	const badge  = document.getElementById(`phase-${n}-badge`);
	const status = document.getElementById(`phase-${n}-status`);
	const detail = document.getElementById(`phase-${n}-detail`);

	badge?.classList.remove('running', 'done');
	badge?.classList.add('error');
	if (status) status.textContent = 'Error';
	if (detail) detail.textContent = message || 'An error occurred.';

	// Hide all subsequent phases so the user cannot proceed past a failed stage
	for (let i = n + 1; i <= 4; i++) {
		document.getElementById(`phase-${i}`)?.classList.add('hidden');
	}
}

/** Reveal the results panel for a completed phase. */
export function revealPhaseResults(n) {
	document.getElementById(`phase-${n}-results`)?.classList.remove('hidden');
}

/** Lock or unlock a list of <select> elements by id. */
export function setSelectsLocked(ids, locked) {
	ids.forEach(id => {
		const el = document.getElementById(id);
		if (el) el.disabled = locked;
	});
}
