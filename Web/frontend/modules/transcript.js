/**
 * modules/transcript.js - Transcript display and speaker-mapping UI.
 *
 * Handles:
 *  - Fetching the transcript JSON from the server after upload.
 *  - Rendering transcript lines in the preview panel.
 *  - Building the speaker-mapping input form (one row per detected speaker).
 *  - Submitting the completed mapping to the server and starting the pipeline.
 */

import { state } from './state.js';
import { connectWebSocket } from './websocket.js';

/**
 * Initialise the "Generate Video" mapping-submit button listener.
 * Must be called once after the DOM is ready.
 *
 * @param {function} switchView - View-switching callback.
 */
export function initTranscript(switchView) {
	document.getElementById('submit-mapping-btn').addEventListener('click', () => {
		submitMapping(switchView);
	});
}

// ---------------------------------------------------------------------------
// Fetch & render
// ---------------------------------------------------------------------------

/**
 * Fetch the transcript for the current job from the server, render it, and
 * switch to the mapping view.
 *
 * @param {function} switchView - View-switching callback.
 */
export async function fetchTranscript(switchView) {
	try {
		const res = await fetch(`/api/transcript/${state.jobId}`);
		if (!res.ok) throw new Error('Transcript fetch failed');

		const data = await res.json();
		state.speakers = data.speakers_detected;

		renderTranscript(data.transcript);
		renderMappingControls(data.speakers_detected);
		switchView('mapping');
	} catch (err) {
		console.error('Transcript error:', err);
		alert('Error retrieving transcript from the server.');
	}
}

/**
 * Render transcript utterance lines into the preview panel.
 *
 * @param {{ speaker: string, text: string }[]} lines - Utterances to display.
 */
export function renderTranscript(lines) {
	const container = document.getElementById('transcript-container');
	container.innerHTML = '';
	lines.forEach(line => {
		const p = document.createElement('p');
		p.className = 'transcript-line';
		// Speaker label is highlighted via CSS; text is appended as plain text
		p.innerHTML = `<strong>${line.speaker}:</strong> `;
		p.appendChild(document.createTextNode(line.text));
		container.appendChild(p);
	});
}

/**
 * Build the speaker-mapping form: one row per detected speaker ID with a
 * text input for the user to enter the corresponding character name.
 *
 * @param {string[]} speakers - Speaker IDs returned by the transcriber.
 */
export function renderMappingControls(speakers) {
	const container = document.getElementById('mapping-controls');
	container.innerHTML = '';
	speakers.forEach(speaker => {
		const row = document.createElement('div');
		row.className = 'mapping-row';
		row.innerHTML = `
			<label>${speaker}</label>
			<input type="text" placeholder="Character Name" data-speaker="${speaker}">
		`;
		container.appendChild(row);
	});
}

// ---------------------------------------------------------------------------
// Submit mapping
// ---------------------------------------------------------------------------

/**
 * Collect speaker->name pairs from the mapping form, POST them to the server,
 * then switch to the loading view and open the WebSocket progress feed.
 *
 * @param {function} switchView - View-switching callback.
 */
async function submitMapping(switchView) {
	// Build mapping object from filled-in inputs (empty inputs are skipped)
	const mapping = {};
	document.querySelectorAll('#mapping-controls input').forEach(input => {
		const name = input.value.trim();
		if (name) mapping[input.dataset.speaker] = name;
	});

	try {
		switchView('loading');

		const res = await fetch(`/api/map_speakers/${state.jobId}`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(mapping),
		});

		if (!res.ok) throw new Error('Mapping submission failed');

		// Pipeline is now running in the background - open WS for live updates
		connectWebSocket(switchView);
	} catch (err) {
		console.error('Mapping error:', err);
		alert('Error submitting speaker mapping.');
		switchView('mapping');
	}
}
