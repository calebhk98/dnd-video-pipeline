/**
 * modules/upload.js - File selection, drag-and-drop, and upload logic.
 *
 * Handles:
 *  - Click-to-browse and drag-and-drop onto the upload zone.
 *  - Updating the UI when a file is selected.
 *  - Submitting the file + model selections to POST /api/upload.
 *  - Switching to the pipeline view on success (Stage 1 is NOT auto-started).
 */

import { state, registerSession } from './state.js';
import { initPipelineView } from './pipeline.js';

/**
 * Initialise all upload-view event listeners.
 * Must be called once after the DOM is ready.
 *
 * @param {function} switchView - Callback that transitions between named views.
 */
export function initUpload(switchView) {
	const dropZone          = document.getElementById('drop-zone');
	const fileInput         = document.getElementById('file-input');
	const uploadBtn         = document.getElementById('upload-btn');
	const autoRunToggle     = document.getElementById('auto-run-toggle');
	const transcriberSelect = document.getElementById('transcriber-select');
	const llmSelect         = document.getElementById('llm-select');
	const videoSelect       = document.getElementById('video-select');

	// --- Load persisted preferences from .env ---
	fetch('/api/settings')
		.then(r => r.json())
		.then(data => {
			if (data.AUTO_RUN) {
				const saved = data.AUTO_RUN === 'true';
				autoRunToggle.checked = saved;
				state.autoRun = saved;
			}
			if (data.DEFAULT_TRANSCRIBER) {
				transcriberSelect.value = data.DEFAULT_TRANSCRIBER;
				state.transcriber = data.DEFAULT_TRANSCRIBER;
			}
			if (data.DEFAULT_LLM) {
				llmSelect.value = data.DEFAULT_LLM;
				state.llm = data.DEFAULT_LLM;
			}
			if (data.DEFAULT_VIDEO_GEN) {
				videoSelect.value = data.DEFAULT_VIDEO_GEN;
				state.videoGen = data.DEFAULT_VIDEO_GEN;
			}
		})
		.catch(() => {});

	// --- Persist preferences on change ---
	autoRunToggle.addEventListener('change', () => {
		state.autoRun = autoRunToggle.checked;
		fetch('/api/settings', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ AUTO_RUN: String(autoRunToggle.checked) }),
		}).catch(() => {});
	});

	transcriberSelect.addEventListener('change', () => {
		fetch('/api/settings', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ DEFAULT_TRANSCRIBER: transcriberSelect.value }),
		}).catch(() => {});
	});

	llmSelect.addEventListener('change', () => {
		fetch('/api/settings', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ DEFAULT_LLM: llmSelect.value }),
		}).catch(() => {});
	});

	videoSelect.addEventListener('change', () => {
		fetch('/api/settings', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ DEFAULT_VIDEO_GEN: videoSelect.value }),
		}).catch(() => {});
	});

	// --- Drag-and-drop ---
	dropZone.addEventListener('click', () => fileInput.click());

	dropZone.addEventListener('dragover', (e) => {
		e.preventDefault();
		dropZone.classList.add('dragover');
	});

	dropZone.addEventListener('dragleave', () => {
		dropZone.classList.remove('dragover');
	});

	dropZone.addEventListener('drop', (e) => {
		e.preventDefault();
		dropZone.classList.remove('dragover');
		if (e.dataTransfer.files.length) {
			handleFileSelect(e.dataTransfer.files[0], dropZone, uploadBtn);
		}
	});

	// --- File picker (click-to-browse) ---
	fileInput.addEventListener('change', (e) => {
		if (e.target.files.length) {
			handleFileSelect(e.target.files[0], dropZone, uploadBtn);
		}
	});

	// --- Upload button ---
	uploadBtn.addEventListener('click', () => handleUpload(uploadBtn, switchView));
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function handleFileSelect(file, dropZone, uploadBtn) {
	state.file = file;
	dropZone.innerHTML = `<p>Selected: <strong>${file.name}</strong></p>`;
	uploadBtn.disabled = false;
}

async function handleUpload(uploadBtn, switchView) {
	if (!state.file) return;

	uploadBtn.disabled = true;
	uploadBtn.textContent = 'Uploading...';

	const speakerCount = document.getElementById('speaker-count').value;
	const transcriber  = document.getElementById('transcriber-select').value;
	const llm          = document.getElementById('llm-select').value;
	const videoGen     = document.getElementById('video-select').value;
	const autoRun      = document.getElementById('auto-run-toggle')?.checked ?? false;

	// Persist selections in shared state so pipeline.js can read them
	state.autoRun = autoRun;
	state.transcriber = transcriber;
	state.llm = llm;
	state.videoGen = videoGen;

	const formData = new FormData();
	formData.append('file',          state.file);
	formData.append('num_speakers',  speakerCount);
	formData.append('transcriber',   transcriber);
	formData.append('llm',           llm);
	formData.append('video_gen',     videoGen);
	formData.append('auto_run',      autoRun);

	try {
		const res = await fetch('/api/upload', { method: 'POST', body: formData });
		if (!res.ok) throw new Error('Upload failed');

		const data = await res.json();
		state.jobId = data.job_id;

		// Register this session in the active-sessions registry so the
		// sidebar can show it and the user can switch back to it later.
		registerSession(data.job_id, state.file.name);
		import('./history.js').then(mod => mod.renderActiveSessions?.());

		// Switch to pipeline view; Stage 1 has NOT started yet
		// initPipelineView shows the "Start Transcription" button (or
		// auto-starts if the toggle was checked).
		switchView('pipeline');
		initPipelineView(state.file.name, switchView);
	} catch (err) {
		console.error('Upload error:', err);
		alert('Error during upload. Check the browser console for details.');
		uploadBtn.disabled = false;
		uploadBtn.textContent = 'Upload & Process';
	}
}
