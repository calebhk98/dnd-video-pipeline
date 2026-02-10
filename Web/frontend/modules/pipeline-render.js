/**
 * modules/pipeline-render.js - Pipeline result rendering helpers.
 *
 * Pure DOM-manipulation functions that display stage results in the pipeline
 * view.  Each function receives its data as arguments and writes to the
 * appropriate container element(s).
 */

import { state } from './state.js';
import { formatTime } from './pipeline-timers.js';

// ---------------------------------------------------------------------------
// Stage 1 render helpers
// ---------------------------------------------------------------------------

export function renderConvertedAudio(url) {
	const container = document.getElementById('converted-audio-container');
	if (!container) return;
	container.innerHTML = `
		<div class="converted-audio-player">
			<h4>Converted Audio</h4>
			<audio src="${url}" controls></audio>
		</div>
	`;
}

export function renderTranscript(lines) {
	const container = document.getElementById('transcript-container');
	if (!container) return;
	container.innerHTML = '';
	lines.forEach(line => {
		const p = document.createElement('p');
		p.className = 'transcript-line';
		p.dataset.speaker = line.speaker;
		p.innerHTML = `<strong>${line.speaker}:</strong> `;
		p.appendChild(document.createTextNode(line.text));
		container.appendChild(p);
	});
}

export function updateTranscriptWithNames(speakerMap) {
	const container = document.getElementById('transcript-container');
	if (!container) return;
	container.querySelectorAll('.transcript-line strong').forEach(strong => {
		const speakerId = strong.textContent.replace(/:$/, '').trim();
		const charName = speakerMap[speakerId];
		if (charName) strong.textContent = `${charName}:`;
	});
}

/**
 * Render the speaker-mapping input controls.
 *
 * @param {string[]} speakers         - Detected speaker IDs.
 * @param {Object|null} pendingSuggestions - LLM suggestions already received (may be null).
 */
export function renderMappingControls(speakers, pendingSuggestions) {
	const container = document.getElementById('mapping-controls');
	if (!container) return;
	container.innerHTML = '';
	// Sort speakers so they appear in a consistent order (A, B, C, ...)
	const sorted = [...speakers].sort();
	sorted.forEach(speaker => {
		const row = document.createElement('div');
		row.className = 'mapping-row';
		// If suggestions already arrived (WS beat the HTTP fetch), pre-fill immediately
		const suggested = pendingSuggestions?.[speaker] || '';
		row.innerHTML = `
			<label>${speaker}</label>
			<input type="text" value="${suggested}" placeholder="${suggested || 'Character Name'}" data-speaker="${speaker}">
		`;
		container.appendChild(row);
	});
	// If no suggestions yet, the placeholder stays as "Character Name" (no false loading text)
}

export function prefillSpeakerInputs(suggestions) {
	document.querySelectorAll('#mapping-controls input[data-speaker]').forEach(input => {
		const name = suggestions[input.dataset.speaker];
		if (name) {
			input.value = name;
			input.placeholder = name;
		} else {
			input.placeholder = 'Character Name';
		}
	});
}

/** Collect and return the current speaker-name mapping from the input controls. */
export function collectSpeakerMapping() {
	const mapping = {};
	document.querySelectorAll('#mapping-controls input[data-speaker]').forEach(input => {
		const name = input.value.trim();
		if (name) mapping[input.dataset.speaker] = name;
	});
	return mapping;
}

// ---------------------------------------------------------------------------
// Stage 2 render helpers
// ---------------------------------------------------------------------------

export function renderSpeakerMap(speakerMap) {
	const container = document.getElementById('speaker-map-display');
	if (!container) return;
	if (!speakerMap || Object.keys(speakerMap).length === 0) {
		container.innerHTML = '<p class="muted-text">No speaker mapping available.</p>';
		return;
	}
	const rows = Object.entries(speakerMap)
		.map(([id, name]) => `<tr><td class="speaker-id">${id}</td><td class="speaker-name">${name}</td></tr>`)
		.join('');
	container.innerHTML = `
		<h4>Speaker Mapping</h4>
		<table class="speaker-map-table">
			<thead><tr><th>Speaker ID</th><th>Character Name</th></tr></thead>
			<tbody>${rows}</tbody>
		</table>
	`;
}

/**
 * Render the speaker visualisation cards.
 *
 * @param {Object} vizMap           - Map of speakerId -> description string.
 * @param {Object} currentSpeakerMap - Map of speakerId -> character name (for display).
 */
export function renderSpeakerVisualization(vizMap, currentSpeakerMap) {
	const container = document.getElementById('speaker-visualization-display');
	if (!container) return;
	if (!vizMap || Object.keys(vizMap).length === 0) {
		container.innerHTML = '';
		return;
	}
	const cardsHtml = Object.entries(vizMap).map(([speakerId, description]) => {
		const charName = currentSpeakerMap[speakerId] || '';
		const nameHtml = charName
			? `<div class="speaker-viz-char-name">${charName}</div>`
			: '';
		return `
		<div class="speaker-viz-card" data-speaker="${speakerId}">
			<div class="speaker-viz-name">${speakerId}</div>
			${nameHtml}
			<textarea class="speaker-viz-textarea" rows="3">${description}</textarea>
			<div class="speaker-viz-actions">
				<button class="speaker-viz-save-btn secondary-btn" data-speaker="${speakerId}">Save</button>
				<span class="speaker-viz-saved-msg hidden">Saved!</span>
			</div>
		</div>`;
	}).join('');
	container.innerHTML = `
		<h4>Character Descriptions</h4>
		<div class="speaker-viz-cards">${cardsHtml}</div>
	`;
	container.querySelectorAll('.speaker-viz-save-btn').forEach(btn => {
		btn.addEventListener('click', async () => {
			const card = btn.closest('.speaker-viz-card');
			const savedMsg = card.querySelector('.speaker-viz-saved-msg');
			const updatedViz = {};
			container.querySelectorAll('.speaker-viz-card').forEach(c => {
				const id = c.dataset.speaker;
				const text = c.querySelector('.speaker-viz-textarea').value.trim();
				if (id && text) updatedViz[id] = text;
			});
			try {
				const res = await fetch(`/api/speaker_visualization/${state.jobId}`, {
					method: 'POST',
					headers: {'Content-Type': 'application/json'},
					body: JSON.stringify(updatedViz),
				});
				if (!res.ok) throw new Error('Save failed');
				savedMsg.classList.remove('hidden');
				setTimeout(() => savedMsg.classList.add('hidden'), 2000);
			} catch (err) {
				console.error('Failed to save speaker visualization:', err);
			}
		});
	});
}

export function renderStoryboard(scenes) {
	const container = document.getElementById('storyboard-display');
	if (!container) return;
	if (!scenes || scenes.length === 0) {
		container.innerHTML = '<p class="muted-text">No scenes generated.</p>';
		return;
	}
	const relevantCount = scenes.filter(s => s.is_relevant !== false).length;
	const cards = scenes.map(scene => {
		const start		= formatTime(scene.start_time);
		const end          = formatTime(scene.end_time);
		const isIrrelevant = scene.is_relevant === false;

		const irrelevantBadge = isIrrelevant
			? `<span class="irrelevant-badge" title="${scene.relevance_reason || 'Out-of-game content'}">Not included</span>`
			: '';

		const shotsHtml = (!isIrrelevant && scene.shots && scene.shots.length > 0)
			? `<div class="scene-shots">
                 <div class="scene-shots-label">${scene.shots.length} shot${scene.shots.length !== 1 ? 's' : ''}</div>
                 ${scene.shots.map(shot => `
                   <div class="shot-item">
                     <span class="shot-number">Shot ${shot.shot_number}</span>
                     <span class="shot-duration">${shot.duration_hint}s</span>
                     <span class="shot-description">${shot.description}</span>
                   </div>`).join('')}
               </div>`
			: '';

		return `
			<div class="scene-card${isIrrelevant ? ' irrelevant' : ''}" data-scene-number="${scene.scene_number}">
				<div class="scene-card-header">
					<span class="scene-number">Scene ${scene.scene_number}</span>
					<span class="scene-time">${start} - ${end}</span>
					${irrelevantBadge}
				</div>
				<div class="scene-location">${scene.location || ''}</div>
				<div class="scene-summary">${scene.summary || ''}</div>
				${scene.prompt && !isIrrelevant ? `<div class="scene-prompt muted-text">${scene.prompt}</div>` : ''}
				${shotsHtml}
				${!isIrrelevant ? '<div class="scene-video-slot hidden"></div>' : ''}
			</div>
		`;
	}).join('');
	const subtitle = relevantCount < scenes.length
		? `${relevantCount} of ${scenes.length} included`
		: `${scenes.length} scene${scenes.length !== 1 ? 's' : ''}`;
	container.innerHTML = `<h4>Storyboard <span class="storyboard-subtitle">(${subtitle})</span></h4><div class="scene-cards">${cards}</div>`;
}

// ---------------------------------------------------------------------------
// Stage 3 render helpers
// ---------------------------------------------------------------------------

export function injectSceneVideo(sceneNumber, videoUrl) {
	// Handle scene-card layout (live streaming during stage 3)
	const card = document.querySelector(`.scene-card[data-scene-number="${sceneNumber}"]`);
	if (card) {
		const slot = card.querySelector('.scene-video-slot');
		if (slot) {
			slot.innerHTML = `<video src="${videoUrl}" controls class="scene-video"></video>`;
			slot.classList.remove('hidden');
		}
	}

	// Handle scene-video-item layout (stage 3 results grid / retry)
	const item = document.querySelector(`.scene-video-item[data-scene-number="${sceneNumber}"]`);
	if (item) {
		item.classList.remove('failed');
		item.innerHTML = `
			<p class="scene-video-label">Scene ${sceneNumber}</p>
			<video src="${videoUrl}" controls class="scene-video"></video>
		`;
	}
}

export function renderSceneVideos(scenes) {
	const container = document.getElementById('scene-videos-grid');
	if (!container) return;
	if (!scenes || scenes.length === 0) {
		container.innerHTML = '<p class="muted-text">No scene videos available.</p>';
		return;
	}
	const items = scenes.map(scene => scene.video_url ? `
		<div class="scene-video-item" data-scene-number="${scene.scene_number}">
			<p class="scene-video-label">Scene ${scene.scene_number}</p>
			<video src="${scene.video_url}" controls class="scene-video"></video>
		</div>` : `
		<div class="scene-video-item failed" data-scene-number="${scene.scene_number}">
			<p class="scene-video-label">Scene ${scene.scene_number}</p>
			<p class="muted-text scene-failed-msg">Generation failed</p>
			<button class="retry-scene-btn" data-scene-number="${scene.scene_number}">Retry</button>
		</div>`
	).join('');
	container.innerHTML = `<div class="scene-video-grid">${items}</div>`;
}

// ---------------------------------------------------------------------------
// Stage 4 render helpers
// ---------------------------------------------------------------------------

export function renderFinalVideo(url) {
	const container = document.getElementById('final-video-container');
	if (!container) return;
	container.innerHTML = url
		? `<h4>Final Video</h4><video src="${url}" controls class="final-video"></video>`
		: '<p class="muted-text">No final video was produced.</p>';
}
