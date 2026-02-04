/**
 * modules/pipeline.js - Phase-by-phase pipeline controller.
 *
 * Manages the pipeline-view: shows a "Start Transcription" button initially
 * (or auto-starts if state.autoRun is set), connects to the WebSocket once
 * Stage 1 begins, routes stage progress messages to the correct phase panel,
 * renders per-stage results, and wires up the "Run Next Stage" buttons.
 *
 * Auto-run chain (when state.autoRun === true):
 *   Upload -> auto-start Stage 1 -> auto-advance to Stage 2 (using LLM
 *   suggestions) -> auto-advance to Stage 3 -> auto-advance to Stage 4.
 *
 * Multi-session support:
 *   `handleNewSession(switchView)` - navigate back to the upload screen while
 *     leaving the current job running in the background.
 *   `switchToSession(jobId, switchView)` - switch the pipeline view to a
 *     different job that is already in progress.
 */

import { state, updateSessionStatus, renameSession } from "./state.js";
import { connectWebSocket, reattachHandler } from "./websocket.js";
import {
	startPhaseTimer,
	stopPhaseTimer,
	freezePhaseTimer,
	resetPhaseTimers,
	startSubPhaseTimer,
	stopSubPhaseTimer,
	markSubPhaseCached,
	resetSubPhaseTimers,
	freezeSubPhaseTimer,
} from "./pipeline-timers.js";
import {
	showPhase,
	setPhaseRunning,
	updatePhaseProgress,
	setPhaseComplete,
	setPhaseError,
	revealPhaseResults,
	setSelectsLocked,
} from "./pipeline-phase.js";
import {
	renderConvertedAudio,
	renderTranscript,
	updateTranscriptWithNames,
	renderMappingControls,
	prefillSpeakerInputs,
	collectSpeakerMapping,
	renderSpeakerMap,
	renderSpeakerVisualization,
	renderStoryboard,
	injectSceneVideo,
	renderSceneVideos,
	renderFinalVideo,
} from "./pipeline-render.js";

// ---------------------------------------------------------------------------
// Module-level switchView reference (set on first initPipelineView call)
// ---------------------------------------------------------------------------
let _switchView = null;

// ---------------------------------------------------------------------------
// Speaker suggestion cache (fixes race condition where WS arrives before DOM)
// ---------------------------------------------------------------------------
let _pendingSuggestions = null;
let _currentSpeakerMap = {};

// ---------------------------------------------------------------------------
// Public init - called by upload.js after a successful upload
// ---------------------------------------------------------------------------

/**
 * Prepare the pipeline view for a new job.
 *
 * @param {string}   filename   - Original uploaded filename shown in the Phase 1 card.
 * @param {function} switchView - View-switching callback from app.js.
 */
export function initPipelineView(filename, switchView) {
	if (switchView) _switchView = switchView;

	// Show the uploaded filename in the Phase 1 "ready" panel
	const fnEl = document.getElementById("phase-1-filename");
	if (fnEl) fnEl.textContent = `File ready: ${filename || ""}`;

	// Initialise pipeline dropdowns to match the selections made on the upload screen
	const transEl = document.getElementById("pipeline-transcriber-select");
	if (transEl) transEl.value = state.transcriber;
	const llmEl = document.getElementById("pipeline-llm-select");
	if (llmEl) llmEl.value = state.llm;
	const s2llmEl = document.getElementById("pipeline-stage2-llm-select");
	if (s2llmEl) s2llmEl.value = state.llm;
	const videoEl = document.getElementById("pipeline-video-select");
	if (videoEl) videoEl.value = state.videoGen;

	// Initialise the session name input (strip file extension as default name)
	const nameInput = document.getElementById("session-name-input");
	if (nameInput) {
		nameInput.value = (filename || "").replace(/\.[^.]+$/, "");
		nameInput.oninput = () => {
			renameSession(state.jobId, nameInput.value || filename);
			import("./history.js").then((mod) => mod.renderActiveSessions?.());
		};
	}

	_resetPipelineUI();
	_setupButtonListeners();

	if (state.autoRun) {
		// Skip the manual start ,  kick off Stage 1 right away
		_handleStartStage1();
	}
}

// ---------------------------------------------------------------------------
// Multi-session navigation
// ---------------------------------------------------------------------------

/**
 * Navigate back to the upload view without stopping the current pipeline.
 * The running job continues in the background; the user can return to it by
 * clicking its entry in the "Active Sessions" sidebar section.
 *
 * @param {function} switchView - View-switching callback.
 */
export function handleNewSession(switchView) {
	if (switchView) _switchView = switchView;
	// Reset the upload form so it is ready for a fresh file
	const dropZone = document.getElementById("drop-zone");
	const uploadBtn = document.getElementById("upload-btn");
	const fileInput = document.getElementById("file-input");
	if (dropZone)
		dropZone.innerHTML =
			"<p>Drag & drop an audio file here, or click to browse</p>";
	if (uploadBtn) {
		uploadBtn.disabled = true;
		uploadBtn.textContent = "Upload & Process";
	}
	if (fileInput) fileInput.value = "";
	state.file = null;

	if (_switchView) _switchView("upload");
}

/**
 * Switch the pipeline view to a different active session.
 *
 * Re-hydrates the UI from the server for any stages that have already
 * completed, then (re)attaches the WebSocket handler so live progress
 * messages update this session's UI.
 *
 * @param {string}   jobId      - The job UUID to switch to.
 * @param {function} switchView - View-switching callback.
 */
export async function switchToSession(jobId, switchView) {
	if (switchView) _switchView = switchView;

	state.jobId = jobId;

	if (_switchView) _switchView("pipeline");

	// Blank the pipeline view before we fill it in
	_resetPipelineUI();
	_setupButtonListeners();

	// Update the filename label and session name input if we can identify it
	const { sessions } = await import("./state.js");
	const session = sessions[jobId];
	const fnEl = document.getElementById("phase-1-filename");
	if (fnEl && session)
		fnEl.textContent = `File ready: ${session.filename || ""}`;
	const nameInput = document.getElementById("session-name-input");
	if (nameInput) {
		nameInput.value = session
			? session.name || (session.filename || "").replace(/\.[^.]+$/, "")
			: "";
		nameInput.oninput = () => {
			if (sessions[jobId]) {
				// Active session: update in-memory registry and refresh sidebar
				renameSession(jobId, nameInput.value || session.filename);
				import("./history.js").then((mod) =>
					mod.renderActiveSessions?.(),
				);
			} else {
				// Past session: persist name to the server via PATCH, then refresh history list
				const fd = new FormData();
				fd.append("name", nameInput.value);
				fetch(`/api/job/${jobId}`, { method: "PATCH", body: fd })
					.then(() =>
						import("./history.js").then((mod) =>
							mod.fetchHistory?.(_switchView),
						),
					)
					.catch(() => {});
			}
		};
	}

	// Try to re-hydrate completed stages from the server
	await _rehydrateSession(jobId);

	// Attach (or re-attach) the WebSocket handler so live updates flow in
	const attached = reattachHandler(jobId, _handleMessage);
	if (!attached) {
		// Connection was closed or never opened for this job -- open a new one
		connectWebSocket(jobId, _handleMessage);
	}

	// Clear past-session highlights and re-render active sessions to reflect new selection
	document
		.querySelectorAll("#history-list .history-item")
		.forEach((el) => el.classList.remove("active"));
	_refreshActiveSessions();
}

// ---------------------------------------------------------------------------
// Re-hydrate a session whose stages may have already completed
// ---------------------------------------------------------------------------

async function _rehydrateSession(jobId) {
	// Ask the server what stage this job is at, including stored stage timings
	let resumeData = null;
	try {
		const res = await fetch(`/api/job_status/${jobId}`);
		if (res.ok) resumeData = await res.json();
	} catch (_) {
		/* endpoint may not exist -- fall through */
	}

	// Restore frozen phase timers from server-recorded stage timings
	if (resumeData?.stage_timings) {
		const timings = resumeData.stage_timings;
		for (const [n, t] of Object.entries(timings)) {
			if (t.started_at && t.ended_at) {
				const elapsedMs = new Date(t.ended_at) - new Date(t.started_at);
				if (elapsedMs > 0) freezePhaseTimer(Number(n), elapsedMs);
			}
		}
	}

	// Restore frozen sub-phase timers from server-recorded sub-stage timings
	if (resumeData?.sub_stage_timings) {
		const subTimings = resumeData.sub_stage_timings;
		for (const [key, t] of Object.entries(subTimings)) {
			if (t.started_at && t.ended_at) {
				const elapsedMs = new Date(t.ended_at) - new Date(t.started_at);
				if (elapsedMs > 0) freezeSubPhaseTimer(key, elapsedMs);
			}
		}
	}

	// Try to load transcript (Stage 1 results)
	try {
		const res = await fetch(`/api/transcript/${jobId}`);
		if (res.ok) {
			const data = await res.json();
			if (data.converted_audio_url) {
				renderConvertedAudio(data.converted_audio_url);
				document
					.getElementById("converted-audio-container")
					?.classList.remove("hidden");
			}
			renderTranscript(data.transcript);
			renderMappingControls(data.speakers_detected, _pendingSuggestions);
			state.speakers = data.speakers_detected || [];
			setPhaseComplete(1, "Transcription complete");
			revealPhaseResults(1);
			if (resumeData)
				_updateVersionNavBtns(
					1,
					resumeData.stage1_ver_count ?? 0,
					resumeData.stage1_cur_ver ?? -1,
				);
			showPhase(2);
		}
	} catch (_) {
		/* stage 1 not done yet */
	}

	// Try to load Stage 2 results
	try {
		const res = await fetch(`/api/stage2_results/${jobId}`);
		if (res.ok) {
			const data = await res.json();
			if (data.speaker_map) {
				_currentSpeakerMap = data.speaker_map;
				prefillSpeakerInputs(data.speaker_map);
			}
			renderSpeakerMap(data.speaker_map);
			updateTranscriptWithNames(data.speaker_map || {});
			if (data.speaker_visualization)
				renderSpeakerVisualization(
					data.speaker_visualization,
					_currentSpeakerMap,
				);
			renderStoryboard(data.scenes);
			_updateVersionNavBtns(2, data.ver_count ?? 0, data.cur_ver ?? -1);
			setPhaseComplete(2, "LLM processing complete");
			revealPhaseResults(2);
			// Show Phase 3
			showPhase(3);
		}
	} catch (_) {
		/* stage 2 not done yet */
	}

	// Try to load Stage 3 results
	try {
		const res = await fetch(`/api/stage3_results/${jobId}`);
		if (res.ok) {
			const data = await res.json();
			renderSceneVideos(data.scenes);
			for (const scene of data.scenes) {
				if (scene.video_url)
					injectSceneVideo(scene.scene_number, scene.video_url);
			}
			setPhaseComplete(3, "Video generation complete");
			revealPhaseResults(3);
			if (resumeData)
				_updateVersionNavBtns(
					3,
					resumeData.stage3_ver_count ?? 0,
					resumeData.stage3_cur_ver ?? -1,
				);
			// Show Phase 4
			showPhase(4);
		}
	} catch (_) {
		/* stage 3 not done yet */
	}

	// Try to load final video (Stage 4 results)
	try {
		const res = await fetch(`/api/videos/${jobId}`);
		if (res.ok) {
			const data = await res.json();
			if (data.status === "completed" && data.videos?.length) {
				renderFinalVideo(data.videos[0]);
				setPhaseComplete(4, "Assembly complete");
				revealPhaseResults(4);
				if (resumeData)
					_updateVersionNavBtns(
						4,
						resumeData.stage4_ver_count ?? 0,
						resumeData.stage4_cur_ver ?? -1,
					);
			}
		}
	} catch (_) {
		/* stage 4 not done yet */
	}
}

// ---------------------------------------------------------------------------
// WebSocket message handler
// ---------------------------------------------------------------------------

function _handleMessage(data) {
	const { status, stage, detail } = data;

	if (status === "error") {
		const phaseNum = _stageToPhaseNum(stage);
		stopPhaseTimer(phaseNum || 1);
		const _phaseSubTimerKeys = {
			1: ["1-convert", "1-upload", "1-transcribe"],
			2: [
				"2-speakers",
				"2-viz",
				"2-relevance",
				"2-storyboard",
				"2-production",
			],
			3: ["3-scenes"],
			4: ["4-stitch", "4-captions", "4-audio"],
		};
		for (const key of _phaseSubTimerKeys[phaseNum] ?? []) {
			stopSubPhaseTimer(key);
		}
		setPhaseError(phaseNum || 1, detail);

		// Re-enable the action button and model select so the user can fix the
		// issue (add credits, swap provider, reconnect) and retry the stage.
		if (phaseNum === 1) {
			document
				.getElementById("start-stage1-btn")
				?.removeAttribute("disabled");
			setSelectsLocked(
				["pipeline-transcriber-select", "pipeline-llm-select"],
				false,
			);
			_setDownstreamRunsBlocked(1, false);
		} else if (phaseNum === 2) {
			document
				.getElementById("run-stage2-btn")
				?.removeAttribute("disabled");
			setSelectsLocked(["pipeline-stage2-llm-select"], false);
			_setDownstreamRunsBlocked(2, false);
		} else if (phaseNum === 3) {
			document
				.getElementById("run-stage3-btn")
				?.removeAttribute("disabled");
			setSelectsLocked(["pipeline-video-select"], false);
			_setDownstreamRunsBlocked(3, false);
		} else if (phaseNum === 4) {
			document
				.getElementById("run-stage4-btn")
				?.removeAttribute("disabled");
		}

		updateSessionStatus(state.jobId, "error");
		_refreshActiveSessions();
		return;
	}

	if (status === "stage_started") {
		const phaseNum = _stageToPhaseNum(stage);
		if (phaseNum)
			startPhaseTimer(
				phaseNum,
				data.timestamp
					? new Date(data.timestamp).getTime()
					: Date.now(),
			);
		// Start sub-phase timers for stages whose first sub-phase begins at stage_started
		if (stage === "stage2")
			startSubPhaseTimer("2-speakers", "Speaker mapping");
		if (stage === "stage3")
			startSubPhaseTimer("3-scenes", "Scene generation");
		return;
	}

	if (status === "processing") {
		const phaseNum = _stageToPhaseNum(stage);
		if (phaseNum) {
			// Start timer if not yet started (fallback if stage_started wasn't received)
			startPhaseTimer(phaseNum);
			updatePhaseProgress(
				phaseNum,
				detail,
				_stageToPercent(stage, detail),
			);
		}
		// Sub-phase transitions driven by detail text
		if (detail?.includes("Converting")) {
			startSubPhaseTimer("1-convert", "Convert to WAV");
		} else if (detail?.includes("Uploading") && stage?.includes("1/4")) {
			stopSubPhaseTimer("1-convert");
			startSubPhaseTimer("1-upload", "Upload to service");
		} else if (detail?.includes("Transcribing")) {
			stopSubPhaseTimer("1-upload");
			startSubPhaseTimer("1-transcribe", "Transcription");
		} else if (detail?.includes("speaker visualizations")) {
			stopSubPhaseTimer("2-speakers");
			startSubPhaseTimer("2-viz", "Speaker visuals");
		} else if (detail?.includes("storyboard")) {
			stopSubPhaseTimer("2-viz");
			startSubPhaseTimer("2-storyboard", "Storyboard");
		} else if (
			detail?.includes("Reviewing scene") ||
			detail?.includes("relevance")
		) {
			stopSubPhaseTimer("2-storyboard");
			startSubPhaseTimer("2-relevance", "Relevance review");
		} else if (detail?.includes("production script")) {
			stopSubPhaseTimer("2-relevance");
			startSubPhaseTimer("2-production", "Production script");
		} else if (detail?.includes("Stitching")) {
			startSubPhaseTimer("4-stitch", "Stitch clips");
		} else if (detail?.includes("Adding captions")) {
			stopSubPhaseTimer("4-stitch");
			startSubPhaseTimer("4-captions", "Add captions");
		} else if (detail?.includes("Overlaying audio")) {
			stopSubPhaseTimer("4-captions");
			startSubPhaseTimer("4-audio", "Audio overlay");
		}
		return;
	}

	if (status === "wav_ready") {
		updatePhaseProgress(1, data.detail, 35);
		if (data.wav_url) {
			renderConvertedAudio(data.wav_url);
			document
				.getElementById("converted-audio-container")
				?.classList.remove("hidden");
		}
		// Sub-phase: handle cached vs freshly converted WAV
		if (data.detail?.includes("Reusing")) {
			markSubPhaseCached("1-convert");
		} else {
			stopSubPhaseTimer("1-convert");
		}
		startSubPhaseTimer("1-upload", "Upload to service");
		return;
	}

	if (status === "scene_ready") {
		injectSceneVideo(data.scene_number, data.video_url);
		return;
	}

	if (status === "stage_complete") {
		const completedPhase = _stageToPhaseNum(stage);
		if (completedPhase)
			stopPhaseTimer(
				completedPhase,
				data.timestamp
					? new Date(data.timestamp).getTime()
					: Date.now(),
			);
		if (stage === "stage1") {
			stopSubPhaseTimer("1-convert");
			stopSubPhaseTimer("1-upload");
			stopSubPhaseTimer("1-transcribe");
			_onStage1Complete(detail);
			return;
		}
		if (stage === "stage2") {
			stopSubPhaseTimer("2-production");
			_onStage2Complete(detail);
			return;
		}
		if (stage === "stage3") {
			stopSubPhaseTimer("3-scenes");
			_onStage3Complete(detail);
			return;
		}
		if (stage === "stage4") {
			stopSubPhaseTimer("4-audio");
			_onStage4Complete();
			return;
		}
	}

	if (status === "speaker_suggestions") {
		_onSpeakerSuggestions(data.suggestions || {});
		return;
	}

	if (status === "speaker_map_ready") {
		_onSpeakerMapReady(data.speaker_map || {});
		return;
	}

	if (status === "speaker_visualization_ready") {
		renderSpeakerVisualization(
			data.speaker_visualization || {},
			_currentSpeakerMap,
		);
		return;
	}

	if (status === "completed") {
		_onStage4Complete();
	}
}

// ---------------------------------------------------------------------------
// Stage completion handlers
// ---------------------------------------------------------------------------

async function _onStage1Complete(detail) {
	const label =
		detail && detail.startsWith("Using cached")
			? detail
			: "Transcription complete";
	setPhaseComplete(1, label);
	setSelectsLocked(
		["pipeline-transcriber-select", "pipeline-llm-select"],
		false,
	);

	try {
		const res = await fetch(`/api/transcript/${state.jobId}`);
		if (!res.ok) throw new Error("Transcript fetch failed");
		const data = await res.json();

		if (data.converted_audio_url) {
			renderConvertedAudio(data.converted_audio_url);
			document
				.getElementById("converted-audio-container")
				?.classList.remove("hidden");
		}
		renderTranscript(data.transcript);
		renderMappingControls(data.speakers_detected, _pendingSuggestions);
		revealPhaseResults(1);
		// Re-enable the start button and downstream stages now stage 1 is done
		document
			.getElementById("start-stage1-btn")
			?.removeAttribute("disabled");
		setSelectsLocked(
			["pipeline-transcriber-select", "pipeline-llm-select"],
			false,
		);
		_setDownstreamRunsBlocked(1, false);
	} catch (err) {
		console.error("Failed to load transcript:", err);
		setPhaseError(1, "Failed to load transcript results.");
	}

	showPhase(2);
}

function _onSpeakerSuggestions(suggestions) {
	_pendingSuggestions = suggestions;
	prefillSpeakerInputs(suggestions);

	if (state.autoRun) {
		_handleRunStage2WithMapping(suggestions);
	}
}

function _onSpeakerMapReady(speakerMap) {
	_currentSpeakerMap = speakerMap || {};
	renderSpeakerMap(speakerMap);
	updateTranscriptWithNames(speakerMap);
	revealPhaseResults(2);
}

async function _onStage2Complete(detail) {
	const label =
		detail && detail.startsWith("Using cached")
			? detail
			: "LLM processing complete";
	setPhaseComplete(2, label);
	setSelectsLocked(["pipeline-stage2-llm-select"], false);

	try {
		const res = await fetch(`/api/stage2_results/${state.jobId}`);
		if (!res.ok) throw new Error("Stage 2 results fetch failed");
		const data = await res.json();

		renderSpeakerMap(data.speaker_map);
		if (data.speaker_visualization)
			renderSpeakerVisualization(
				data.speaker_visualization,
				_currentSpeakerMap,
			);
		renderStoryboard(data.scenes);
		_updateVersionNavBtns(2, data.ver_count ?? 0, data.cur_ver ?? -1);
		revealPhaseResults(2);
		// Re-enable the run button and downstream stages now stage 2 is done
		document.getElementById("run-stage2-btn")?.removeAttribute("disabled");
		_setDownstreamRunsBlocked(2, false);
	} catch (err) {
		console.error("Failed to load Stage 2 results:", err);
		setPhaseError(2, "Failed to load LLM results.");
	}

	// Reveal Phase 3
	showPhase(3);
	document.getElementById("phase-3-start")?.classList.remove("hidden"); // reveal video model selector

	if (state.autoRun) _handleRunStage3();
}

async function _onStage3Complete(detail) {
	const label =
		detail && detail.startsWith("Using cached")
			? detail
			: "Video generation complete";
	setPhaseComplete(3, label);
	setSelectsLocked(["pipeline-video-select"], false);

	try {
		const res = await fetch(`/api/stage3_results/${state.jobId}`);
		if (!res.ok) throw new Error("Stage 3 results fetch failed");
		const data = await res.json();

		renderSceneVideos(data.scenes);
		for (const scene of data.scenes) {
			if (scene.video_url)
				injectSceneVideo(scene.scene_number, scene.video_url);
		}
		revealPhaseResults(3);
		// Re-enable the run button, update version nav, and re-enable stage 4
		document.getElementById("run-stage3-btn")?.removeAttribute("disabled");
		const job3 = await _fetchVersionInfo(3);
		if (job3) _updateVersionNavBtns(3, job3.ver_count, job3.cur_ver);
		_setDownstreamRunsBlocked(3, false);
	} catch (err) {
		console.error("Failed to load Stage 3 results:", err);
		setPhaseError(3, "Failed to load scene videos.");
	}

	// Reveal Phase 4
	showPhase(4);

	if (state.autoRun) _handleRunStage4();
}

async function _onStage4Complete() {
	setPhaseComplete(4, "Assembly complete");

	try {
		const res = await fetch(`/api/videos/${state.jobId}`);
		if (!res.ok) throw new Error("Video fetch failed");
		const data = await res.json();

		renderFinalVideo(
			data.status === "completed" && data.videos?.length
				? data.videos[0]
				: null,
		);
		revealPhaseResults(4);
		// Re-enable the run button and update version nav for stage 4
		document.getElementById("run-stage4-btn")?.removeAttribute("disabled");
		const job4 = await _fetchVersionInfo(4);
		if (job4) _updateVersionNavBtns(4, job4.ver_count, job4.cur_ver);
	} catch (err) {
		console.error("Failed to load final video:", err);
		setPhaseError(4, "Failed to load final video.");
	}

	updateSessionStatus(state.jobId, "completed");
	_refreshActiveSessions();
}

// ---------------------------------------------------------------------------
// Button listeners
// ---------------------------------------------------------------------------

function _setupButtonListeners() {
	_rewireButton("start-stage1-btn", _handleStartStage1);
	_rewireButton("run-stage2-btn", _handleRunStage2);
	_rewireButton("run-stage3-btn", _handleRunStage3);
	_rewireButton("run-stage4-btn", _handleRunStage4);
	_rewireButton("prev-stage1-btn", () => _handleNavigateStage(1, "prev"));
	_rewireButton("next-stage1-btn", () => _handleNavigateStage(1, "next"));
	_rewireButton("prev-stage2-btn", () => _handleNavigateStage(2, "prev"));
	_rewireButton("next-stage2-btn", () => _handleNavigateStage(2, "next"));
	_rewireButton("prev-stage3-btn", () => _handleNavigateStage(3, "prev"));
	_rewireButton("next-stage3-btn", () => _handleNavigateStage(3, "next"));
	_rewireButton("prev-stage4-btn", () => _handleNavigateStage(4, "prev"));
	_rewireButton("next-stage4-btn", () => _handleNavigateStage(4, "next"));

	// Delegated handler for per-scene retry buttons (rendered dynamically)
	const sceneGrid = document.getElementById("scene-videos-grid");
	if (sceneGrid) {
		sceneGrid.addEventListener("click", async (e) => {
			const btn = e.target.closest(".retry-scene-btn");
			if (!btn) return;
			const sceneNumber = parseInt(btn.dataset.sceneNumber, 10);
			if (!state.jobId || isNaN(sceneNumber)) return;
			btn.disabled = true;
			btn.textContent = "Retrying...";
			await fetch(`/api/rerun_scene/${state.jobId}/${sceneNumber}`, {
				method: "POST",
			});
		});
	}
}

function _rewireButton(id, handler) {
	const btn = document.getElementById(id);
	if (!btn) return;
	const fresh = btn.cloneNode(true);
	btn.parentNode.replaceChild(fresh, btn);
	fresh.addEventListener("click", handler);
}

async function _handleStartStage1() {
	const btn = document.getElementById("start-stage1-btn");
	if (btn) btn.disabled = true;

	const isRerun = document
		.getElementById("phase-1-badge")
		?.classList.contains("done");

	const transcriber =
		document.getElementById("pipeline-transcriber-select")?.value ||
		state.transcriber;
	const llm =
		document.getElementById("pipeline-llm-select")?.value || state.llm;
	await _updateJobProviders({ transcriber, llm });

	setSelectsLocked(
		["pipeline-transcriber-select", "pipeline-llm-select"],
		true,
	);

	if (isRerun) {
		document.getElementById("phase-1-results")?.classList.add("hidden");
		_setDownstreamRunsBlocked(1, true);
	}

	const progressPanel = document.getElementById("phase-1-progress");
	if (progressPanel) progressPanel.classList.remove("hidden");

	setPhaseRunning(1);

	const _ws = connectWebSocket(state.jobId, _handleMessage);
	if (_ws.readyState !== WebSocket.OPEN) {
		await new Promise((resolve) =>
			_ws.addEventListener("open", resolve, { once: true }),
		);
	}

	try {
		const endpoint = isRerun
			? `/api/rerun_stage1/${state.jobId}`
			: `/api/start_stage1/${state.jobId}`;
		const res = await fetch(endpoint, { method: "POST" });
		if (!res.ok) throw new Error("Failed to start Stage 1");
	} catch (err) {
		console.error("Stage 1 start error:", err);
		setPhaseError(1, "Failed to start transcription.");
		if (btn) btn.disabled = false;
		if (progressPanel) progressPanel.classList.add("hidden");
		setSelectsLocked(
			["pipeline-transcriber-select", "pipeline-llm-select"],
			false,
		);
		if (isRerun) {
			revealPhaseResults(1);
			_setDownstreamRunsBlocked(1, false);
		}
	}
}

async function _handleRunStage2() {
	const mapping = collectSpeakerMapping();
	await _handleRunStage2WithMapping(mapping);
}

async function _handleRunStage2WithMapping(mapping) {
	const btn = document.getElementById("run-stage2-btn");
	if (btn) btn.disabled = true;

	const isRerun = document
		.getElementById("phase-2-badge")
		?.classList.contains("done");

	const llm =
		document.getElementById("pipeline-stage2-llm-select")?.value ||
		state.llm;
	await _updateJobProviders({ llm });
	setSelectsLocked(["pipeline-stage2-llm-select"], true);

	if (isRerun) {
		document.getElementById("phase-2-results")?.classList.add("hidden");
		_setDownstreamRunsBlocked(2, true);
	}

	setPhaseRunning(2);
	document.getElementById("phase-2-progress")?.classList.remove("hidden");

	try {
		let res;
		if (isRerun) {
			res = await fetch(`/api/regenerate_stage2/${state.jobId}`, {
				method: "POST",
			});
		} else {
			res = await fetch(`/api/map_speakers/${state.jobId}`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(mapping),
			});
		}
		if (!res.ok) throw new Error("Failed to start Stage 2");
	} catch (err) {
		console.error("Stage 2 start error:", err);
		setPhaseError(2, "Failed to start LLM processing.");
		if (btn) btn.disabled = false;
		setSelectsLocked(["pipeline-stage2-llm-select"], false);
		if (isRerun) {
			revealPhaseResults(2);
			_setDownstreamRunsBlocked(2, false);
		}
	}
}

async function _handleRunStage3() {
	const video_gen =
		document.getElementById("pipeline-video-select")?.value ||
		state.videoGen;
	await _updateJobProviders({ video_gen });

	const btn = document.getElementById("run-stage3-btn");
	if (btn) btn.disabled = true;

	const isRerun = document
		.getElementById("phase-3-badge")
		?.classList.contains("done");

	setSelectsLocked(["pipeline-video-select"], true);

	if (isRerun) {
		document.getElementById("phase-3-results")?.classList.add("hidden");
		_setDownstreamRunsBlocked(3, true);
	}

	setPhaseRunning(3);
	document.getElementById("phase-3-progress")?.classList.remove("hidden");

	try {
		const endpoint = isRerun
			? `/api/rerun_stage3/${state.jobId}`
			: `/api/run_stage3/${state.jobId}`;
		const res = await fetch(endpoint, { method: "POST" });
		if (!res.ok) throw new Error("Failed to start Stage 3");
	} catch (err) {
		console.error("Stage 3 start error:", err);
		setPhaseError(3, "Failed to start video generation.");
		if (btn) btn.disabled = false;
		setSelectsLocked(["pipeline-video-select"], false);
		if (isRerun) {
			revealPhaseResults(3);
			_setDownstreamRunsBlocked(3, false);
		}
	}
}

async function _handleRunStage4() {
	const btn = document.getElementById("run-stage4-btn");
	if (btn) btn.disabled = true;

	const isRerun = document
		.getElementById("phase-4-badge")
		?.classList.contains("done");

	if (isRerun) {
		document.getElementById("phase-4-results")?.classList.add("hidden");
	}

	setPhaseRunning(4);
	document.getElementById("phase-4-progress")?.classList.remove("hidden");

	try {
		const endpoint = isRerun
			? `/api/rerun_stage4/${state.jobId}`
			: `/api/run_stage4/${state.jobId}`;
		const res = await fetch(endpoint, { method: "POST" });
		if (!res.ok) throw new Error("Failed to start Stage 4");
	} catch (err) {
		console.error("Stage 4 start error:", err);
		setPhaseError(4, "Failed to start assembly.");
		if (btn) btn.disabled = false;
		if (isRerun) revealPhaseResults(4);
	}
}

/** Navigate to previous or next version for the given stage. */
async function _handleNavigateStage(stageNum, direction) {
	const prevBtn = document.getElementById(`prev-stage${stageNum}-btn`);
	const nextBtn = document.getElementById(`next-stage${stageNum}-btn`);
	if (prevBtn) prevBtn.disabled = true;
	if (nextBtn) nextBtn.disabled = true;

	try {
		const res = await fetch(
			`/api/navigate_stage/${state.jobId}/${stageNum}?direction=${direction}`,
			{ method: "POST" },
		);
		if (!res.ok) throw new Error(`Navigate stage ${stageNum} failed`);
		const data = await res.json();
		_updateVersionNavBtns(stageNum, data.ver_count, data.cur_ver);

		// Re-render stage-specific results
		if (stageNum === 1) {
			const r = await fetch(`/api/transcript/${state.jobId}`);
			if (r.ok) {
				const d = await r.json();
				renderTranscript(d.transcript);
			}
		} else if (stageNum === 2) {
			const r = await fetch(`/api/stage2_results/${state.jobId}`);
			if (r.ok) {
				const d = await r.json();
				renderSpeakerMap(d.speaker_map);
				if (d.speaker_visualization)
					renderSpeakerVisualization(
						d.speaker_visualization,
						_currentSpeakerMap,
					);
				renderStoryboard(d.scenes);
			}
		} else if (stageNum === 3) {
			const r = await fetch(`/api/stage3_results/${state.jobId}`);
			if (r.ok) {
				const d = await r.json();
				renderSceneVideos(d.scenes);
				for (const scene of d.scenes) {
					if (scene.video_url)
						injectSceneVideo(scene.scene_number, scene.video_url);
				}
			}
		} else if (stageNum === 4) {
			const r = await fetch(`/api/videos/${state.jobId}`);
			if (r.ok) {
				const d = await r.json();
				renderFinalVideo(
					d.status === "completed" && d.videos?.length
						? d.videos[0]
						: null,
				);
			}
		}
	} catch (err) {
		console.error(`Navigate stage ${stageNum} error:`, err);
	} finally {
		// Re-enable buttons based on updated state (handled by _updateVersionNavBtns)
	}
}

/**
 * Show/hide arrow navigation buttons for a stage based on version count and current position.
 * Also ensures the re-run button for that stage is visible.
 */
function _updateVersionNavBtns(stageNum, verCount, curVer) {
	const prevBtn = document.getElementById(`prev-stage${stageNum}-btn`);
	const nextBtn = document.getElementById(`next-stage${stageNum}-btn`);

	const hasPrev = curVer > 0 || (curVer === -1 && verCount > 0);
	const hasNext = curVer !== -1;

	if (prevBtn) {
		prevBtn.disabled = false;
		if (hasPrev) prevBtn.classList.remove("hidden");
		else prevBtn.classList.add("hidden");
	}
	if (nextBtn) {
		nextBtn.disabled = false;
		if (hasNext) nextBtn.classList.remove("hidden");
		else nextBtn.classList.add("hidden");
	}
}

/**
 * Fetch version info for a stage from the job status endpoint.
 * Returns { ver_count, cur_ver } or null on failure.
 */
async function _fetchVersionInfo(stageNum) {
	try {
		const res = await fetch(`/api/job_status/${state.jobId}`);
		if (!res.ok) return null;
		const data = await res.json();
		return {
			ver_count: data[`stage${stageNum}_ver_count`] ?? 0,
			cur_ver: data[`stage${stageNum}_cur_ver`] ?? -1,
		};
	} catch {
		return null;
	}
}

/**
 * Disable or enable run/rerun buttons for all stages downstream of fromStage.
 * fromStage=1 blocks stages 2,3,4; fromStage=2 blocks 3,4; fromStage=3 blocks 4.
 */
function _setDownstreamRunsBlocked(fromStage, blocked) {
	const downstreamBtnIds = {
		1: ["run-stage2-btn", "run-stage3-btn", "run-stage4-btn"],
		2: ["run-stage3-btn", "run-stage4-btn"],
		3: ["run-stage4-btn"],
	};
	const ids = downstreamBtnIds[fromStage] || [];
	for (const id of ids) {
		const btn = document.getElementById(id);
		if (btn) btn.disabled = blocked;
	}
}

// ---------------------------------------------------------------------------
// Pipeline UI reset (used when switching sessions)
// ---------------------------------------------------------------------------

function _resetPipelineUI() {
	_pendingSuggestions = null;
	_currentSpeakerMap = {};
	resetPhaseTimers();

	for (let n = 1; n <= 4; n++) {
		document
			.getElementById(`phase-${n}-badge`)
			?.classList.remove("running", "done", "error");
		const status = document.getElementById(`phase-${n}-status`);
		if (status) status.textContent = "";
		const detail = document.getElementById(`phase-${n}-detail`);
		if (detail) detail.textContent = "";
		const bar = document.getElementById(`phase-${n}-bar`);
		if (bar) bar.style.width = "0%";
		const progress = document.getElementById(`phase-${n}-progress`);
		if (progress) progress.classList.add("hidden");
		const results = document.getElementById(`phase-${n}-results`);
		if (results) results.classList.add("hidden");
		const timer = document.getElementById(`phase-${n}-timer`);
		if (timer) timer.textContent = "";
	}

	// Hide phases 2-4 until the previous stage completes
	for (let n = 2; n <= 4; n++) {
		document.getElementById(`phase-${n}`)?.classList.add("hidden");
	}

	// Hide the video selector panel for phase 3 (revealed when stage 2 finishes)
	document.getElementById("phase-3-start")?.classList.add("hidden");

	// Re-enable all primary action buttons and hide version nav arrows
	document.getElementById("start-stage1-btn")?.removeAttribute("disabled");
	document.getElementById("run-stage2-btn")?.removeAttribute("disabled");
	document.getElementById("run-stage3-btn")?.removeAttribute("disabled");
	document.getElementById("run-stage4-btn")?.removeAttribute("disabled");
	for (const n of [1, 2, 3, 4]) {
		document.getElementById(`prev-stage${n}-btn`)?.classList.add("hidden");
		document.getElementById(`next-stage${n}-btn`)?.classList.add("hidden");
	}

	// Hide the total time display
	document.getElementById("pipeline-total-time")?.classList.add("hidden");

	// Clear result containers
	const ids = [
		"transcript-container",
		"mapping-controls",
		"speaker-map-display",
		"speaker-visualization-display",
		"storyboard-display",
		"scene-videos-grid",
		"final-video-container",
		"converted-audio-container",
	];
	ids.forEach((id) => {
		const el = document.getElementById(id);
		if (el) el.innerHTML = "";
	});

	// Show the start panel for phase 1
	document.getElementById("phase-1-start")?.classList.remove("hidden");

	import("./history.js")
		.then((mod) => {
			if (mod.renderActiveSessions) mod.renderActiveSessions();
		})
		.catch(() => {});
}

// ---------------------------------------------------------------------------
// Active sessions sidebar refresh
// ---------------------------------------------------------------------------

function _refreshActiveSessions() {
	import("./history.js")
		.then((mod) => {
			if (mod.renderActiveSessions) mod.renderActiveSessions();
		})
		.catch(() => {});
}

// ---------------------------------------------------------------------------
// Provider update helper
// ---------------------------------------------------------------------------

async function _updateJobProviders(overrides) {
	const form = new FormData();
	for (const [k, v] of Object.entries(overrides)) form.append(k, v);
	try {
		await fetch(`/api/job/${state.jobId}`, { method: "PATCH", body: form });
	} catch (err) {
		console.warn("Could not update job providers:", err);
	}
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

function _stageToPhaseNum(stage) {
	if (!stage) return 0;
	if (stage.includes("1/4") || stage === "stage1") return 1;
	if (stage.includes("2/4") || stage === "stage2") return 2;
	if (stage.includes("3/4") || stage === "stage3") return 3;
	if (stage.includes("4/4") || stage === "stage4") return 4;
	return 0;
}

function _stageToPercent(stage, detail) {
	if (stage.includes("1/4")) {
		if (detail?.includes("Converting")) return 15;
		if (detail?.includes("Uploading")) {
			const pct = detail?.match(/(\d+)%/);
			return pct ? Math.floor(15 + parseInt(pct[1], 10) * 0.35) : 25;
		}
		if (detail?.includes("Transcribing")) return 55;
		if (detail?.includes("suggestions")) return 88;
		return 40;
	}
	if (stage.includes("2/4")) {
		if (detail?.includes("visualizations")) return 25;
		if (detail?.includes("storyboard")) return 45;
		if (detail?.includes("relevance")) return 65;
		if (detail?.includes("production")) return 85;
		return 30;
	}
	if (stage.includes("3/4")) {
		const match = detail?.match(/\((\d+)%\)/);
		if (match) return parseInt(match[1], 10);
		return 50;
	}
	if (stage.includes("4/4")) return 70;
	return 20;
}
