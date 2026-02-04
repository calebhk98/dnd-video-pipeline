/**
 * modules/websocket.js - Real-time pipeline progress via WebSocket.
 *
 * Maintains one WebSocket connection per job so multiple pipelines can run
 * concurrently.  Each connection stays open even when the user switches to a
 * different session in the UI; it is only explicitly closed when the pipeline
 * finishes or an error occurs.
 */

/** Open WebSocket connections keyed by jobId. */
const _connections = {};

/**
 * Open (or reuse) a WebSocket for the given job and stream progress updates.
 *
 * If a live connection for `jobId` already exists the old message handler is
 * replaced with the new one so the currently-visible session receives updates.
 *
 * @param {string}   jobId          - The job UUID to connect to.
 * @param {function} messageHandler - Called with each parsed JSON message.
 * @returns {WebSocket} The open WebSocket instance.
 */
export function connectWebSocket(jobId, messageHandler) {
	// Reuse an already-open connection; just swap the handler.
	if (_connections[jobId] && _connections[jobId].readyState === WebSocket.OPEN) {
		_connections[jobId]._messageHandler = messageHandler;
		return _connections[jobId];
	}

	const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
	const wsUrl = `${protocol}//${window.location.host}/api/ws/progress/${jobId}`;
	const ws = new WebSocket(wsUrl);

	// Store the handler on the socket itself so it can be swapped later.
	ws._messageHandler = messageHandler;

	ws.onopen = () => {
		console.log(`WebSocket connected for job ${jobId}`);
	};

	ws.onmessage = (event) => {
		try {
			const data = JSON.parse(event.data);
			console.log(`WebSocket msg [${jobId}]:`, data);
			if (ws._messageHandler) ws._messageHandler(data);
		} catch (err) {
			console.error('Error parsing WebSocket message:', err);
		}
	};

	ws.onerror = (err) => {
		console.error(`WebSocket error [${jobId}]:`, err);
	};

	ws.onclose = () => {
		console.log(`WebSocket disconnected for job ${jobId}`);
		delete _connections[jobId];
	};

	_connections[jobId] = ws;
	return ws;
}

/**
 * Attach a new message handler to an existing connection without reopening it.
 * Used when the user switches back to a session that is already connected.
 *
 * @param {string}   jobId          - The job UUID.
 * @param {function} messageHandler - New handler to attach.
 * @returns {boolean} True if a live connection was found and updated.
 */
export function reattachHandler(jobId, messageHandler) {
	const ws = _connections[jobId];
	if (ws && ws.readyState === WebSocket.OPEN) {
		ws._messageHandler = messageHandler;
		return true;
	}
	return false;
}

/**
 * Close and remove the WebSocket for a finished job.
 *
 * @param {string} jobId - The job UUID whose connection should be closed.
 */
export function disconnectWebSocket(jobId) {
	const ws = _connections[jobId];
	if (ws) {
		ws.close();
		delete _connections[jobId];
	}
}
