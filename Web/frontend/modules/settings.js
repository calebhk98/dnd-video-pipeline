/**
 * modules/settings.js - Settings modal: open, close, load, and save API keys.
 *
 * The settings modal lets users store API keys for each pipeline stage.
 * On open, existing keys are fetched from the server and shown as masked
 * placeholders so users know which keys are already configured.
 * On save, only fields the user has actually typed into are sent to the server.
 *
 * The modal body is rendered dynamically from providers.js ,  to add a new
 * provider, edit that file only.
 */

import { showToast } from './state.js';
import { PROVIDERS } from './providers.js';

/**
 * Initialise all settings-modal event listeners.
 * Must be called once after the DOM is ready.
 */
export function initSettings() {
	const settingsBtn      = document.getElementById('settings-btn');
	const settingsModal	= document.getElementById('settings-modal');
	const closeSettingsBtn = document.getElementById('close-settings-btn');
	const saveSettingsBtn  = document.getElementById('save-settings-btn');
	const settingsForm     = document.getElementById('settings-form');

	// Build the form from the provider config
	renderProviders(settingsForm);

	// Open modal and populate with current (masked) key values
	settingsBtn.addEventListener('click', () => openSettings(settingsModal));

	// Close when the x button is clicked
	closeSettingsBtn.addEventListener('click', () => closeModal(settingsModal));

	// Close when clicking the dark overlay outside the modal card
	settingsModal.addEventListener('click', (e) => {
		if (e.target === settingsModal || e.target.classList.contains('modal-overlay')) {
			closeModal(settingsModal);
		}
	});

	// Persist any changed keys to the server
	saveSettingsBtn.addEventListener('click', () => saveSettings(settingsModal, saveSettingsBtn));
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Build and inject the API key groups into the settings form container.
 * Mirrors the static HTML structure exactly so existing CSS applies unchanged.
 */
function renderProviders(container) {
	container.innerHTML = PROVIDERS.map(({ group, providers }) => `
		<div class="api-key-group">
			<p class="api-key-group-title">${group}</p>
			${providers.map(({ id, label, envKey, keyUrl, note, type, options }) => type === 'select' ? `
			<div class="input-group">
				<label for="${id}">${label}</label>
				<select id="${id}" data-key="${envKey}">
					${options.map(o => `<option value="${o.value}">${o.label}</option>`).join('')}
				</select>
			</div>` : type === 'text' ? `
			<div class="input-group">
				<label for="${id}">
					${label}${note ? ` <span class="label-note">(${note})</span>` : ''}
					<a href="${keyUrl}" target="_blank" rel="noopener" class="key-link">Get key -></a>
				</label>
				<input type="text" id="${id}" data-key="${envKey}">
			</div>` : `
			<div class="input-group">
				<label for="${id}">
					${label}${note ? ` <span class="label-note">(${note})</span>` : ''}
					<a href="${keyUrl}" target="_blank" rel="noopener" class="key-link">Get key -></a>
				</label>
				<input type="password" id="${id}" data-key="${envKey}">
			</div>`).join('')}
		</div>`).join('');
}

/** Open the settings modal and load current key values from the server. */
async function openSettings(modal) {
	modal.classList.remove('hidden');

	try {
		const res = await fetch('/api/settings');
		if (!res.ok) throw new Error(`HTTP ${res.status}`);

		const settings = await res.json();

		// For each password input, show the masked value as a placeholder
		// so the user can see which keys are already set without editing them
		modal.querySelectorAll('input[type="password"]').forEach(input => {
			input.value = '';
			const masked = settings[input.dataset.key];
			input.placeholder = masked || 'Not configured';
		});

		// For each plain-text input, show the current value directly
		modal.querySelectorAll('input[type="text"]').forEach(input => {
			input.value = settings[input.dataset.key] || '';
			input.placeholder = 'Not configured';
		});

		// For each select input, restore the saved value
		modal.querySelectorAll('select[data-key]').forEach(sel => {
			const val = settings[sel.dataset.key];
			if (val) sel.value = val;
		});
	} catch (err) {
		console.error('Failed to load settings:', err);
		showToast('Failed to load existing settings', 'error');
	}
}

/** Hide the settings modal. */
function closeModal(modal) {
	modal.classList.add('hidden');
}

/**
 * Collect changed API key values and POST them to /api/settings.
 * Only inputs the user has typed into (non-empty, not the masked placeholder)
 * are included in the request body.
 */
async function saveSettings(modal, saveBtn) {
	const payload = {};
	let hasUpdates = false;

	modal.querySelectorAll('input[type="password"]').forEach(input => {
		const val = input.value.trim();
		// Skip empty fields and masked values echoed back from the server
		if (val && !val.startsWith('sk-***')) {
			payload[input.dataset.key] = val;
			hasUpdates = true;
		}
	});

	// Collect plain-text inputs (e.g. AWS_REGION)
	modal.querySelectorAll('input[type="text"]').forEach(input => {
		const val = input.value.trim();
		if (val) {
			payload[input.dataset.key] = val;
			hasUpdates = true;
		}
	});

	// Always persist select values (they always have a valid selection)
	modal.querySelectorAll('select[data-key]').forEach(sel => {
		payload[sel.dataset.key] = sel.value;
		hasUpdates = true;
	});

	// Nothing to save - just close the modal
	if (!hasUpdates) {
		closeModal(modal);
		return;
	}

	// Disable button while the request is in flight
	saveBtn.disabled = true;
	saveBtn.textContent = 'Saving...';

	try {
		const res = await fetch('/api/settings', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(payload),
		});

		if (res.ok) {
			showToast('Settings saved successfully', 'success');
			closeModal(modal);
			// Clear inputs so stale values don't linger
			modal.querySelectorAll('input[type="password"]').forEach(i => (i.value = ''));
			modal.querySelectorAll('input[type="text"]').forEach(i => (i.value = ''));
		} else {
			showToast('Failed to save settings', 'error');
		}
	} catch (err) {
		console.error('Failed to save settings:', err);
		showToast('Failed to save settings', 'error');
	} finally {
		saveBtn.disabled = false;
		saveBtn.textContent = 'Save API Keys';
	}
}

