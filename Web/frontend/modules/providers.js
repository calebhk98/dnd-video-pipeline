/**
 * modules/providers.js - Single source of truth for all AI provider API keys.
 *
 * Provider data lives in providers.json so the Python backend can also read it
 * (avoiding duplication of the envKey list in settings.py).
 *
 * To add a new provider:
 *   1. Pick the right group (or add a new one) in providers.json.
 *   2. Append an entry to that group's `providers` array.
 *   3. That's it ,  the settings modal renders from this config automatically.
 *
 * Each provider object:
 *   id          - HTML element id for the <input>
 *   label       - Human-readable service name shown in the label
 *   envKey      - Backend environment variable name (used as data-key)
 *   keyUrl      - Direct link to the provider's API key page
 *   note		- (optional) Extra clarifying text shown in the label
 */
import PROVIDERS_DATA from './providers.json' with { type: 'json' };
export const PROVIDERS = PROVIDERS_DATA;
