"""
Stage 2 LLM - Prompt Builders
==============================
Centralised prompt engineering functions shared by every LLM backend.
Editing a prompt here propagates to all backends simultaneously.
"""

import json


def build_speaker_mapping_prompt(
	transcript_text: str,
	character_sheet_context: str = "",
	distinct_speakers: list = None,
) -> str:
	"""
    Build the canonical speaker-mapping prompt shared by all LLM backends.

    Args:
        transcript_text: The transcript content to include in the prompt.
        character_sheet_context: Optional free-text description of campaign
            characters to help the LLM make better attributions.
        distinct_speakers: Optional list of speaker labels to map explicitly.

    Returns:
        A formatted prompt string ready to be sent to any LLM.
    """
	speakers_section = (
		f"\nDistinct Speakers to map: {distinct_speakers}\n"
		if distinct_speakers is not None
		else ""
	)
	return (
		"Given the following transcript from a D&D session, identify each speaker's "
		"real name (the player), their in-character name, and their character class.\n\n"
		f"Transcript:\n{transcript_text}\n"
		f"{speakers_section}"
		f"\nCharacter Sheet Context (if any):\n{character_sheet_context}\n\n"
		"Output only a JSON dictionary mapping the generic speaker labels to a string "
		'in the format "Real Name - Character Name - Class".\n'
		'Example: {"Speaker A": "Travis Willingham - Magnus - Human Fighter", '
		'"Speaker B": "Justin McElroy - Taako - Elven Wizard"}\n'
		"Return only valid JSON, no markdown."
	)


def build_speaker_visualization_prompt(speaker_map: dict) -> str:
	"""
    Build the canonical speaker-visualization prompt shared by all LLM backends.

    Args:
        speaker_map: Dict mapping speaker labels to
                     "Real Name - Character Name - Class" strings.

    Returns:
        A formatted prompt string ready to be sent to any LLM.
    """
	entries = json.dumps(speaker_map, indent=2)
	return (
		"You are helping visualize characters from a D&D session.\n\n"
		"For each speaker in the mapping below, write a SHORT visual description "
		"(1-2 sentences) of what they look like in the game world.\n\n"
		"Rules:\n"
		"- If the entry contains 'DM', 'Dungeon Master', or 'Game Master', output exactly: "
		"\"Is the Dungeon Master / narrator of the story.\"\n"
		"- If the entry contains 'guest' (case-insensitive) or has no character class, output: "
		"\"Is a guest player with no permanent character.\"\n"
		"- Otherwise, infer a vivid physical appearance from the character name and class "
		"(race, build, distinctive features, equipment). Be creative but consistent with D&D lore.\n\n"
		f"Speaker map:\n{entries}\n\n"
		"Output only a JSON dictionary mapping each speaker label to its description string.\n"
		"Example: {\"Speaker A\": \"A stocky dwarven cleric with braided auburn beard and holy symbol "
		"etched in silver.\", \"Speaker B\": \"Is the Dungeon Master / narrator of the story.\"}\n"
		"Return only valid JSON, no markdown."
	)


def build_storyboard_prompt(
	transcript_text: str,
	speaker_map: dict,
	fn_keyword: str = "tool",
) -> str:
	"""
    Build the canonical storyboard generation prompt shared by Claude, Gemini, and Deepseek.

    Args:
        transcript_text: Full transcript text to include in the prompt.
        speaker_map:     Dict mapping generic speaker labels to character name strings.
        fn_keyword:      The word used to refer to the API construct: ``"tool"`` for
                         Claude/Deepseek or ``"function"`` for Gemini.

    Returns:
        A formatted prompt string ready to be sent to the LLM.
    """
	return (
		"Analyze the following D&D transcript and generate a storyboard for a cinematic video.\n"
		"1. Filter out Out-of-Character (OOC) table talk.\n"
		"2. Identify key narrative scenes.\n"
		"3. For each scene, provide start/end times, a location, a summary, and a detailed cinematic visual prompt.\n\n"
		f"Speaker Map: {json.dumps(speaker_map)}\n"
		f"Transcript: {transcript_text}\n\n"
		f"Use the 'generate_storyboard' {fn_keyword} to provide the structured output."
	)


def build_production_script_prompt(
	storyboard_json: str,
	transcript_text: str,
	fn_keyword: str = "tool",
) -> str:
	"""
    Build the canonical production script prompt shared by Claude, Gemini, and Deepseek.

    Args:
        storyboard_json: JSON string of the storyboard (from ``model_dump_json()``).
        transcript_text: Original transcript text for additional LLM context.
        fn_keyword:      ``"tool"`` for Claude/Deepseek, ``"function"`` for Gemini.

    Returns:
        A formatted prompt string ready to be sent to the LLM.
    """
	return (
		"Given the following storyboard and original transcript, expand each scene into a detailed production script.\n"
		"Provide granular stage directions, specific character actions, and a refined final video prompt for each scene.\n\n"
		f"Storyboard: {storyboard_json}\n"
		f"Original Transcript Context: {transcript_text}\n\n"
		f"Use the 'generate_production_script' {fn_keyword} to provide the structured output."
	)


def build_scene_relevance_prompt(scenes_summary: str, fn_keyword: str = "tool") -> str:
	"""
    Build the canonical scene relevance review prompt shared by all LLM backends.

    Args:
        scenes_summary: JSON string of scenes with scene_number, location, and
                        narrative_summary fields.
        fn_keyword:     ``"tool"`` for OpenAI/Deepseek, ``"function"`` for Gemini.
                        Not used for local models (omit the tool-call instruction).

    Returns:
        A formatted prompt string ready to be sent to any LLM.
    """
	tool_instruction = (
		f"\nUse the 'review_scene_relevance' {fn_keyword} to return your assessment for every scene."
		if fn_keyword
		else "\nReturn a JSON object with a key \"scenes\" containing an array of objects, each with scene_number (integer), relevance_reason (string), and is_relevant (boolean)."
	)
	return (
		"You are reviewing scenes from a D&D session recording to decide which contain\n"
		"actual in-game narrative content versus out-of-character (OOC) table talk.\n\n"
		"Mark a scene as NOT relevant (is_relevant: false) ONLY when it is ENTIRELY made\n"
		"up of OOC content such as:\n"
		"  - Players taking a lunch or bathroom break\n"
		"  - Pre-game or post-game player introductions (players introducing themselves,\n"
		"    not their characters)\n"
		"  - Extended rules debates or technical setup with zero in-game events\n"
		"  - Session housekeeping (scheduling, logistics)\n\n"
		"Mark a scene as relevant (is_relevant: true) if ANY in-game narrative events\n"
		"occur within it, even if it starts or ends with OOC chatter.\n\n"
		f"Scenes to review:\n{scenes_summary}"
		+ tool_instruction
	)


def build_scene_shots_prompt(
	storyboard_json: str,
	scene_transcripts_json: str,
	fn_keyword: str = "tool",
) -> str:
	"""
    Build the canonical scene shots generation prompt shared by all LLM backends.

    Args:
        storyboard_json: JSON string of the storyboard (from ``model_dump_json()``).
        scene_transcripts_json: JSON string mapping scene_number -> dialogue text.
        fn_keyword: ``"tool"`` for OpenAI/Deepseek, ``"function"`` for Gemini.
            Not used for local models.

    Returns:
        A formatted prompt string ready to be sent to any LLM.
    """
	tool_instruction = (
		f"\nUse the 'generate_scene_shots' {fn_keyword} to return the complete structured output."
		if fn_keyword
		else "\nReturn only the JSON object."
	)
	return (
		"You are a cinematic director expanding a D&D storyboard into a detailed production\n"
		"script with shot-by-shot breakdowns for AI video generation.\n\n"
		"For EACH scene:\n"
		"1. Write concise stage_directions (camera angles, lighting, mood).\n"
		"2. Write character_actions (what each character physically does and expresses).\n"
		"3. Write a final_video_prompt combining location, mood, and action for the whole scene.\n"
		"4. Break the scene into as many SHOTS as needed to cover it completely.\n"
		"   Each shot is a discrete visual moment lasting ~5-10 seconds that can be rendered\n"
		"   as a standalone video clip. Use the transcript for that scene to identify every\n"
		"   distinct action, spell cast, movement, or dialogue beat. Shots must:\n"
		"   - Flow sequentially to tell the scene's story from start to finish.\n"
		"   - Have a self-contained visual_prompt (no 'continued from previous' references).\n"
		"   - Cover distinct actions or beats (don't repeat the same image with different words).\n\n"
		f"Storyboard: {storyboard_json}\n\n"
		f"Scene transcripts (dialogue for each scene's time range):\n{scene_transcripts_json}"
		+ tool_instruction
	)
