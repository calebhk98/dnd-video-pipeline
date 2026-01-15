"""
OpenAI Tool Schema Definitions
================================
OpenAI-format tool/function schemas used by OpenAIGPTProcessor for structured output.
Centralising them here keeps method bodies focused on logic rather than schema plumbing.
"""

_RELEVANCE_ITEM_SCHEMA = {
	"type": "object",
	"properties": {
		"scene_number": {"type": "integer"},
		"relevance_reason": {
			"type": "string",
			"description": "One sentence explaining why this scene is or is not in-game content."
		},
		"is_relevant": {
			"type": "boolean",
			"description": "True if the scene contains in-game narrative events; False if purely OOC table talk."
		},
	},
	"required": ["scene_number", "relevance_reason", "is_relevant"],
}

RELEVANCE_TOOL = {
	"type": "function",
	"function": {
		"name": "review_scene_relevance",
		"description": "Review each storyboard scene and decide whether it contains in-game narrative content.",
		"parameters": {
			"type": "object",
			"properties": {
				"scenes": {
					"type": "array",
					"items": _RELEVANCE_ITEM_SCHEMA,
				}
			},
			"required": ["scenes"],
		},
	},
}

_SHOT_SCHEMA = {
	"type": "object",
	"properties": {
		"shot_number": {"type": "integer"},
		"description": {"type": "string"},
		"visual_prompt": {"type": "string"},
		"duration_hint": {"type": "integer"},
	},
	"required": ["shot_number", "description", "visual_prompt", "duration_hint"],
}

_SCENE_SHOTS_ITEM_SCHEMA = {
	"type": "object",
	"properties": {
		"scene_number": {"type": "integer"},
		"start_time": {"type": "number"},
		"end_time": {"type": "number"},
		"location": {"type": "string"},
		"narrative_summary": {"type": "string"},
		"visual_prompt": {"type": "string"},
		"stage_directions": {"type": "string"},
		"character_actions": {"type": "string"},
		"final_video_prompt": {"type": "string"},
		"shots": {
			"type": "array",
			"description": "Ordered list of shots covering this scene completely.",
			"items": _SHOT_SCHEMA,
		},
	},
	"required": [
		"scene_number", "start_time", "end_time", "location",
		"narrative_summary", "visual_prompt",
		"stage_directions", "character_actions", "final_video_prompt",
		"shots",
	],
}

SCENE_SHOTS_TOOL = {
	"type": "function",
	"function": {
		"name": "generate_scene_shots",
		"description": "Expand storyboard scenes into production details and break each into short video shots.",
		"parameters": {
			"type": "object",
			"properties": {
				"scenes": {
					"type": "array",
					"items": _SCENE_SHOTS_ITEM_SCHEMA,
				}
			},
			"required": ["scenes"],
		},
	},
}
