"""Claude Tool Schemas
=====================
Defines the Anthropic tool-calling schemas used by ``ClaudeProcessor``.  Each
constant is a list containing a single tool definition, ready to be passed
directly as the ``tools`` argument to ``client.messages.create()``.
"""

_STORYBOARD_SCENE_SCHEMA = {
	"type": "object",
	"properties": {
		"scene_number": {"type": "integer"},
		"start_time": {"type": "number"},
		"end_time": {"type": "number"},
		"location": {"type": "string"},
		"narrative_summary": {"type": "string"},
		"visual_prompt": {"type": "string"}
	},
	"required": [
		"scene_number", "start_time", "end_time",
		"location", "narrative_summary", "visual_prompt"
	]
}

STORYBOARD_TOOL = [
	{
		"name": "generate_storyboard",
		"description": "Standardized storyboard generation for D&D session video prompts.",
		"input_schema": {
			"type": "object",
			"properties": {
				"scenes": {
					"type": "array",
					"items": _STORYBOARD_SCENE_SCHEMA
				}
			},
			"required": ["scenes"]
		}
	}
]

_PRODUCTION_SCRIPT_SCENE_SCHEMA = {
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
		"final_video_prompt": {"type": "string"}
	},
	"required": [
		"scene_number", "start_time", "end_time", "location",
		"narrative_summary", "visual_prompt",
		"stage_directions", "character_actions", "final_video_prompt"
	]
}

PRODUCTION_SCRIPT_TOOL = [
	{
		"name": "generate_production_script",
		"description": "Granular production script generation for AI video production.",
		"input_schema": {
			"type": "object",
			"properties": {
				"scenes": {
					"type": "array",
					"items": _PRODUCTION_SCRIPT_SCENE_SCHEMA
				}
			},
			"required": ["scenes"]
		}
	}
]

_SCENE_RELEVANCE_ITEM_SCHEMA = {
	"type": "object",
	"properties": {
		"scene_number": {"type": "integer"},
		"relevance_reason": {
			"type": "string",
			"description": "First, explain in one sentence why this scene is or is not in-game content."
		},
		"is_relevant": {
			"type": "boolean",
			"description": "True if the scene contains in-game narrative events; False if it is purely out-of-character table talk."
		}
	},
	"required": ["scene_number", "relevance_reason", "is_relevant"]
}

SCENE_RELEVANCE_TOOL = [
	{
		"name": "review_scene_relevance",
		"description": "Review each storyboard scene and decide whether it contains in-game narrative content.",
		"input_schema": {
			"type": "object",
			"properties": {
				"scenes": {
					"type": "array",
					"items": _SCENE_RELEVANCE_ITEM_SCHEMA
				}
			},
			"required": ["scenes"]
		}
	}
]

_SHOT_SCHEMA = {
	"type": "object",
	"properties": {
		"shot_number": {
			"type": "integer",
			"description": "1-indexed position within this scene."
		},
		"description": {
			"type": "string",
			"description": "Present-tense narrative description of what happens in this shot."
		},
		"visual_prompt": {
			"type": "string",
			"description": "Self-contained text-to-video prompt for this shot. Include setting, characters, action, and mood."
		},
		"duration_hint": {
			"type": "integer",
			"description": "Suggested clip length in seconds (3, 5, 7, or 10)."
		}
	},
	"required": ["shot_number", "description", "visual_prompt", "duration_hint"]
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
			"description": "Ordered list of shots covering this scene completely. Use as many shots as the scene requires -- a short travel scene may need 3-4, a long combat encounter may need 15 or more. Every distinct action beat should be its own shot.",
			"items": _SHOT_SCHEMA
		}
	},
	"required": [
		"scene_number", "start_time", "end_time", "location",
		"narrative_summary", "visual_prompt",
		"stage_directions", "character_actions", "final_video_prompt",
		"shots"
	]
}

SCENE_SHOTS_TOOL = [
	{
		"name": "generate_scene_shots",
		"description": "Expand storyboard scenes into production details and break each into short video shots.",
		"input_schema": {
			"type": "object",
			"properties": {
				"scenes": {
					"type": "array",
					"items": _SCENE_SHOTS_ITEM_SCHEMA
				}
			},
			"required": ["scenes"]
		}
	}
]
