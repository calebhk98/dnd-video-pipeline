"""
schemas.py ,  Pydantic data models for the video/narrative production pipeline.

The models in this module form a progressive enrichment hierarchy:

1. Raw transcription output
Transcript  +- List[Utterance]  (one per speaker turn)

2. Narrative structure derived from the transcript
Storyboard +- List[Scene]  (ordered narrative scenes)

3. Full production detail ready for video generation
ProductionScript     +- List[ProductionScene]  (Scene + filming details)

All models use Pydantic v2 (BaseModel) so they support automatic validation,
serialisation to/from JSON, and IDE-friendly type hints.
"""

from pydantic import BaseModel, ConfigDict, field_validator, Field
from typing import List, Optional


class Utterance(BaseModel):
	"""
	A single spoken segment produced by one speaker during a recording.

	Utterances are the atomic unit of a transcript ,  each represents one
	continuous block of speech from a single identified speaker before the
	speaker changes or a significant pause occurs.

	Timestamps (`start`, `end`) are expressed in **seconds** from the beginning
	of the audio file and are provided by the upstream speech-to-text service.
	"""

	# The speaker identifier assigned by the diarisation step, e.g. "Speaker 1",
	# "Speaker 2".  Labels are consistent within a single transcript but are NOT
	# guaranteed to carry meaning across different recordings.
	speaker: str

	# The recognised speech content for this utterance, as plain text.
	# Punctuation and capitalisation depend on the transcription model used.
	text: str

	# Start time of this utterance in seconds from the start of the audio file.
	# Always >= 0 and strictly less than `end`.
	start: float

	# End time of this utterance in seconds from the start of the audio file.
	# Always > `start` and <= the parent Transcript's `audio_duration`.
	end: float


class Transcript(BaseModel):
	"""
	A complete, structured transcript of a single audio recording.

	A Transcript is produced by the transcription service and aggregates all
	speaker utterances for an audio file along with metadata about the recording
	and the transcription job itself.

	The `utterances` list is ordered chronologically by `start` time.
	The `full_text` field is a convenience concatenation of all utterance texts;
	it can be used for quick keyword searches without iterating utterances.
	"""

	# Total length of the source audio file in **seconds**.
	# Used downstream to validate that utterance timestamps are in bounds and
	# to calculate scene boundaries relative to the full recording.
	audio_duration: float

	# The completion status of the transcription job, as returned by the
	# transcription service (e.g. "completed", "failed", "processing").
	# Consumers should assert status == "completed" before using the data.
	status: str

	# Ordered list of all utterances in the recording, sorted chronologically
	# by `start` time.  May be empty if no speech was detected.
	utterances: List[Utterance]

	# Flat string containing all recognised speech concatenated in order.
	# Typically used for LLM prompts or full-text search without needing to
	# iterate through individual utterances.
	full_text: str


class Scene(BaseModel):
	"""
	A single narrative scene extracted from the transcript.

	Scenes are produced by an LLM that analyses the transcript and segments it
	into coherent narrative moments (e.g. "arrival at location", "confrontation",
	"resolution").  Each scene maps back to a time range in the original audio so
	that the corresponding audio/video footage can be retrieved.

	Scenes are the intermediate representation between raw transcription and full
	production detail ,  they carry the *what* and *where* but not yet the *how*.
	"""

	# 1-indexed sequential number identifying this scene within the storyboard.
	# Scenes should be processed and displayed in ascending scene_number order.
	scene_number: int

	# Time in **seconds** from the start of the audio at which this scene begins.
	# Corresponds to the start timestamp of the earliest utterance included in
	# this scene.
	start_time: float

	# Time in **seconds** from the start of the audio at which this scene ends.
	# Corresponds to the end timestamp of the latest utterance included in this
	# scene.  Always > start_time.
	end_time: float

	# Short textual description of the physical or conceptual setting for this
	# scene, e.g. "urban street at night" or "hospital waiting room".
	# Used as context when constructing visual prompts.
	location: str

	# Human-readable summary of what happens in this scene, written in present
	# tense from a narrative perspective.  Intended for review by human editors
	# and as input context for downstream LLM steps.
	narrative_summary: str

	# Text prompt intended for an image or video generation model describing the
	# visual composition of this scene.  Should be self-contained (not rely on
	# context from other scenes) so it can be sent directly to a generation API.
	visual_prompt: str

	# Whether this scene contains in-game content (True) or is purely out-of-character
	# table talk such as bathroom breaks, player introductions, or rule discussions
	# that should be excluded from the final video (False).  None means not yet reviewed.
	is_relevant: Optional[bool] = None

	# Human-readable explanation of why the scene was judged relevant or irrelevant,
	# produced by the relevance-reviewer LLM step.
	relevance_reason: str = ""

	@field_validator('start_time', 'end_time', mode='before')
	@classmethod
	def _strip_time_suffix(cls, v):
		"""Accept timestamps with a trailing 's' (e.g. '0.56s') from LLM output."""
		if isinstance(v, str):
			return float(v.rstrip('s'))
		return v


class Storyboard(BaseModel):
	"""
	An ordered collection of scenes forming the complete narrative storyboard.

	A Storyboard is the direct output of the scene-extraction step and serves as
	the blueprint for the production pipeline.  It captures the full narrative arc
	of the recording as a sequence of discrete, visually-described scenes.

	The `scenes` list MUST be ordered by `scene_number` (ascending).  Downstream
	steps rely on this ordering to assemble the final video in the correct sequence.
	"""

	# All scenes in the storyboard, ordered by scene_number (1-indexed, ascending).
	# The list should be non-empty; a storyboard with zero scenes indicates that
	# no meaningful narrative could be extracted from the source transcript.
	scenes: List[Scene]


class SceneShot(BaseModel):
	"""
	A single short visual moment within a larger scene, suitable for a 3-10 second video clip.

	Scenes can span several minutes of gameplay.  SceneShots break a scene down into
	discrete, sequentially ordered visual actions -- each one self-contained enough to
	be rendered as a standalone video clip by Stage 3.

	Example: the "Road to Phandalin" scene (15:43-19:58) might produce shots:
		1. Wide shot of the party's wagon rolling along a dirt road through countryside.
		2. Close-up of two dead horses blocking the road ahead.
		3. Party members exchanging wary glances from atop the wagon.
		4. Party draws weapons and fans out into a defensive formation.
	"""

	# 1-indexed position of this shot within its parent scene.
	shot_number: int

	# Narrative description of what happens in this shot, written in present tense.
	description: str

	# Self-contained visual prompt for a text-to-video model -- does not rely on
	# surrounding shots for context so it can be sent directly to a generation API.
	visual_prompt: str

	# Suggested clip duration in seconds.  Guides video generation but is not enforced.
	# Typical values: 3, 5, 7, or 10 seconds.
	duration_hint: int


class ProductionScene(Scene):
	"""
	A fully detailed production scene, extending Scene with filming instructions.

	ProductionScene inherits all fields from Scene (scene_number, start_time,
	end_time, location, narrative_summary, visual_prompt) and adds three
	additional fields that specify *how* the scene should be filmed and rendered.

	These extra fields are populated by a second LLM pass that takes the
	Storyboard as input and enriches each scene with actionable production detail.
	The `final_video_prompt` supersedes `visual_prompt` for actual video generation
	,  it incorporates the stage directions and character actions to produce a richer,
	more specific generation prompt.
	"""

	# Narrative stage directions describing camera angles, lighting, mood, and
	# overall cinematographic approach for this scene, e.g.
	# "Low-angle shot, harsh shadows, flickering fluorescent light overhead."
	stage_directions: str

	# Description of what characters are doing within the scene ,  their movements,
	# expressions, and interactions ,  written to guide both human directors and
	# generative video models, e.g.
	# "Protagonist paces nervously; supporting character stands with arms crossed."
	character_actions: str

	# The definitive video generation prompt for this scene.  Combines location,
	# stage directions, character actions, and visual style into a single,
	# self-contained text prompt ready to be sent to a video generation API.
	# This field replaces `visual_prompt` for the actual generation step.
	final_video_prompt: str

	# Ordered list of short shots that break this scene into 3-10 second visual moments.
	# When populated, Stage 3 renders one video clip per shot rather than one per scene.
	# Empty list means no shot breakdown was performed; the scene is rendered as a whole.
	shots: List[SceneShot] = Field(default_factory=list)


class ProductionScript(BaseModel):
	"""
	The complete production script ,  the final output of the enrichment pipeline.

	A ProductionScript is the fully detailed, generation-ready form of a
	Storyboard.  Each Scene has been upgraded to a ProductionScene containing
	all the information needed to generate corresponding video footage.

	This model is the primary artefact handed off to the video generation service.
	Like Storyboard, the `scenes` list MUST be ordered by `scene_number` ascending
	so that the generated clips can be assembled into a coherent final video.
	"""

	# All production scenes, ordered by scene_number (1-indexed, ascending).
	# Each entry contains the full set of fields needed by the video generation
	# pipeline: location, visual prompts, stage directions, and character actions.
	scenes: List[ProductionScene]
