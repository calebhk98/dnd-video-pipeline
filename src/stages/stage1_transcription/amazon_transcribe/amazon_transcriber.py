"""
Stage 1 Transcription ,  Amazon Transcribe Backend
==================================================
Uses the Amazon Web Services (AWS) Transcribe managed service to convert audio
to text with automatic speaker diarization.

Workflow summary:
	1. Upload the local audio file to a user-supplied S3 bucket.
	2. Start an Amazon Transcribe job with ``ShowSpeakerLabels=True``.
	3. Poll the job status every ``poll_interval`` seconds until it reaches
	COMPLETED or FAILED (up to ``timeout`` seconds).
	4. Download the resulting JSON from the presigned ``TranscriptFileUri``.
	5. Parse the JSON's ``speaker_labels`` to reconstruct per-speaker utterances.
	6. Clean up the temporary S3 object to avoid storage costs.

Required AWS resources:
	- An S3 bucket the caller has read/write access to.
	- IAM credentials with ``transcribe:StartTranscriptionJob``,
	``transcribe:GetTranscriptionJob``, ``s3:PutObject``, ``s3:DeleteObject``.

Dependencies:
	pip install boto3
"""

import os
import time
import json
import logging
import uuid
import urllib.request
from typing import Dict, Any
import boto3
from src.stages.stage1_transcription.base import BaseTranscriber
from src.shared.schemas import Transcript, Utterance

logger = logging.getLogger(__name__)


class AmazonTranscriber(BaseTranscriber):
	"""
	Concrete implementation of BaseTranscriber using Amazon Transcribe.

	Uploads the audio file to S3, starts a transcription job with speaker
	diarization, polls for completion, downloads the JSON result, and maps it
	to the shared ``Transcript`` schema.

	Config keys (all optional if the corresponding env var is set):
		aws_access_key_id     ,  AWS access key (falls back to AWS_ACCESS_KEY_ID).
		aws_secret_access_key ,  AWS secret key (falls back to AWS_SECRET_ACCESS_KEY).
		aws_region            ,  AWS region, e.g. 'us-east-1' (falls back to
			AWS_DEFAULT_REGION; default: 'us-east-1').
		s3_bucket             ,  Name of the S3 bucket for temporary audio uploads
			(falls back to AWS_S3_BUCKET; **required**).
		max_speakers          ,  Upper bound on the number of distinct speakers
			(default: 6).  Amazon Transcribe uses this to tune
			its diarization model.
	"""

	def __init__(self, config: Dict[str, Any]):
		"""
		Initialize the AWS clients for Transcribe and S3.

		Credential resolution order:
			1. Explicit values in ``config``.
			2. Standard AWS environment variables (AWS_ACCESS_KEY_ID, etc.).
			3. AWS default credential chain (IAM role, ~/.aws/credentials, etc.)
			when no explicit credentials are provided ,  boto3 handles this
			automatically if neither ``config`` nor env vars supply them.

		Args:
			config: Configuration dictionary.  See class docstring for keys.

		Raises:
			ValueError: If ``s3_bucket`` is absent from both ``config`` and env.
		"""
		# Resolve credentials from config first, then fall back to standard env vars.
		self.aws_access_key_id = config.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
		self.aws_secret_access_key = config.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
		self.region = config.get("aws_region") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")

		# S3 bucket is required ,  Transcribe writes results there and we stage audio.
		self.s3_bucket = config.get("s3_bucket") or os.getenv("AWS_S3_BUCKET")
		self.max_speakers = config.get("max_speakers", 6)

		if not self.s3_bucket:
			raise ValueError(
				"S3 bucket is required. Provide 's3_bucket' in config or set AWS_S3_BUCKET env var."
			)

		# Build kwargs dict for boto3 ,  only pass explicit credentials when provided
		# so that boto3's default credential chain still works when they are absent.
		boto_kwargs = {"region_name": self.region}
		if self.aws_access_key_id and self.aws_secret_access_key:
			boto_kwargs["aws_access_key_id"] = self.aws_access_key_id
			boto_kwargs["aws_secret_access_key"] = self.aws_secret_access_key

		# Create separate clients for the Transcribe service and S3 storage.
		self.transcribe_client = boto3.client("transcribe", **boto_kwargs)
		self.s3_client = boto3.client("s3", **boto_kwargs)

	def transcribe(self, audio_filepath: str) -> Transcript:
		"""
		Upload audio to S3, run an Amazon Transcribe job, and return a Transcript.

		Each call generates a unique S3 key and job name via UUID to prevent
		collisions between concurrent or repeated runs.  The S3 object is always
		deleted in the ``finally`` block to avoid lingering storage costs.

		Args:
			audio_filepath: Path to the local audio file (mp3, wav, flac, etc.).

		Returns:
			A populated ``Transcript`` with speaker-diarized utterances.

		Raises:
			FileNotFoundError: If ``audio_filepath`` does not exist on disk.
			RuntimeError: If the Transcribe job reaches FAILED status.
			TimeoutError: If the job does not complete within 3600 seconds.
		"""
		if not os.path.exists(audio_filepath):
			raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

		# Derive the media format from the file extension (Amazon Transcribe needs this).
		file_ext = os.path.splitext(audio_filepath)[1].lstrip(".").upper() or "MP3"

		# Use UUIDs to guarantee unique S3 key and job name across parallel runs.
		s3_key = f"transcribe-input/{uuid.uuid4()}.{file_ext.lower()}"
		job_name = f"dnd-pipeline-{uuid.uuid4()}"

		try:
			# Step 1: Upload the audio file to S3 so Amazon Transcribe can access it.
			self.s3_client.upload_file(audio_filepath, self.s3_bucket, s3_key)
			s3_uri = f"s3://{self.s3_bucket}/{s3_key}"

			# Step 2: Start the transcription job with speaker diarization enabled.
			# ShowSpeakerLabels=True activates diarization.
			# MaxSpeakerLabels caps the number of speakers the model considers , 
			# setting it too high can reduce accuracy; too low may merge speakers.
			self.transcribe_client.start_transcription_job(
				TranscriptionJobName=job_name,
				Media={"MediaFileUri": s3_uri},
				MediaFormat=file_ext.lower(),
				LanguageCode="en-US",
				Settings={
					"ShowSpeakerLabels": True,
					"MaxSpeakerLabels": self.max_speakers,
				},
			)

			# Step 3: Block until the job finishes (poll every 10 s, up to 1 hour).
			result = self._poll_job(job_name)

			# Step 4: Download the JSON result from the presigned URL provided by AWS.
			# Amazon Transcribe stores the output in a temporary S3 location and
			# gives us a short-lived HTTPS URL.
			transcript_uri = result["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
			with urllib.request.urlopen(transcript_uri) as response:
				transcript_data = json.loads(response.read().decode("utf-8"))

			return self._map_response_to_transcript(transcript_data)

		except Exception as e:
			logger.error(f"Error during Amazon Transcribe job: {str(e)}")
			raise
		finally:
			# Always clean up the uploaded audio file from S3 to avoid storage costs,
			# even when the transcription fails.
			try:
				self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
			except Exception:
				# Deletion failure is non-fatal; log silently and continue.
				pass

	def _poll_job(self, job_name: str, poll_interval: int = 10, timeout: int = 3600) -> dict:
		"""
		Repeatedly check the transcription job status until it completes or fails.

		Amazon Transcribe is an async service ,  starting a job returns immediately
		and we must poll ``GetTranscriptionJob`` to learn when it finishes.

		Args:
			job_name:      The unique job name passed to ``start_transcription_job``.
			poll_interval: Seconds to wait between status checks (default: 10).
			timeout:       Maximum total seconds to wait before giving up (default: 3600).

		Returns:
			The full ``get_transcription_job`` response dict when status is COMPLETED.

		Raises:
			RuntimeError: If the job reaches FAILED status.
			TimeoutError: If ``timeout`` seconds elapse without completion.
		"""
		elapsed = 0
		while elapsed < timeout:
			response = self.transcribe_client.get_transcription_job(
				TranscriptionJobName=job_name
			)
			status = response["TranscriptionJob"]["TranscriptionJobStatus"]

			if status == "COMPLETED":
				return response
			if status == "FAILED":
				# FailureReason is a human-readable string from AWS explaining the error.
				reason = response["TranscriptionJob"].get("FailureReason", "Unknown")
				raise RuntimeError(f"Amazon Transcribe job failed: {reason}")

			time.sleep(poll_interval)
			elapsed += poll_interval

		raise TimeoutError(f"Amazon Transcribe job '{job_name}' timed out after {timeout}s.")

	def _map_response_to_transcript(self, data: dict) -> Transcript:
		"""
		Parse the Amazon Transcribe JSON response and build the shared Transcript schema.

		Amazon Transcribe's output structure:
			results.items         ,  individual word/punctuation tokens with timestamps.
			results.speaker_labels.segments ,  speaker diarization: each segment maps
				a start_time to a speaker_label (e.g. "spk_0").

		Strategy:
			1. Build a lookup table: ``{start_time_str: speaker_label}`` from segments.
			2. Walk ``items`` in order, accumulating words per speaker.
			3. When the speaker changes, flush the accumulated words as a new Utterance.
			4. Attach punctuation to the preceding word (no separate Utterance needed).

		Args:
			data: Parsed JSON dict from the Transcribe output file.

		Returns:
			A ``Transcript`` with speaker-diarized utterances in chronological order.
		"""
		results = data.get("results", {})
		items = results.get("items", [])
		speaker_labels = results.get("speaker_labels", {})
		segments = speaker_labels.get("segments", [])

		# Build a fast lookup: start_time string -> speaker_label.
		# Amazon's diarization attaches speaker info to segments, not individual items,
		# so we index by the segment's item start times.
		time_to_speaker: dict = {}
		for segment in segments:
			for item in segment.get("items", []):
				time_to_speaker[item.get("start_time", "")] = segment.get("speaker_label", "spk_0")

		# Accumulate words into per-speaker runs, flushing on speaker change.
		utterances = []
		full_text_parts = []
		current_speaker = None
		current_words = []
		current_start = None
		current_end = None

		for item in items:
			# Punctuation items (commas, periods) have no start_time; append to last word.
			if item.get("type") == "punctuation":
				if current_words:
					current_words[-1] += item["alternatives"][0]["content"]
				continue

			start_time = float(item.get("start_time", 0.0))
			end_time = float(item.get("end_time", 0.0))
			word = item["alternatives"][0]["content"]

			# Resolve the speaker for this word using the start_time lookup.
			speaker = time_to_speaker.get(item.get("start_time", ""), "spk_0")
			full_text_parts.append(word)

			if speaker != current_speaker:
				# Flush the current run as a completed Utterance before starting a new one.
				if current_words and current_speaker is not None:
					utterances.append(Utterance(
						# Convert "spk_0" -> "Speaker 0", "spk_1" -> "Speaker 1", etc.
						speaker=f"Speaker {current_speaker.replace('spk_', '')}",
						text=" ".join(current_words),
						start=current_start,
						end=current_end,
					))
				# Begin a new speaker run.
				current_speaker = speaker
				current_words = [word]
				current_start = start_time
				current_end = end_time
			else:
				# Same speaker ,  extend the current run.
				current_words.append(word)
				current_end = end_time

		# Flush the last accumulated run after the loop ends.
		if current_words and current_speaker is not None:
			utterances.append(Utterance(
				speaker=f"Speaker {current_speaker.replace('spk_', '')}",
				text=" ".join(current_words),
				start=current_start,
				end=current_end,
			))

		# Use the end time of the last utterance as the total audio duration.
		audio_duration = utterances[-1].end if utterances else 0.0

		return Transcript(
			audio_duration=audio_duration,
			status="completed",
			utterances=utterances,
			full_text=" ".join(full_text_parts),
		)
