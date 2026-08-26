"""Vision understanding abstraction & local VLM backend (Phase 6).

This module defines:
- ``BaseVisionAnalyzer``: abstract interface for vision model backends.
- ``LocalVisionAnalyzer``: local vision-language model backend (baseline: HuggingFaceTB/SmolVLM-256M-Instruct).

Features:
- 100% local, free inference (₹0 cost).
- Configurable via environment variables (VISION_MODEL, VISION_DEVICE).
- Conservative visual observation prompt ("Describe only what is visibly supported...").
- Structured outcome: VisionResult (subjects, objects, actions, scenes, observations).
"""

from __future__ import annotations

import abc
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from processor.models import (
    FrameObservation,
    VisionFailureCategory,
    VisionResult,
)

logger = logging.getLogger("analyzer.processor.vision")

DEFAULT_VISION_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def _vision_config() -> Tuple[str, str]:
    """Read vision model settings from environment."""
    model_name = os.environ.get("VISION_MODEL", DEFAULT_VISION_MODEL).strip()
    device = os.environ.get("VISION_DEVICE", "auto").strip().lower()
    return model_name, device


class BaseVisionAnalyzer(abc.ABC):
    """Abstract vision analyzer interface."""

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Return human-readable identifier of the underlying model."""
        pass

    @abc.abstractmethod
    def analyze_frames(
        self,
        frame_paths: List[Path],
        media_path: Optional[Path] = None,
        input_type: str = "video",
        frame_extraction_seconds: Optional[float] = None,
    ) -> VisionResult:
        """Analyze a sequence of image/frame paths and return a VisionResult."""
        pass


class LocalVisionAnalyzer(BaseVisionAnalyzer):
    """Local Vision-Language Model backend for visual understanding (Phase 6).

    Uses transformers AutoModelForImageTextToText and AutoProcessor for local model inference.
    Baseline model: HuggingFaceTB/SmolVLM-256M-Instruct
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        env_model, env_device = _vision_config()
        self._model_name = model_name or env_model
        self._device = device or env_device
        self._model = None
        self._processor = None
        self._model_load_seconds: Optional[float] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_model(self) -> None:
        """Lazily load the vision model on first use."""
        if self._model is not None:
            return

        t0 = time.perf_counter()
        logger.info(
            "Loading local vision model: name=%s device=%s",
            self._model_name,
            self._device,
        )

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            target_device = self._device
            if target_device == "auto":
                target_device = "cuda" if torch.cuda.is_available() else "cpu"

            self._processor = AutoProcessor.from_pretrained(self._model_name, trust_remote_code=True)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self._model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if target_device == "cuda" else torch.float32,
            ).to(target_device)

            self._model_load_seconds = time.perf_counter() - t0
            logger.info("Vision model loaded in %.2fs", self._model_load_seconds)

        except Exception as exc:
            logger.error("Failed to load vision model %s: %s", self._model_name, exc)
            raise RuntimeError(f"Vision model load failed ({self._model_name}): {exc}") from exc

    def _prompt_frame(self, image: Image.Image, prompt: str) -> str:
        """Perform conservative visual question answering / captioning on an image."""
        self._ensure_model()

        if self._processor is not None:
            import torch

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                text_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self._processor(text=text_prompt, images=[image], return_tensors="pt").to(self._model.device)
            except Exception:
                inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                generate_ids = self._model.generate(**inputs, max_new_tokens=128)

            # Strip input tokens from output
            if "input_ids" in inputs:
                generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

            output = self._processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
            return output.strip()
        else:
            return "Unable to process frame."

    def analyze_frames(
        self,
        frame_paths: List[Path],
        media_path: Optional[Path] = None,
        input_type: str = "video",
        frame_extraction_seconds: Optional[float] = None,
    ) -> VisionResult:
        """Analyze a list of image/frame paths and return a populated VisionResult."""
        total_t0 = time.perf_counter()

        if not frame_paths:
            return VisionResult(
                success=False,
                media_path=str(media_path) if media_path else None,
                input_type=input_type,
                failure_category=VisionFailureCategory.FRAME_EXTRACTION_FAILED.value,
                failure_message="No frames provided for analysis.",
            )

        # Step 1: Ensure model is loaded
        try:
            self._ensure_model()
        except Exception as exc:
            return VisionResult(
                success=False,
                media_path=str(media_path) if media_path else None,
                input_type=input_type,
                failure_category=VisionFailureCategory.MODEL_LOAD_FAILED.value,
                failure_message=str(exc),
                total_processing_seconds=time.perf_counter() - total_t0,
            )

        # Step 2: Analyze each frame
        inf_t0 = time.perf_counter()
        frame_observations: List[FrameObservation] = []
        all_observations: List[str] = []
        all_subjects: set[str] = set()
        all_objects: set[str] = set()
        all_actions: set[str] = set()
        all_scenes: set[str] = set()
        all_demonstrations: set[str] = set()

        conservative_prompt = (
            "Describe only what is visibly supported by this image. "
            "Identify the main subject, visible objects, action, and scene context. "
            "If something is uncertain, say uncertain."
        )

        try:
            for idx, fpath in enumerate(frame_paths):
                if not Path(fpath).exists():
                    continue

                with Image.open(fpath) as img:
                    img_rgb = img.convert("RGB")
                    desc = self._prompt_frame(img_rgb, conservative_prompt)

                timestamp = float(idx * 2.0)  # nominal frame spacing indicator
                all_observations.append(desc)

                # Simple rule-based extraction from description
                subjects, objects, actions, scenes, demos = _extract_visual_tags(desc)
                all_subjects.update(subjects)
                all_objects.update(objects)
                all_actions.update(actions)
                all_scenes.update(scenes)
                all_demonstrations.update(demos)

                frame_observations.append(
                    FrameObservation(
                        frame_index=idx,
                        timestamp_seconds=timestamp,
                        description=desc,
                        subjects=subjects,
                        objects=objects,
                        actions=actions,
                        scenes=scenes,
                    )
                )

            inference_time = time.perf_counter() - inf_t0
            total_time = time.perf_counter() - total_t0

            return VisionResult(
                success=True,
                media_path=str(media_path) if media_path else None,
                input_type=input_type,
                frames_analyzed=len(frame_observations),
                subjects=sorted(list(all_subjects)),
                objects=sorted(list(all_objects)),
                actions=sorted(list(all_actions)),
                scenes=sorted(list(all_scenes)),
                demonstrations=sorted(list(all_demonstrations)),
                observations=all_observations,
                frame_observations=frame_observations,
                model_name=self._model_name,
                frame_extraction_seconds=frame_extraction_seconds,
                model_load_seconds=self._model_load_seconds,
                inference_seconds=inference_time,
                total_processing_seconds=total_time,
            )

        except Exception as exc:
            logger.error("Vision inference error: %s", exc)
            return VisionResult(
                success=False,
                media_path=str(media_path) if media_path else None,
                input_type=input_type,
                failure_category=VisionFailureCategory.VISION_INFERENCE_FAILED.value,
                failure_message=str(exc),
                total_processing_seconds=time.perf_counter() - total_t0,
            )


def _extract_visual_tags(description: str) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Extract heuristic visual tags (subjects, objects, actions, scenes, demonstrations) from description."""
    desc_lower = description.lower()

    # Subjects
    subjects = []
    for s in ["person", "man", "woman", "speaker", "presenter", "child", "dog", "cat", "robot", "hand", "people"]:
        if s in desc_lower:
            subjects.append(s)

    # Objects
    objects = []
    for o in ["laptop", "computer", "phone", "screen", "desk", "text", "paper", "chart", "table", "chair", "microphone", "camera", "whiteboard", "book", "code"]:
        if o in desc_lower:
            objects.append(o)

    # Actions
    actions = []
    for a in ["speaking", "talking", "presenting", "gesturing", "typing", "coding", "demonstrating", "showing", "explaining", "sitting", "standing", "holding", "looking"]:
        if a in desc_lower:
            actions.append(a)

    # Scenes
    scenes = []
    for sc in ["office", "room", "studio", "indoor", "outdoor", "stage", "conference", "home", "classroom", "workshop"]:
        if sc in desc_lower:
            scenes.append(sc)

    # Demonstrations
    demos = []
    if any(k in desc_lower for k in ["demonstrat", "showing how", "step", "tutorial", "code", "explain", "diagram"]):
        demos.append("visual_demonstration_or_explanation")

    return subjects, objects, actions, scenes, demos
