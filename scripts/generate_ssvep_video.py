"""
SSVEP Stimulus Video Generator
==============================
This module generates high-fidelity Steady-State Visual Evoked Potential (SSVEP)
stimulus videos with precise frequency control. It supports both standard solid-color
modulations and alpha-blended transparent stimulations on custom backgrounds.
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Union, Optional

import cv2
import numpy as np

# Configure standard logging for academic transparency
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure FFmpeg codec settings for lossless/high-quality H.264 rendering
os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"] = "crf;0|preset;veryslow"


def check_frequency_jitter(freq: float, fps: float) -> bool:
    """
    Verifies if the rendering frequency is a perfect multiple of the display frame rate.
    This check ensures zero phase-drift and zero frame-jitter as mathematically 
    derived in the paradigm specifications.

    Args:
        freq: Target flickering frequency (Hz).
        fps: Video frames per second (Hz).

    Returns:
        bool: True if zero-jitter criteria are met, False otherwise.
    """
    frames_per_cycle = fps / freq
    is_integer = np.isclose(frames_per_cycle, round(frames_per_cycle))
    if not is_integer:
        logger.warning(
            f"Non-integer division observed: {fps} FPS / {freq} Hz = {frames_per_cycle:.4f} frames per cycle. "
            f"This discrepancy will introduce temporal jitter and phase drift during playback. "
            f"Consider adjusting FPS to an integer multiple of the frequency (e.g., {freq * round(fps / freq)} Hz)."
        )
        return False
    logger.info(f"Zero-jitter verification passed: {frames_per_cycle:.1f} frames per stimulus cycle.")
    return True


def generate_ssvep_video(
        freq: float,
        duration: float,
        fps: int,
        width: int,
        height: int,
        output_path: Union[str, Path],
        color1: Tuple[int, int, int] = (0, 0, 0),
        color2: Tuple[int, int, int] = (255, 255, 255),
        wave_type: str = 'sine',
        background_path: Optional[Union[str, Path]] = None,
        transparency: float = 1.0,
        fourcc_code: str = 'avc1'
) -> None:
    """
    Generates an SSVEP visual stimulus video with exact timing and customizable overlays.

    Args:
        freq: Target flickering frequency in Hz.
        duration: Duration of the video in seconds.
        fps: Frames per second (must match display refresh rate).
        width: Frame width in pixels.
        height: Frame height in pixels.
        output_path: Destination path for the saved video.
        color1: RGB boundary color 1 (BGR format for OpenCV), default black (0, 0, 0).
        color2: RGB boundary color 2 (BGR format for OpenCV), default white (255, 255, 255).
        wave_type: Waveform modulation style. Options: 'sine' (smooth) or 'square' (binary).
        background_path: Optional path to an image file used as a static background layer.
        transparency: Stimulus layer transparency overlay (1.0 = fully opaque, 0.0 = fully transparent).
        fourcc_code: Video codec identifier (default 'avc1' for H.264).
    """
    # 1. Path and Directory Validation
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 2. Mathematical Rigor Validation
    check_frequency_jitter(freq, fps)

    # 3. Load and Scale Background Image (if specified)
    bg_frame = None
    if background_path:
        bg_path = Path(background_path).resolve()
        if not bg_path.exists():
            raise FileNotFoundError(f"Specified background image not found: {bg_path}")
        # Read the background and resize it to match the stimulus target resolution
        bg_frame = cv2.imread(str(bg_path))
        if bg_frame is None:
            raise ValueError(f"Failed to decode image from: {bg_path}")
        bg_frame = cv2.resize(bg_frame, (width, height))
        logger.info(f"Loaded background image from {bg_path} with resolution {width}x{height}.")

    # 4. Prepare Constant Base Arrays to Optimize Loop Computation (Vectorization)
    img_color1 = np.full((height, width, 3), color1, dtype=np.uint8)
    img_color2 = np.full((height, width, 3), color2, dtype=np.uint8)

    # 5. Initialize OpenCV VideoWriter using the requested FFMPEG codec
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    writer = cv2.VideoWriter(
        str(out_file),
        cv2.CAP_FFMPEG,
        fourcc,
        float(fps),
        (width, height),
        isColor=True
    )

    if not writer.isOpened():
        # Fallback to standard MP4V writer if system does not support AVC1/H264 natively
        logger.error(f"Failed to open video writer with codec '{fourcc_code}'. Attempting 'mp4v' fallback.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_file), fourcc, float(fps), (width, height), isColor=True)
        if not writer.isOpened():
            raise IOError("Unable to initialize OpenCV VideoWriter with native or fallback codecs.")

    num_frames = int(duration * fps)
    logger.info(f"Rendering {num_frames} frames ({duration}s @ {fps} FPS)...")

    try:
        for i in range(num_frames):
            # Calculate instant modulation weight (alpha)
            if wave_type == 'sine':
                # Sine wave: continuous sinusoidal color blending
                alpha = 0.5 * (1.0 + np.sin(2.0 * np.pi * freq * (i / fps)))
            elif wave_type == 'square':
                # Square wave: discrete switching
                alpha = 1.0 if (i * freq * 2 / fps) % 2 < 1 else 0.0
            else:
                raise ValueError(f"Unsupported wave_type '{wave_type}'. Choose 'sine' or 'square'.")

            # Interpolate between color1 and color2 using vector scaling (OpenCV implementation)
            stimulus_frame = cv2.addWeighted(img_color1, 1.0 - alpha, img_color2, alpha, 0)

            # Perform background blending based on transparency parameter
            if bg_frame is not None and transparency < 1.0:
                # Formula: I(t) = transparency * S(t) + (1 - transparency) * B(t)
                rendered_frame = cv2.addWeighted(stimulus_frame, transparency, bg_frame, 1.0 - transparency, 0)
            else:
                rendered_frame = stimulus_frame

            writer.write(rendered_frame)

    finally:
        writer.release()
        logger.info(f"Video file generated successfully: {out_file}")


# --- Exemplary Usage demonstrating experimental paradigms ---
if __name__ == "__main__":
    # Create sample media directory for testing output
    output_directory = Path("./media/videos")
    output_directory.mkdir(parents=True, exist_ok=True)

    # 1. Standard Paradigm: Black-White Sine Flicker at 10 Hz (Opaque)
    generate_ssvep_video(
        freq=10.0,
        duration=5.0,
        fps=60,
        width=512,
        height=512,
        output_path=output_directory / "ssvep_bw_10hz.mp4",
        color1=(0, 0, 0),  # Black
        color2=(255, 255, 255),  # White
        wave_type='sine'
    )

    # 2. Ergonomic Paradigm: Comfortable Green-Black Sine Flicker at 12 Hz (Opaque)
    generate_ssvep_video(
        freq=12.0,
        duration=5.0,
        fps=60,
        width=512,
        height=512,
        output_path=output_directory / "ssvep_gb_12hz.mp4",
        color1=(0, 0, 0),  # Black
        color2=(0, 255, 0),  # Green (BGR format: 0, 255, 0)
        wave_type='sine'
    )
