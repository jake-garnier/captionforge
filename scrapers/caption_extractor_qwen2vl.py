"""
Qwen2-VL-2B caption extractor - middle-ground VLM for OCR.

Uses Qwen/Qwen2-VL-2B-Instruct with:
- Single GPU operation (fits in ~5GB VRAM)
- FP16 precision for good accuracy
- VLM-style text recognition (better than traditional OCR for stylized text)

This is a balanced option between:
- OlmOCR-7B (SOTA but requires multi-GPU, slow)
- PaddleOCR (fast but less accurate on stylized text)
"""
import cv2
import numpy as np
import torch
import logging
import base64
from io import BytesIO
from typing import List, Dict, Optional
from PIL import Image
import re

logger = logging.getLogger(__name__)

# Global model instance (loaded once, reused)
_model = None
_processor = None
_last_used = None  # Timestamp of last model usage for auto-unload


def get_qwen2vl_model(gpu_id: int = 0):
    """
    Load Qwen2-VL-2B model on a single GPU.
    Model is cached globally to avoid reloading.

    Args:
        gpu_id: GPU device ID (default: 0)
    """
    global _model, _processor, _last_used
    import time

    # Update last used timestamp
    _last_used = time.time()

    if _model is not None:
        return _model, _processor

    logger.info(f"Loading Qwen2-VL-2B model on GPU {gpu_id}...")

    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        # Load model in FP16 on single GPU (~5GB VRAM)
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        ).eval()

        # Load processor
        _processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            trust_remote_code=True,
        )

        logger.info(f"Qwen2-VL-2B loaded successfully on {device}")

        # Log VRAM usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            logger.info(f"GPU {gpu_id} VRAM allocated: {allocated:.2f} GB")

        return _model, _processor

    except Exception as e:
        logger.error(f"Failed to load Qwen2-VL-2B model: {e}")
        raise


def is_qwen2vl_loaded() -> bool:
    """Check if Qwen2-VL-2B model is currently loaded in memory."""
    return _model is not None


def get_qwen2vl_model_info() -> dict:
    """
    Get info about the currently loaded Qwen2-VL model.

    Returns:
        Dictionary with model info or None if not loaded.
    """
    global _model, _last_used

    if _model is None:
        return None

    import time

    # Calculate idle time
    idle_seconds = int(time.time() - _last_used) if _last_used else 0

    return {
        "name": "Qwen2-VL-2B-Instruct",
        "type": "ocr",
        "vram_estimate_gb": 5.0,
        "idle_seconds": idle_seconds,
    }


class Qwen2VLCaptionExtractor:
    """
    Caption extractor using Qwen2-VL-2B VLM.

    Features:
    - Single GPU operation (~5GB VRAM in FP16)
    - VLM-based text recognition (understands context)
    - Good balance of speed and accuracy
    - No cross-GPU P2P issues
    """

    def __init__(
        self,
        gpu_id: int = 0,
        max_new_tokens: int = 1024,
        enable_deduplication: bool = True,
        dedup_threshold: int = 5,
    ):
        """
        Initialize Qwen2-VL-2B extractor.

        Args:
            gpu_id: GPU device to use (default: 0)
            max_new_tokens: Maximum tokens to generate per image
            enable_deduplication: Use perceptual hashing to skip duplicate frames
            dedup_threshold: Frame similarity threshold (0-64, lower=stricter)
        """
        self.gpu_id = gpu_id
        self.max_new_tokens = max_new_tokens
        self.enable_deduplication = enable_deduplication
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        # Load model (cached globally)
        self.model, self.processor = get_qwen2vl_model(gpu_id)

        # Initialize frame deduplicator if enabled
        if enable_deduplication:
            from scrapers.frame_deduplicator import FrameDeduplicator
            self.deduplicator = FrameDeduplicator(
                similarity_threshold=dedup_threshold,
                hash_method='phash'
            )
            logger.info(f"Frame deduplicator initialized (threshold={dedup_threshold})")
        else:
            self.deduplicator = None

        logger.info("Qwen2-VL-2B caption extractor initialized")

    def _frame_to_pil(self, frame: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR frame to PIL RGB Image."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    def _resize_image(self, image: Image.Image, max_size: int = 1280) -> Image.Image:
        """
        Resize image if needed. Qwen2-VL handles various sizes but
        limiting to 1280 max dimension for memory efficiency.
        """
        width, height = image.size
        longest = max(width, height)

        if longest <= max_size:
            return image

        scale = max_size / longest
        new_width = int(width * scale)
        new_height = int(height * scale)

        logger.debug(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _build_ocr_prompt(self) -> str:
        """Build the OCR prompt for text extraction."""
        return (
            "Read all the text in this image carefully. "
            "Output only the text content, preserving the reading order "
            "from top to bottom, left to right. "
            "Do not add any explanations or descriptions."
        )

    def _clean_output(self, text: str) -> str:
        """
        Clean up model output - remove artifacts and noise.
        """
        # Remove excessive repeated characters
        text = re.sub(r'(.)\1{4,}', r'\1\1', text)

        # Remove lines that are just punctuation/symbols
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and re.search(r'[a-zA-Z0-9]', stripped):
                cleaned_lines.append(stripped)

        return '\n'.join(cleaned_lines)

    def _normalize_text_for_dedup(self, text: str) -> str:
        """
        Normalize text for deduplication comparison.
        Makes text lowercase, removes punctuation and extra whitespace.
        This catches frames with same text but different styling.
        """
        if not text:
            return ""

        # Lowercase
        normalized = text.lower()

        # Remove punctuation (keep alphanumeric and spaces)
        normalized = re.sub(r'[^\w\s]', '', normalized)

        # Collapse whitespace
        normalized = ' '.join(normalized.split())

        # Only consider text long enough to be meaningful
        if len(normalized) < 10:
            return ""

        return normalized

    def extract_text_from_frame(self, frame: np.ndarray) -> List[str]:
        """
        Extract text from a single frame using Qwen2-VL-2B.

        Args:
            frame: Video frame (BGR format from OpenCV)

        Returns:
            List of detected text strings
        """
        try:
            # Convert to PIL Image and resize
            pil_image = self._frame_to_pil(frame)
            pil_image = self._resize_image(pil_image)

            # Build conversation format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": self._build_ocr_prompt()},
                    ],
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                     for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            # Decode output
            generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]

            # Clean up the output
            output_text = self._clean_output(output_text)

            # Split into lines and filter empty
            texts = [line.strip() for line in output_text.split('\n') if line.strip()]

            return texts

        except Exception as e:
            logger.warning(f"Qwen2-VL extraction failed: {e}")
            return []

    def extract_from_video(
        self,
        video_path: str,
        sample_rate: int = 60,
        video_id: int = None,
        max_duration_seconds: int = 1800,
    ) -> Dict:
        """
        Extract captions from video by sampling frames.

        Args:
            video_path: Path to video file
            sample_rate: Sample every Nth frame (default: 60 = ~0.5 fps for 30fps video)
            video_id: Video ID for heartbeat tracking
            max_duration_seconds: Max processing time

        Returns:
            Dictionary with extracted caption and metadata
        """
        try:
            from utils.worker_heartbeat import heartbeat
        except ImportError:
            heartbeat = lambda **kwargs: None

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return {'caption_text': '', 'num_frames_processed': 0, 'error': 'Failed to open video'}

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"Processing video with Qwen2-VL-2B: {total_frames} frames at {fps:.1f} fps")

            frame_texts = []
            frames_sampled = 0
            frames_processed = 0
            frames_skipped_dedup = 0
            frames_skipped_text_dedup = 0  # New: text-based deduplication counter
            frame_idx = 0

            seen_hashes = []
            seen_texts = set()  # New: track normalized text content to skip duplicates

            import time
            start_time = time.time()
            last_heartbeat = start_time

            while True:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > max_duration_seconds:
                    logger.warning(f"Qwen2-VL timeout after {elapsed:.0f}s - processed {frames_processed} frames")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    frames_sampled += 1

                    # Heartbeat every 5 frames
                    if frames_sampled % 5 == 0:
                        current_time = time.time()
                        if current_time - last_heartbeat > 10:
                            heartbeat(
                                task_name='extract_caption_qwen2vl',
                                video_id=video_id,
                                stage=f'qwen2vl_frame_{frames_processed}'
                            )
                            last_heartbeat = current_time

                    # Frame deduplication
                    is_duplicate = False
                    if self.enable_deduplication and self.deduplicator:
                        frame_hash = self.deduplicator.compute_frame_hash(frame)

                        for seen_hash in seen_hashes:
                            if self.deduplicator.is_similar(frame_hash, seen_hash):
                                is_duplicate = True
                                frames_skipped_dedup += 1
                                break

                        if not is_duplicate:
                            seen_hashes.append(frame_hash)

                    # Process unique frames
                    if not is_duplicate:
                        texts = self.extract_text_from_frame(frame)
                        frames_processed += 1

                        if texts:
                            frame_text = ' '.join(texts)

                            # Text-based deduplication: normalize and check if we've seen this text
                            normalized_text = self._normalize_text_for_dedup(frame_text)

                            if normalized_text and normalized_text not in seen_texts:
                                # New unique text content
                                frame_texts.append(frame_text)
                                seen_texts.add(normalized_text)
                            elif normalized_text:
                                # Text was seen before, skip it
                                frames_skipped_text_dedup += 1
                                logger.debug(f"Skipping duplicate text: {frame_text[:50]}...")

                        # Log progress every 5 frames
                        if frames_processed % 5 == 0:
                            logger.info(f"Qwen2-VL progress: {frames_processed} frames, {len(frame_texts)} unique text segments")

                frame_idx += 1

            cap.release()

            # Stats
            visual_dedup_reduction = 0
            text_dedup_reduction = 0
            if frames_sampled > 0:
                visual_dedup_reduction = (frames_skipped_dedup / frames_sampled) * 100
            if frames_processed > 0:
                text_dedup_reduction = (frames_skipped_text_dedup / frames_processed) * 100

            logger.info(
                f"Qwen2-VL complete: {frames_sampled} sampled, "
                f"{frames_processed} processed, "
                f"{frames_skipped_dedup} visual dupes ({visual_dedup_reduction:.1f}%), "
                f"{frames_skipped_text_dedup} text dupes ({text_dedup_reduction:.1f}%)"
            )

            if not frame_texts:
                return {
                    'caption_text': '',
                    'num_frames_processed': frames_processed,
                    'frames_sampled': frames_sampled,
                    'frames_skipped_dedup': frames_skipped_dedup,
                    'frames_skipped_text_dedup': frames_skipped_text_dedup,
                    'total_frames': total_frames,
                    'extractor': 'qwen2vl'
                }

            # Join frames with delimiter
            caption_text = ' *|* '.join(frame_texts)

            logger.info(f"Qwen2-VL extracted: {len(caption_text)} chars from {len(frame_texts)} unique text segments")

            return {
                'caption_text': caption_text,
                'num_frames_processed': frames_processed,
                'frames_sampled': frames_sampled,
                'frames_skipped_dedup': frames_skipped_dedup,
                'frames_skipped_text_dedup': frames_skipped_text_dedup,
                'visual_dedup_reduction_percent': round(visual_dedup_reduction, 1),
                'text_dedup_reduction_percent': round(text_dedup_reduction, 1),
                'total_frames': total_frames,
                'unique_text_segments': len(frame_texts),
                'extractor': 'qwen2vl'
            }

        except Exception as e:
            logger.error(f"Qwen2-VL video extraction failed: {e}")
            return {'caption_text': '', 'num_frames_processed': 0, 'error': str(e), 'extractor': 'qwen2vl'}

    def extract_from_image(self, image_path: str, video_id: int = None) -> Dict:
        """
        Extract captions from a static image.

        Args:
            image_path: Path to image file
            video_id: Video ID for heartbeat tracking

        Returns:
            Dictionary with extracted caption and metadata
        """
        try:
            from utils.worker_heartbeat import heartbeat
        except ImportError:
            heartbeat = lambda **kwargs: None

        try:
            heartbeat(task_name='extract_caption_qwen2vl', video_id=video_id, stage='loading_image')

            image = cv2.imread(image_path)

            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return {'caption_text': '', 'num_frames_processed': 0, 'error': 'Failed to load image'}

            height, width = image.shape[:2]
            logger.info(f"Processing image with Qwen2-VL-2B: {width}x{height}")

            heartbeat(task_name='extract_caption_qwen2vl', video_id=video_id, stage='qwen2vl_image')

            texts = self.extract_text_from_frame(image)

            if not texts:
                return {
                    'caption_text': '',
                    'num_frames_processed': 1,
                    'total_frames': 1,
                    'resolution': f'{width}x{height}',
                    'extractor': 'qwen2vl'
                }

            caption_text = ' '.join(texts)

            logger.info(f"Qwen2-VL extracted from image: {len(caption_text)} chars, {len(texts)} segments")

            return {
                'caption_text': caption_text,
                'num_frames_processed': 1,
                'total_frames': 1,
                'unique_text_segments': len(texts),
                'resolution': f'{width}x{height}',
                'extractor': 'qwen2vl'
            }

        except Exception as e:
            logger.error(f"Qwen2-VL image extraction failed: {e}")
            return {'caption_text': '', 'num_frames_processed': 0, 'error': str(e), 'extractor': 'qwen2vl'}

    def extract_from_gif(self, gif_path: str, video_id: int = None, max_frames: int = 30) -> Dict:
        """
        Extract captions from an animated GIF by sampling multiple frames.

        Uses PIL to properly read all frames of an animated GIF, unlike cv2.imread()
        which only reads the first frame.

        Args:
            gif_path: Path to GIF file
            video_id: Video ID for heartbeat tracking
            max_frames: Maximum frames to process (evenly distributed across GIF)

        Returns:
            Dictionary with extracted caption and metadata
        """
        try:
            from utils.worker_heartbeat import heartbeat
        except ImportError:
            heartbeat = lambda **kwargs: None

        try:
            heartbeat(task_name='extract_caption_qwen2vl', video_id=video_id, stage='loading_gif')

            # Open GIF with PIL to access all frames
            gif = Image.open(gif_path)

            # Count total frames
            total_frames = 0
            try:
                while True:
                    total_frames += 1
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass

            # Reset to first frame
            gif.seek(0)

            logger.info(f"Processing animated GIF with {total_frames} frames: {gif_path}")

            # If single frame (static GIF), treat as image
            if total_frames <= 1:
                logger.info("GIF has only 1 frame, treating as static image")
                gif.close()
                return self.extract_from_image(gif_path, video_id=video_id)

            # Calculate sample rate to get max_frames evenly distributed
            sample_rate = max(1, total_frames // max_frames)

            frame_texts = []
            frames_processed = 0
            frames_skipped_dedup = 0
            frames_skipped_text_dedup = 0
            seen_hashes = []
            seen_texts = set()

            import time
            start_time = time.time()
            last_heartbeat = start_time

            for frame_idx in range(total_frames):
                try:
                    gif.seek(frame_idx)
                except EOFError:
                    break

                # Sample every Nth frame
                if frame_idx % sample_rate != 0:
                    continue

                # Heartbeat every 5 processed frames
                if frames_processed % 5 == 0:
                    current_time = time.time()
                    if current_time - last_heartbeat > 10:
                        heartbeat(
                            task_name='extract_caption_qwen2vl',
                            video_id=video_id,
                            stage=f'qwen2vl_gif_frame_{frames_processed}'
                        )
                        last_heartbeat = current_time

                # Convert PIL frame to numpy array (RGB)
                frame = gif.convert('RGB')
                frame_array = np.array(frame)
                # Convert RGB to BGR for OpenCV compatibility
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

                # Frame deduplication
                is_duplicate = False
                if self.enable_deduplication and self.deduplicator:
                    frame_hash = self.deduplicator.compute_frame_hash(frame_bgr)

                    for seen_hash in seen_hashes:
                        if self.deduplicator.is_similar(frame_hash, seen_hash):
                            is_duplicate = True
                            frames_skipped_dedup += 1
                            break

                    if not is_duplicate:
                        seen_hashes.append(frame_hash)

                # Process unique frames
                if not is_duplicate:
                    texts = self.extract_text_from_frame(frame_bgr)
                    frames_processed += 1

                    if texts:
                        frame_text = ' '.join(texts)

                        # Text-based deduplication
                        normalized_text = self._normalize_text_for_dedup(frame_text)

                        if normalized_text and normalized_text not in seen_texts:
                            frame_texts.append(frame_text)
                            seen_texts.add(normalized_text)
                        elif normalized_text:
                            frames_skipped_text_dedup += 1
                            logger.debug(f"Skipping duplicate GIF text: {frame_text[:50]}...")

                    # Log progress every 5 frames
                    if frames_processed % 5 == 0:
                        logger.info(f"GIF progress: {frames_processed} frames, {len(frame_texts)} unique text segments")

            gif.close()

            # Stats
            frames_sampled = (total_frames // sample_rate) + (1 if total_frames % sample_rate else 0)
            visual_dedup_reduction = (frames_skipped_dedup / frames_sampled * 100) if frames_sampled > 0 else 0
            text_dedup_reduction = (frames_skipped_text_dedup / frames_processed * 100) if frames_processed > 0 else 0

            logger.info(
                f"GIF extraction complete: {total_frames} total, "
                f"{frames_sampled} sampled, {frames_processed} processed, "
                f"{frames_skipped_dedup} visual dupes ({visual_dedup_reduction:.1f}%), "
                f"{frames_skipped_text_dedup} text dupes ({text_dedup_reduction:.1f}%)"
            )

            if not frame_texts:
                return {
                    'caption_text': '',
                    'num_frames_processed': frames_processed,
                    'frames_sampled': frames_sampled,
                    'frames_skipped_dedup': frames_skipped_dedup,
                    'frames_skipped_text_dedup': frames_skipped_text_dedup,
                    'total_frames': total_frames,
                    'extractor': 'qwen2vl'
                }

            # Join frames with delimiter
            caption_text = ' *|* '.join(frame_texts)

            logger.info(f"GIF extracted: {len(caption_text)} chars from {len(frame_texts)} unique text segments")

            return {
                'caption_text': caption_text,
                'num_frames_processed': frames_processed,
                'frames_sampled': frames_sampled,
                'frames_skipped_dedup': frames_skipped_dedup,
                'frames_skipped_text_dedup': frames_skipped_text_dedup,
                'visual_dedup_reduction_percent': round(visual_dedup_reduction, 1),
                'text_dedup_reduction_percent': round(text_dedup_reduction, 1),
                'total_frames': total_frames,
                'unique_text_segments': len(frame_texts),
                'extractor': 'qwen2vl'
            }

        except Exception as e:
            logger.error(f"GIF extraction failed: {e}")
            return {'caption_text': '', 'num_frames_processed': 0, 'error': str(e), 'extractor': 'qwen2vl'}

    def extract_from_gallery(self, gallery_paths: List[str], video_id: int = None) -> Dict:
        """
        Extract captions from all images in a gallery.

        Args:
            gallery_paths: List of paths to gallery images
            video_id: Video ID for heartbeat tracking

        Returns:
            Dictionary with extracted captions from all images
        """
        try:
            from utils.worker_heartbeat import heartbeat
        except ImportError:
            heartbeat = lambda **kwargs: None

        try:
            all_texts = []
            images_processed = 0

            for idx, image_path in enumerate(gallery_paths):
                heartbeat(
                    task_name='extract_caption_qwen2vl',
                    video_id=video_id,
                    stage=f'qwen2vl_gallery_{idx+1}_of_{len(gallery_paths)}'
                )

                result = self.extract_from_image(image_path, video_id=None)
                if result.get('caption_text'):
                    all_texts.append(result['caption_text'])
                    images_processed += 1

            if not all_texts:
                return {
                    'caption_text': '',
                    'num_frames_processed': len(gallery_paths),
                    'total_frames': len(gallery_paths),
                    'extractor': 'qwen2vl'
                }

            caption_text = ' *|* '.join(all_texts)

            logger.info(f"Qwen2-VL gallery: {len(caption_text)} chars from {images_processed}/{len(gallery_paths)} images")

            return {
                'caption_text': caption_text,
                'num_frames_processed': len(gallery_paths),
                'total_frames': len(gallery_paths),
                'images_with_text': images_processed,
                'unique_text_segments': len(all_texts),
                'extractor': 'qwen2vl'
            }

        except Exception as e:
            logger.error(f"Qwen2-VL gallery extraction failed: {e}")
            return {'caption_text': '', 'num_frames_processed': 0, 'error': str(e), 'extractor': 'qwen2vl'}


def cleanup_model():
    """Release GPU memory by unloading the model.

    Properly frees CUDA memory by:
    1. Moving model to CPU (releases VRAM)
    2. Deleting all references
    3. Running garbage collection
    4. Synchronizing and emptying CUDA cache
    """
    global _model, _processor, _last_used
    import gc

    if _model is not None:
        try:
            # Move model to CPU first to release VRAM
            _model.to('cpu')
        except Exception as e:
            logger.debug(f"Could not move model to CPU: {e}")
        del _model
        _model = None

    if _processor is not None:
        del _processor
        _processor = None

    _last_used = None

    # Force garbage collection to release Python references
    gc.collect()

    if torch.cuda.is_available():
        # Synchronize to ensure all operations complete
        torch.cuda.synchronize()
        # Empty the cache
        torch.cuda.empty_cache()
        # Run GC again after cache clear
        gc.collect()

    logger.info("Qwen2-VL-2B model unloaded and GPU memory cleared")


def unload_if_idle(idle_timeout_minutes: int = 5) -> bool:
    """
    Unload model if it hasn't been used recently.

    Args:
        idle_timeout_minutes: Minutes of inactivity before unloading (default: 5)

    Returns:
        True if model was unloaded, False otherwise
    """
    global _model, _last_used
    import time

    if _model is None:
        return False  # Nothing to unload

    if _last_used is None:
        return False  # No usage timestamp

    idle_seconds = time.time() - _last_used
    idle_minutes = idle_seconds / 60

    if idle_minutes >= idle_timeout_minutes:
        logger.info(f"Qwen2-VL-2B idle for {idle_minutes:.1f} minutes, unloading...")
        cleanup_model()
        return True

    return False


def is_model_loaded() -> bool:
    """Check if model is currently loaded."""
    return _model is not None


def get_model_status() -> dict:
    """Get current model status including idle time."""
    import time

    status = {
        "loaded": _model is not None,
        "last_used": None,
        "idle_seconds": None,
    }

    if _last_used is not None:
        status["last_used"] = _last_used
        status["idle_seconds"] = time.time() - _last_used

    return status
