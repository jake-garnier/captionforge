"""
Perceptual frame deduplication using image hashing
Skips processing visually similar frames to reduce duplicate OCR extractions
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass
import imagehash
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    """Information about a video frame"""
    frame_index: int
    timestamp: float
    hash_value: imagehash.ImageHash
    is_unique: bool = True


class FrameDeduplicator:
    """
    Deduplicate video frames using perceptual hashing
    Identifies visually similar frames to avoid redundant OCR processing
    """

    def __init__(
        self,
        hash_size: int = 8,
        similarity_threshold: int = 5,
        hash_method: str = 'phash'
    ):
        """
        Initialize frame deduplicator

        Args:
            hash_size: Size of perceptual hash (default: 8x8 = 64-bit hash)
            similarity_threshold: Maximum hamming distance for "similar" frames (0-64)
                                 Lower = stricter (more unique frames)
                                 Higher = looser (fewer unique frames)
                                 Recommended: 3-7 for caption extraction
            hash_method: Hash algorithm - 'phash' (perceptual), 'dhash' (difference),
                        'ahash' (average), or 'whash' (wavelet)
        """
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold
        self.hash_method = hash_method

        # Map hash method names to functions
        self.hash_func = {
            'phash': lambda img: imagehash.phash(img, hash_size=hash_size),
            'dhash': lambda img: imagehash.dhash(img, hash_size=hash_size),
            'ahash': lambda img: imagehash.average_hash(img, hash_size=hash_size),
            'whash': lambda img: imagehash.whash(img, hash_size=hash_size)
        }.get(hash_method, imagehash.phash)

        logger.info(
            f"FrameDeduplicator initialized: "
            f"hash_method={hash_method}, "
            f"hash_size={hash_size}, "
            f"similarity_threshold={similarity_threshold}"
        )

    def compute_frame_hash(self, frame: np.ndarray) -> imagehash.ImageHash:
        """
        Compute perceptual hash for a frame

        Args:
            frame: OpenCV frame (BGR format)

        Returns:
            Perceptual hash of the frame
        """
        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Compute hash
        return self.hash_func(pil_image)

    def is_similar(self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> bool:
        """
        Check if two frame hashes are similar

        Args:
            hash1: First frame hash
            hash2: Second frame hash

        Returns:
            True if frames are similar (within threshold)
        """
        # Hamming distance = number of different bits
        distance = hash1 - hash2
        return distance <= self.similarity_threshold

    def find_unique_frames(
        self,
        video_path: str,
        sample_rate: int = 1,
        scene_change_threshold: Optional[float] = None
    ) -> List[FrameInfo]:
        """
        Identify unique frames in a video

        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame (for initial sampling)
            scene_change_threshold: Optional histogram difference threshold for scene changes
                                   If provided, marks scene boundaries as unique

        Returns:
            List of FrameInfo objects, with is_unique=True for frames to process
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Analyzing {total_frames} frames for deduplication (sample_rate={sample_rate})")

        frame_infos: List[FrameInfo] = []
        last_unique_hash: Optional[imagehash.ImageHash] = None
        last_frame_gray = None
        unique_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process sampled frames
            if frame_idx % sample_rate == 0:
                timestamp = frame_idx / fps if fps > 0 else 0

                # Compute perceptual hash
                frame_hash = self.compute_frame_hash(frame)

                # Check if this is a unique frame
                is_unique = False

                # First frame is always unique
                if last_unique_hash is None:
                    is_unique = True
                else:
                    # Check perceptual similarity
                    if not self.is_similar(frame_hash, last_unique_hash):
                        is_unique = True

                    # Additional scene change detection (if enabled)
                    if scene_change_threshold is not None and last_frame_gray is not None:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        hist_diff = self._compute_histogram_difference(last_frame_gray, frame_gray)

                        if hist_diff > scene_change_threshold:
                            is_unique = True
                            logger.debug(f"Scene change detected at frame {frame_idx} (diff={hist_diff:.2f})")

                        last_frame_gray = frame_gray

                if is_unique:
                    last_unique_hash = frame_hash
                    if scene_change_threshold is not None and last_frame_gray is None:
                        last_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    unique_count += 1

                frame_info = FrameInfo(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    hash_value=frame_hash,
                    is_unique=is_unique
                )
                frame_infos.append(frame_info)

            frame_idx += 1

        cap.release()

        sampled_count = len(frame_infos)
        reduction_pct = 100 * (1 - unique_count / sampled_count) if sampled_count > 0 else 0

        logger.info(
            f"Frame deduplication complete: "
            f"{unique_count} unique / {sampled_count} sampled "
            f"({reduction_pct:.1f}% reduction)"
        )

        return frame_infos

    def _compute_histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute histogram difference between two grayscale frames
        Used for scene change detection

        Args:
            frame1: First grayscale frame
            frame2: Second grayscale frame

        Returns:
            Histogram difference score (0-1, higher = more different)
        """
        # Compute histograms
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])

        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        # Compute correlation (higher = more similar)
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Convert to difference (higher = more different)
        return 1.0 - correlation

    def get_unique_frame_indices(self, frame_infos: List[FrameInfo]) -> List[int]:
        """
        Extract list of unique frame indices

        Args:
            frame_infos: List of FrameInfo objects

        Returns:
            List of frame indices to process
        """
        return [info.frame_index for info in frame_infos if info.is_unique]
