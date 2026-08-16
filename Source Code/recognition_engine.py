"""
recognition_engine.py — camera capture, hand segmentation and letter recognition.

Segmentation notes
------------------
The CNN was trained on clean binary silhouettes: one white hand on pure black,
filling roughly a third of a 64x64 frame (see Dataset/training_set). Anything
that does not look like that — speckle, a second blob, a fully white frame —
produces a confident but meaningless letter, so the mask is auto-thresholded,
cleaned and reduced to its largest blob before it reaches the model.
"""
import os
from collections import Counter, deque

import cv2
import numpy as np

from model_compat import load_asl_model

SAMPLE_DIR = "SampleGestures"
LABELS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# The region of interest is a square on the right of the frame, sized as a
# fraction of frame height so it behaves the same on any webcam resolution.
ROI_SIZE_FRAC = 0.42
ROI_CENTER_X_FRAC = 0.72
ROI_CENTER_Y_FRAC = 0.42

# A usable mask covers a sensible slice of the box. Outside this band we are
# looking at an empty box or a blown-out threshold, not a hand.
MIN_FILL = 0.08
MAX_FILL = 0.75

# Otsu always splits the box in two, even when it holds nothing but a wall, and
# the model will happily label that noise. Requiring the two sides to differ in
# brightness is what separates a real hand from an empty box: measured over the
# training set a hand clears 26 even in poor light, while an empty box sits
# near 13.
MIN_SEPARATION = 22.0

# A letter is only reported once it holds still: it must win a majority of the
# recent frames and clear the confidence bar.
HISTORY_LEN = 9
MIN_VOTES = 5
MIN_CONFIDENCE = 0.60

# Custom gestures are silhouettes, so overlap compares them far more reliably
# than keypoint matching (SIFT finds almost nothing on a smooth blob).
CUSTOM_MATCH_IOU = 0.75


class RecognitionEngine:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ASLModel.h5')
        self.classifier = load_asl_model(model_path)
        self.cam = None

        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._history = deque(maxlen=HISTORY_LEN)

        self._samples = {}       # name -> 64x64 binary mask
        self._samples_mtime = None

    # ── Camera lifecycle ───────────────────────────────────────────────────

    def start(self):
        if self.cam is None or not self.cam.isOpened():
            self.cam = cv2.VideoCapture(0)
        self._history.clear()

    def stop(self):
        if self.cam is not None:
            self.cam.release()
            self.cam = None
        self._history.clear()

    # ── Segmentation ───────────────────────────────────────────────────────

    @staticmethod
    def roi_box(width, height):
        """Return (x1, y1, x2, y2) of the capture square for a frame."""
        side = int(height * ROI_SIZE_FRAC)
        cx = int(width * ROI_CENTER_X_FRAC)
        cy = int(height * ROI_CENTER_Y_FRAC)
        half = side // 2
        x1 = max(0, min(cx - half, width - side))
        y1 = max(0, min(cy - half, height - side))
        return x1, y1, x1 + side, y1 + side

    def _segment(self, roi_bgr, manual_threshold=None):
        """Binary mask of the hand: white hand, black background.

        With no manual threshold, Otsu picks the split between the hand and
        the background automatically, which is what makes this work across
        different rooms and lighting instead of only the one it was tuned in.

        Returns (mask, separation) where separation is the brightness gap
        between the two sides of the split.
        """
        value = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        value = cv2.GaussianBlur(value, (7, 7), 0)

        if manual_threshold is None:
            _, mask = cv2.threshold(value, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(value, int(manual_threshold), 255,
                                    cv2.THRESH_BINARY_INV)

        front, back = value[mask > 0], value[mask == 0]
        separation = (abs(float(front.mean()) - float(back.mean()))
                      if front.size and back.size else 0.0)

        # The hand sits in the middle of the box and the background runs along
        # its edges, so whichever colour dominates the border is the background.
        # This is what lets a hand lighter than its background work too.
        border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
        if border.mean() > 127:
            mask = cv2.bitwise_not(mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return self._largest_blob_bbox(mask), separation

    @staticmethod
    def _largest_blob_bbox(mask):
        """Drop everything but the biggest white region and return it with its bounding box."""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return mask, (0, 0, mask.shape[1], mask.shape[0])
        biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h, _ = stats[biggest]
        return np.where(labels == biggest, 255, 0).astype(np.uint8), (x, y, w, h)

    # ── Custom gestures ────────────────────────────────────────────────────

    def _refresh_samples(self):
        """Reload saved gesture masks when the folder changes on disk."""
        if not os.path.isdir(SAMPLE_DIR):
            self._samples = {}
            return
        mtime = os.path.getmtime(SAMPLE_DIR)
        if mtime == self._samples_mtime:
            return
        self._samples_mtime = mtime

        samples = {}
        for name in os.listdir(SAMPLE_DIR):
            if not name.endswith('.png') or name.startswith('.'):
                continue
            img = cv2.imread(os.path.join(SAMPLE_DIR, name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (64, 64):
                img = cv2.resize(img, (64, 64))
            samples[name[:-4]] = (img > 127)
        self._samples = samples

    def _match_custom(self, mask_64):
        """Best saved gesture whose silhouette overlaps this one, or None."""
        self._refresh_samples()
        if not self._samples:
            return None, 0.0

        current = mask_64 > 127
        best_name, best_iou = None, 0.0
        for name, sample in self._samples.items():
            union = np.logical_or(current, sample).sum()
            if union == 0:
                continue
            iou = np.logical_and(current, sample).sum() / union
            if iou > best_iou:
                best_name, best_iou = name, iou

        if best_iou >= CUSTOM_MATCH_IOU:
            return (' ' if best_name == 'sp' else best_name), float(best_iou)
        return None, float(best_iou)

    # ── Recognition ────────────────────────────────────────────────────────

    def _classify(self, mask_64):
        rgb = cv2.cvtColor(mask_64, cv2.COLOR_GRAY2RGB).astype(np.float32)
        probs = self.classifier.predict(np.expand_dims(rgb, axis=0))[0]
        index = int(np.argmax(probs))
        return LABELS[index], float(probs[index])

    def _stable_letter(self, letter, confidence):
        """Hold a letter back until it wins a majority of the recent frames.

        Raw per-frame output flickers between neighbours while a hand settles;
        voting is what turns that into a letter a person can actually read.
        """
        self._history.append((letter, confidence))
        if len(self._history) < MIN_VOTES:
            return None, 0.0

        votes = Counter(l for l, _ in self._history)
        winner, count = votes.most_common(1)[0]
        if count < MIN_VOTES:
            return None, 0.0

        mean_conf = float(np.mean([c for l, c in self._history if l == winner]))
        if mean_conf < MIN_CONFIDENCE:
            return None, mean_conf
        return winner, mean_conf

    def process_frame(self, threshold=None, flip=True):
        """Grab one frame and recognize it.

        `threshold` is the manual mask cutoff; pass None (the default) to let
        Otsu choose it. Returns None when the camera has no frame to give.
        """
        if self.cam is None or not self.cam.isOpened():
            return None

        ok, frame = self.cam.read()
        if not ok:
            return None

        if flip:
            frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.roi_box(width, height)

        (mask, bbox), separation = self._segment(frame[y1:y2, x1:x2], threshold)
        
        # Center and scale the hand to take up ~65% of the 64x64 frame
        bx, by, bw, bh = bbox
        if bw > 0 and bh > 0:
            cropped = mask[by:by+bh, bx:bx+bw]
            max_dim = max(bw, bh)
            pad_size = int(max_dim / 0.65)
            square = np.zeros((pad_size, pad_size), dtype=np.uint8)
            off_x = (pad_size - bw) // 2
            off_y = (pad_size - bh) // 2
            square[off_y:off_y+bh, off_x:off_x+bw] = cropped
            mask_64 = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
        else:
            mask_64 = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_AREA)

        mask_64 = np.where(mask_64 > 127, 255, 0).astype(np.uint8)

        fill = float((mask_64 > 127).mean())
        hand_present = (MIN_FILL <= fill <= MAX_FILL) and separation >= MIN_SEPARATION

        prediction, confidence, source = None, 0.0, None
        if hand_present:
            custom, _ = self._match_custom(mask_64)
            if custom is not None:
                prediction, confidence, source = custom, 1.0, 'custom'
                self._history.clear()
            else:
                letter, letter_conf = self._classify(mask_64)
                prediction, confidence = self._stable_letter(letter, letter_conf)
                source = 'model'
        else:
            self._history.clear()

        vis = frame.copy()
        box_color = (0, 255, 0) if hand_present else (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

        return {
            "camera_frame": np.ascontiguousarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)),
            "mask_frame": np.ascontiguousarray(mask),
            "mask_64": mask_64,
            "prediction": prediction,
            "confidence": confidence,
            "hand_present": hand_present,
            "fill": fill,
            "separation": separation,
            "source": source,
        }
