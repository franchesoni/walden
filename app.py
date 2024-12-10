import h5py
import csv
import heapq
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route
from starlette.staticfiles import StaticFiles


class DataManager:
    def __init__(self, root: Path):
        self.root = root
        self.labeled_indices = set()

        # Load features and bounding boxes
        feats_list = []
        bboxes_list = []
        for i in [1, 2, 3, 5, 6]:
            imgname = f"img{i}"
            feat_file = Path(root / "out" / imgname / "masks" / "features.h5")
            bbox_file = Path(root / "out" / imgname / "masks" / "global_bboxes.txt")

            feat = h5py.File(feat_file, "r")["dataset"][:]
            bboxs = self._load_bounding_boxes_csv(bbox_file, i)

            assert len(feat) == len(bboxs)
            feats_list.append(feat)
            bboxes_list.append(bboxs)

        self.feats = np.concatenate(feats_list, axis=0)  # (N, D)
        self.bboxes = np.concatenate(bboxes_list, axis=0)  # (N, 5)
        self.N = self.feats.shape[0]
        self.unlabeled_indices = set(range(self.N))

    def _load_bounding_boxes_csv(self, csv_file, imgnumber):
        bboxes = []
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                bboxes.append([imgnumber] + [int(float(x)) for x in row])
        return np.array(bboxes)

    def get_features(self, idx):
        return self.feats[idx]

    def get_image(self, idx, center_crop=False):
        # Fetch the bounding box and load the corresponding image
        bbox = self.bboxes[idx]
        img = self._load_image_with_bbox(bbox, center_crop=center_crop)
        return img

    def _load_image_with_bbox(self, bbox, center_crop=False):
        """
        Load a 3x3 composite of tiles around the given bounding box and optionally
        return a 512x512 crop centered on the bounding box.

        Args:
            bbox (list or tuple): [imgnumber, row, col, height, width]
                - imgnumber: The image number (e.g., 5 for 'img5')
                - row, col: Top-left coordinates of the bounding box in global coordinates
                - height, width: Size of the bounding box
            center_crop (bool): If True, return a 512x512 crop centered on the bounding box.
                                If False, return the full 768x768 (3x3 tiles) composite.

        Returns:
            PIL.Image: The assembled image with bounding box drawn.
        """
        # Unpack the bounding box
        imgnumber, bbox_row, bbox_col, bbox_height, bbox_width = bbox

        TILE_SIZE = 256
        COMPOSITE_SIZE = TILE_SIZE * 3  # 768x768
        CROP_SIZE = 512
        HALF_CROP = CROP_SIZE // 2

        # Directory with tiles
        imgname = f"img{imgnumber}"
        tiles_dir = Path(self.root) / f"dataset/{imgname}/tiles"

        # Determine the tile grid start
        # We find the tile that contains the top-left corner of the bbox
        tile_row_start = (bbox_row // TILE_SIZE) * TILE_SIZE
        tile_col_start = (bbox_col // TILE_SIZE) * TILE_SIZE

        # Create a composite image (3x3 tiles)
        composite_image = Image.new(
            "RGB", (COMPOSITE_SIZE, COMPOSITE_SIZE), (255, 255, 255)
        )

        # Load surrounding 3x3 tiles
        for i, drow in enumerate([-TILE_SIZE, 0, TILE_SIZE]):
            for j, dcol in enumerate([-TILE_SIZE, 0, TILE_SIZE]):
                tile_row = tile_row_start + drow
                tile_col = tile_col_start + dcol
                tile_name = f"tile_{int(tile_row)}_{int(tile_col)}.jpeg"
                tile_path = tiles_dir / tile_name

                if tile_path.exists():
                    try:
                        tile_image = Image.open(tile_path)
                        if tile_image.mode != "RGB":
                            tile_image = tile_image.convert("RGB")
                        composite_image.paste(
                            tile_image, (j * TILE_SIZE, i * TILE_SIZE)
                        )
                    except Exception as e:
                        # If tile loading fails, use a placeholder
                        placeholder = Image.new(
                            "RGB", (TILE_SIZE, TILE_SIZE), (200, 200, 200)
                        )
                        draw_placeholder = ImageDraw.Draw(placeholder)
                        draw_placeholder.line(
                            (0, 0) + placeholder.size, fill=(150, 150, 150), width=3
                        )
                        draw_placeholder.line(
                            (0, placeholder.size[1], placeholder.size[0], 0),
                            fill=(150, 150, 150),
                            width=3,
                        )
                        composite_image.paste(
                            placeholder, (j * TILE_SIZE, i * TILE_SIZE)
                        )
                else:
                    # Missing tile placeholder
                    placeholder = Image.new(
                        "RGB", (TILE_SIZE, TILE_SIZE), (200, 200, 200)
                    )
                    draw_placeholder = ImageDraw.Draw(placeholder)
                    draw_placeholder.line(
                        (0, 0) + placeholder.size, fill=(150, 150, 150), width=3
                    )
                    draw_placeholder.line(
                        (0, placeholder.size[1], placeholder.size[0], 0),
                        fill=(150, 150, 150),
                        width=3,
                    )
                    composite_image.paste(placeholder, (j * TILE_SIZE, i * TILE_SIZE))

        # Draw the bounding box on the composite image
        # Calculate the bounding box coordinates relative to the composite image
        # The composite image's center tile corresponds to (tile_row_start, tile_col_start) in global coords
        # Top-left tile in composite is at (tile_row_start - TILE_SIZE, tile_col_start - TILE_SIZE)
        composite_top_row = tile_row_start - TILE_SIZE
        composite_left_col = tile_col_start - TILE_SIZE

        bbox_row_rel = bbox_row - composite_top_row
        bbox_col_rel = bbox_col - composite_left_col
        bbox_bottom_rel = bbox_row_rel + bbox_height
        bbox_right_rel = bbox_col_rel + bbox_width

        draw = ImageDraw.Draw(composite_image)
        draw.rectangle(
            [bbox_col_rel, bbox_row_rel, bbox_right_rel, bbox_bottom_rel],
            outline="green",
            width=3,
        )

        if center_crop:
            # We want to produce a 512x512 crop centered on the bbox center
            bbox_center_row = bbox_row_rel + bbox_height / 2
            bbox_center_col = bbox_col_rel + bbox_width / 2

            # Center the BBox in the crop
            # The BBox center should map to the center of the crop (256, 256)
            left = int(bbox_center_col - HALF_CROP)
            upper = int(bbox_center_row - HALF_CROP)
            right = left + CROP_SIZE
            lower = upper + CROP_SIZE

            # Ensure we don't go outside the composite image boundaries
            if left < 0:
                right -= left
                left = 0
            if upper < 0:
                lower -= upper
                upper = 0
            if right > COMPOSITE_SIZE:
                left -= right - COMPOSITE_SIZE
                right = COMPOSITE_SIZE
            if lower > COMPOSITE_SIZE:
                upper -= lower - COMPOSITE_SIZE
                lower = COMPOSITE_SIZE

            # Crop the image
            cropped_image = composite_image.crop((left, upper, right, lower))

            # If needed, pad to ensure exactly 512x512
            w, h = cropped_image.size
            if w < CROP_SIZE or h < CROP_SIZE:
                padded = Image.new("RGB", (CROP_SIZE, CROP_SIZE), (255, 255, 255))
                padded.paste(
                    cropped_image, ((CROP_SIZE - w) // 2, (CROP_SIZE - h) // 2)
                )
                cropped_image = padded

            return cropped_image
        else:
            return composite_image

    def mark_labeled(self, idx):
        self.labeled_indices.add(idx)
        self.unlabeled_indices.remove(idx)

    def unmark_labeled(self, idx):
        # Used during revert
        self.labeled_indices.discard(idx)
        self.unlabeled_indices.add(idx)

    def get_unlabeled_indices(self):
        return list(self.unlabeled_indices)


class IncrementalModel:
    def __init__(self, n_features):
        self.clf = SGDClassifier(
            loss="log_loss", warm_start=True, class_weight="balanced"
        )
        self.scaler = StandardScaler()
        self._initialized = False
        self.n_features = n_features

    def partial_fit(self, X, y):
        X = np.atleast_2d(X)
        y = np.asarray(y)
        if not self._initialized:
            X = self.scaler.fit_transform(X)
            self.clf.partial_fit(X, y, classes=[0, 1])
            self._initialized = True
        else:
            self.scaler = self.scaler.partial_fit(X)
            X = self.scaler.transform(X)
            self.clf.partial_fit(X, y)
        self.debug_probabilities(X)

    def predict_proba(self, X):
        if not self._initialized:
            return np.ones((len(X), 2)) * 0.5
        X = self.scaler.transform(X)
        scores = self.clf.decision_function(X)
        probs = 1.0 / (1.0 + np.exp(-scores))
        return np.vstack([1 - probs, probs]).T

    def reset(self):
        # Reset model
        self.clf = SGDClassifier(loss="log_loss", warm_start=True)
        self._initialized = False

    def debug_probabilities(self, X):
        X_scaled = self.scaler.transform(X)
        probs = self.clf.predict_proba(X_scaled)
        print("Probabilities range:", np.min(probs[:, 1]), "-", np.max(probs[:, 1]))


class Orchestrator:
    def __init__(self, root_dir, annotation_file):
        self.data_mgr = DataManager(root_dir)
        self.model = IncrementalModel(n_features=self.data_mgr.feats.shape[1])
        self.annotation_file = annotation_file
        self.annotations = []  # store (idx, label)

        self.current_sample = None
        self.candidates = []
        self.previous_samples = []  # Stack for backtracking
        self.batch_size = 10000

        self.positive_count = 0  # Counter for positive annotations
        self.negative_count = 0  # Counter for negative annotations

        self.warmup_samples = 100
        self.warmup_idx = np.random.choice(
            self.data_mgr.N, self.warmup_samples, replace=False
        ).tolist()
        self.current_sample = self.warmup_idx.pop() if self.warmup_idx else None
        self.next_sample = None  # For prefetching

        # If annotation file exists, load it and retrain to restore state
        if self.annotation_file.exists():
            self._load_annotations_and_retrain()

        self._ensure_next_sample()

    def _load_annotations_and_retrain(self):
        # Load existing annotations from file
        loaded_annotations = []
        with open(self.annotation_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                idx, label = int(row[0]), int(row[1])
                loaded_annotations.append((idx, label))
        # Retrain model
        self.retrain(loaded_annotations)

    def retrain(self, annotations):
        # Reset model and data sets
        self.model.reset()
        self.data_mgr.labeled_indices.clear()
        self.data_mgr.unlabeled_indices = set(range(self.data_mgr.N))
        for idx, label in annotations:
            X = self.data_mgr.get_features(idx)
            self.model.partial_fit(X[np.newaxis, :], [label])
            self.data_mgr.mark_labeled(idx)
        self.annotations = annotations.copy()
        # Clear candidates so they are re-generated on next request
        self.candidates.clear()

    def get_current_sample_index(self):
        return self.current_sample

    def get_next_sample_index(self):
        # Return next sample for prefetching if available
        return self.next_sample

    def handle_annotation(self, idx, label):
        # Save annotation
        with open(self.annotation_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([idx, label])

        # Update counters
        if label == 1:
            self.positive_count += 1
        elif label == 0:
            self.negative_count += 1

        # Update model
        X = self.data_mgr.get_features(idx)
        self.model.partial_fit(X[np.newaxis, :], [label])
        self.data_mgr.mark_labeled(idx)
        self.annotations.append((idx, label))

        # Push current sample onto the previous_samples stack
        self.previous_samples.append(self.current_sample)

        # Move on to next sample
        self._pick_next_sample()

    def _pick_next_sample(self):
        if self.warmup_idx:
            self.current_sample = self.warmup_idx.pop()
        else:
            if not self.candidates:
                self.refill_candidates()
            if self.candidates:
                _, self.current_sample = heapq.heappop(self.candidates)
            else:
                self.current_sample = None

        self._ensure_next_sample()

    def _ensure_next_sample(self):
        # Prefetch next sample from candidates if available
        if self.warmup_idx:
            # Next sample is known
            if self.warmup_idx:
                self.next_sample = (
                    self.warmup_idx[-1] if len(self.warmup_idx) > 0 else None
                )
            else:
                self.next_sample = None
        else:
            # If we have candidates, look at the next one in the heap
            if len(self.candidates) < 2:
                self.refill_candidates()
            if len(self.candidates) > 0:
                # Peek next candidate without popping it
                self.next_sample = self.candidates[0][1]
            else:
                self.next_sample = None

    def refill_candidates(self):
        unlabeled = self.data_mgr.get_unlabeled_indices()
        if not unlabeled:
            return
        batch = np.random.choice(
            unlabeled, size=min(self.batch_size, len(unlabeled)), replace=False
        )
        X = np.array([self.data_mgr.get_features(i) for i in batch])
        probs = self.model.predict_proba(X)
        # uncertainty = distance from 0.5
        uncertainty = np.abs(probs[:, 1] - 0.5)
        for u, idx in zip(uncertainty, batch):
            heapq.heappush(self.candidates, (u, idx))

    def revert_last_annotation(self):
        if not self.annotations or not self.previous_samples:
            return

        # Pop last annotation and previous sample
        idx, label = self.annotations.pop()
        self.current_sample = self.previous_samples.pop()

        # Remove annotation from file: rewrite file
        with open(self.annotation_file, "w", newline="") as f:
            writer = csv.writer(f)
            for aidx, albl in self.annotations:
                writer.writerow([aidx, albl])

        # Unmark the last labeled sample
        self.data_mgr.unmark_labeled(idx)

        # Retrain model from scratch
        self.retrain(self.annotations)

    def get_probability_for(self, idx):
        # Return model probability for a single sample
        X = self.data_mgr.get_features(idx)[np.newaxis, :]
        probs = self.model.predict_proba(X)
        return probs[0, 1]


app_root = Path("")
orchestrator = Orchestrator(root_dir=app_root, annotation_file=Path("annotations.csv"))


async def homepage(request):
    idx = orchestrator.get_current_sample_index()
    if idx is None:
        return HTMLResponse("<h1>No more samples!</h1>")

    img = orchestrator.data_mgr.get_image(idx, center_crop=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    img_base64 = "data:image/jpeg;base64," + __import__("base64").b64encode(
        img_bytes
    ).decode("utf-8")

    # Probability
    prob = orchestrator.get_probability_for(idx)
    prob_html = f"<p>Predicted Probability (Positive): {prob:.3f}</p>"

    # Annotation Counters
    positive_count = orchestrator.positive_count
    negative_count = orchestrator.negative_count
    counter_html = f"""
    <p>Annotations: Positive = {positive_count}, Negative = {negative_count}</p>
    """

    # Prefetch next image if available
    next_idx = orchestrator.get_next_sample_index()
    prefetch_img_html = ""
    if next_idx is not None:
        # Prefetch route
        prefetch_img_html = f'<img id="prefetch" src="/prefetch?idx={next_idx}" style="display:none;" />'

    # JavaScript for key handling: 'a' or Left Arrow = Negative, 'd' or Right Arrow = Positive, Backspace = Revert
    script = """
    <script>
    document.addEventListener('keydown', function(event) {
        if (event.key === 'a' || event.key === 'ArrowLeft') {
            document.getElementById('negButton').click();
        } else if (event.key === 'd' || event.key === 'ArrowRight') {
            document.getElementById('posButton').click();
        } else if (event.key === 'Backspace') {
            event.preventDefault();
            window.location.href = '/revert';
        }
    });
    </script>
    """

    html = f"""
    <html>
    <body>
    <h1>Sample {idx}</h1>
    {prob_html}
    {counter_html}
    <img src="{img_base64}" width="512" height="512" />
    {prefetch_img_html}
    <form method="POST" action="/label" id="labelForm">
        <input type="hidden" name="idx" value="{idx}" />
        <button name="label" value="0" id="negButton">Negative (a/left)</button>
        <button name="label" value="1" id="posButton">Positive (d/right)</button>
    </form>
    {script}
    </body>
    </html>
    """
    return HTMLResponse(html)


async def prefetch_handler(request):
    idx = int(request.query_params["idx"])
    img = orchestrator.data_mgr.get_image(idx, center_crop=True)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return Response(buf.getvalue(), media_type="image/jpeg")


async def label_handler(request):
    form = await request.form()
    idx = int(form["idx"])
    label = int(form["label"])
    orchestrator.handle_annotation(idx, label)
    # Redirect immediately to homepage (no confirmation)
    return RedirectResponse("/", status_code=303)


async def revert_handler(request):
    orchestrator.revert_last_annotation()
    return RedirectResponse("/", status_code=303)


app = Starlette(
    debug=True,
    routes=[
        Route("/", homepage),
        Route("/label", label_handler, methods=["POST"]),
        Route("/prefetch", prefetch_handler),
        Route("/revert", revert_handler),
    ],
)
