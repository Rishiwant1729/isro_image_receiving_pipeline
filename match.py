import cv2
import numpy as np
import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff")
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test"
DETECTION_THRESHOLD = 8

class RobustTerrainMatcher:
    """
    An expert-level classical computer vision pipeline for terrain matching.
    Designed to handle partial crops, rotation, and scale variations.
    """
    def __init__(self, confidence_threshold=15):
        self.confidence_threshold = confidence_threshold
        
        # Expert ORB parameters:
        # nfeatures: 5000 allows for dense terrain textures
        # scaleFactor: 1.2 provides good scale invariance with enough levels
        # edgeThreshold/patchSize: 31/31 is standard, but keeping it robust for corners
        self.orb = cv2.ORB_create(
            nfeatures=5000,
            scaleFactor=1.2,
            nlevels=12,
            edgeThreshold=15, # Lower threshold to catch features near crop edges
            patchSize=31,
            fastThreshold=20
        )
        
        # BFMatcher with Hamming distance (optimal for ORB descriptors)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # CLAHE for contrast enhancement (handles lighting variations)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def preprocess(self, image_path):
        """Loads image, converts to grayscale, and applies CLAHE."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE enhancement
        enhanced = self.clahe.apply(gray)
        
        return enhanced, img

    def match(self, ref_gray, test_gray):
        """Detects features and performs geometric verification using RANSAC."""
        # Feature detection and description
        kp1, des1 = self.orb.detectAndCompute(ref_gray, None)
        kp2, des2 = self.orb.detectAndCompute(test_gray, None)
        
        if des1 is None or des2 is None:
            return 0, 0, 0.0, (kp1 or [], kp2 or [], []), None
        
        # KNN matching for Lowe's Ratio Test
        matches = self.bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's Ratio Test (filters out ambiguous matches)
        good_matches = []
        for m_list in matches:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
        # Geometric Verification via RANSAC Homography
        inliers = 0
        homography = None
        final_matches = []
        
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find the transform from test (crop) to reference
            # Note: We swap src/dst to find where the test image lies in the reference
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            if mask is not None:
                inliers = int(np.sum(mask))
                homography = M
                # Keep only inlier matches for visualization
                final_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
            
        # Confidence score is purely based on geometric inliers
        # In terrain matching, 15+ inliers is usually a very strong match
        confidence = float(inliers)
        
        return len(good_matches), inliers, confidence, (kp1, kp2, final_matches), homography

    def evaluate(self, ref_paths, test_path):
        """Returns the best reference match for one test image."""
        print(f"Loading test image: {test_path}")
        test_gray, test_img = self.preprocess(test_path)
        test_h, test_w = test_gray.shape
        
        results = []
        for ref_path in ref_paths:
            print(f"Checking reference: {ref_path}...")
            ref_gray, ref_img = self.preprocess(ref_path)
            
            good, inliers, conf, match_data, H = self.match(ref_gray, test_gray)
            
            results.append({
                'name': ref_path,
                'good_matches': good,
                'inliers': inliers,
                'confidence': conf,
                'match_data': match_data,
                'homography': H,
                'ref_img': ref_img,
                'test_img': test_img
            })
            
            print(f"  - Total Matches: {good}")
            print(f"  - RANSAC Inliers: {inliers}")
            print(f"  - Final Score: {conf:.2f}")
            print("-" * 30)

        best = max(results, key=lambda x: x['confidence'])

        return best, test_h, test_w

    def run(self, ref_paths, test_path, output_path="match_output.jpg"):
        """Executes the pipeline across multiple reference images."""
        best, test_h, test_w = self.evaluate(ref_paths, test_path)

        print("\n" + "="*40)
        print(f"BEST MATCH: {best['name']}")
        print(f"Confidence score: {best['confidence']:.2f}")
        
        # Multi-level decision logic based on suggested thresholds
        if best['confidence'] >= 15:
            decision = "MATCH (High Confidence)"
        elif best['confidence'] >= 8:
            decision = "MATCH (Likely / Ambiguous)"
        else:
            decision = "NO MATCH (Insufficient Features)"
            
        print(f"Result: {decision}")
        print("="*40)

        # Final Visualization
        kp1, kp2, matches = best['match_data']
        ref_vis = best['ref_img'].copy()
        test_vis = best['test_img'].copy()
        
        # If a match was found, draw a bounding box around the detected area in reference
        if best['homography'] is not None and best['inliers'] >= 4:
            # Corners of the test image
            pts = np.float32([[0, 0], [0, test_h-1], [test_w-1, test_h-1], [test_w-1, 0]]).reshape(-1, 1, 2)
            # Project corners into reference image space
            dst = cv2.perspectiveTransform(pts, best['homography'])
            # Draw bounding box (Green)
            ref_vis = cv2.polylines(ref_vis, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # Create side-by-side match visualization
        vis_img = cv2.drawMatches(
            ref_vis, kp1, test_vis, kp2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0) # Inliers in Green
        )
        
        # Add labels
        cv2.putText(vis_img, f"Best Match: {best['name']} ({decision})", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imwrite(str(output_path), vis_img)
        print(f"\nResult visualization saved to: {output_path}")

        return best, decision


def ensure_folder(path):
    Path(path).mkdir(exist_ok=True)


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def reference_paths():
    ref_files = []
    for i in range(1, 4):
        for ext in IMAGE_EXTENSIONS:
            path = BASE_DIR / f"ref{i}{ext}"
            if path.exists():
                ref_files.append(path)
                break

    return ref_files


def reference_detection_folder(reference_path):
    return BASE_DIR / f"{Path(reference_path).stem}_detections"


def unique_destination(folder, source_path):
    destination = folder / source_path.name
    if not destination.exists():
        return destination

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder / f"{source_path.stem}_{timestamp}{source_path.suffix}"


def wait_until_file_is_ready(path, checks=3, delay=0.4):
    last_size = -1
    stable_checks = 0

    while stable_checks < checks:
        current_size = path.stat().st_size
        if current_size == last_size and current_size > 0:
            stable_checks += 1
        else:
            stable_checks = 0
            last_size = current_size
        time.sleep(delay)


def file_signature(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def process_test_image(path, matcher, refs, threshold):
    wait_until_file_is_ready(path)
    best, _, _ = matcher.evaluate([str(ref) for ref in refs], str(path))
    score = best["confidence"]
    detected = score >= threshold

    if detected:
        output_folder = reference_detection_folder(best["name"])
        ensure_folder(output_folder)
        destination = unique_destination(output_folder, path)
        shutil.copy2(path, destination)
        status = f"DETECTED -> {output_folder.name}/{destination.name}"
    else:
        status = "NO_MATCH"

    print(
        f"{status}: {path.name} | "
        f"best_reference={Path(best['name']).name} | "
        f"score={score:.0f} | "
        f"good_matches={best['good_matches']} | "
        f"inliers={best['inliers']}"
    )

    return detected


def run_test_folder(threshold=DETECTION_THRESHOLD, poll_seconds=1.0, once=False):
    refs = reference_paths()
    if len(refs) < 3:
        print("Error: Required reference images not found. Need ref1, ref2, and ref3.")
        return

    ensure_folder(TEST_DIR)
    for ref in refs:
        ensure_folder(reference_detection_folder(ref))

    matcher = RobustTerrainMatcher(confidence_threshold=threshold)
    processed = {}

    print(f"Watching test images in: {TEST_DIR}")
    print(f"Detection threshold: {threshold} RANSAC inliers")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            for path in sorted(TEST_DIR.iterdir()):
                if not is_image_file(path):
                    continue

                signature = file_signature(path)
                if processed.get(path.name) == signature:
                    continue

                try:
                    process_test_image(path, matcher, refs, threshold)
                except FileNotFoundError as error:
                    print(error)
                processed[path.name] = file_signature(path)

            if once:
                break
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nTest folder matcher stopped.")

def main():
    parser = argparse.ArgumentParser(
        description="Match test terrain images against ref1, ref2, and ref3."
    )
    parser.add_argument(
        "test_image",
        nargs="?",
        help="Optional single test image. If omitted, test.png/jpg/jpeg is used.",
    )
    parser.add_argument(
        "--watch-test-folder",
        action="store_true",
        help="Continuously process images received in the test folder.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process current test folder images once, then exit.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DETECTION_THRESHOLD,
        help="Minimum RANSAC inlier score needed to save into a detection folder.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="How often to check the test folder in watch mode.",
    )
    args = parser.parse_args()

    if args.watch_test_folder or args.once:
        run_test_folder(
            threshold=args.threshold,
            poll_seconds=args.poll_seconds,
            once=args.once,
        )
        return

    ref_files = [str(path) for path in reference_paths()]

    test_file = args.test_image
    if not test_file:
        for ext in IMAGE_EXTENSIONS:
            path = BASE_DIR / f"test{ext}"
            if path.exists():
                test_file = str(path)
                break

    if not test_file or len(ref_files) < 3:
        print("Error: Required images not found. Need ref1, ref2, ref3 and test.")
        print("Usage: python3 match.py [path_to_test_image]")
        return

    # Initialize matcher with a recommended threshold of 15-20 inliers for terrain
    matcher = RobustTerrainMatcher(confidence_threshold=args.threshold)
    matcher.run(ref_files, test_file)

if __name__ == "__main__":
    main()
