"""
Optimized inference script for defect detection on custom images.

Key improvements over inference.py:
  1. Letterbox resize: preserves aspect ratio instead of direct stretch
  2. CLAHE preprocessing: enhances local contrast for subtle defects
  3. Mask post-processing: morphological cleaning + connected component filter
  4. Overlapping-crop ensemble: runs inference on multiple crops for large images
  5. Multi-threshold analysis: finds best mask threshold per image

Usage: python inference_v2.py --image_dir <path_to_images>
"""
import argparse
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from config import IMAGE_SIZE
from utils import concatImage

tf.logging.set_verbosity(tf.logging.ERROR)

# Target network input size (height, width)
TARGET_H = IMAGE_SIZE[0]   # 1280
TARGET_W = IMAGE_SIZE[1]   # 512

# ---------------------------------------------------------------------------
#  Preprocessing
# ---------------------------------------------------------------------------

def letterbox_resize(img, target_h=TARGET_H, target_w=TARGET_W):
    """Resize image to fit target while preserving aspect ratio, pad with replicate border."""
    h, w = img.shape
    scale = min(float(target_h) / h, float(target_w) / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center-pad
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        borderType=cv2.BORDER_REPLICATE
    )
    # Return padding info for mask un-padding later
    return img_padded, (top, bottom, left, right, scale)


def apply_clahe(img, clip_limit=2.0, tile_size=8):
    """Apply CLAHE to enhance local contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(img)


def preprocess_image_v2(image_path, use_clahe=True):
    """Enhanced preprocessing with letterbox + optional CLAHE."""
    img = cv2.imread(image_path, 0)
    if img is None:
        print("ERROR: Cannot read image {}".format(image_path))
        return None, None, None

    original_shape = img.shape

    # Step 1: CLAHE contrast enhancement
    if use_clahe:
        img = apply_clahe(img)

    # Step 2: Letterbox resize (preserve aspect ratio)
    img_resized, pad_info = letterbox_resize(img)
    img_input = np.array(img_resized[:, :, np.newaxis], dtype=np.float32)
    img_input = np.expand_dims(img_input, axis=0)
    return img_input, original_shape, pad_info


# ---------------------------------------------------------------------------
#  Mask post-processing
# ---------------------------------------------------------------------------

def postprocess_mask(mask_raw, min_area_ratio=0.0001, morph_close_size=7, morph_open_size=5):
    """
    Clean up raw mask with morphological operations and connected component filtering.

    Args:
        mask_raw: 2D numpy array (0-255), raw sigmoid output
        min_area_ratio: minimum connected component area as fraction of total pixels
        morph_close_size: kernel size for morphological closing (fills gaps)
        morph_open_size: kernel size for morphological opening (removes noise)
    Returns:
        cleaned binary mask (0-255)
    """
    h, w = mask_raw.shape
    total_pixels = h * w
    min_area = int(total_pixels * min_area_ratio)

    # Adaptive threshold: use Otsu if there's enough variation
    if mask_raw.max() - mask_raw.min() > 10:
        _, binary = cv2.threshold(mask_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Fallback to fixed threshold
        _, binary = cv2.threshold(mask_raw, 30, 255, cv2.THRESH_BINARY)

    # Morphological closing: fill small gaps within defect regions
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (morph_close_size, morph_close_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Morphological opening: remove small isolated noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (morph_open_size, morph_open_size))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # Connected component filtering: keep only significant regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    cleaned = np.zeros_like(opened)
    kept_regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255
            kept_regions.append({
                'area': area,
                'centroid': (int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2),
                             int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2)),
                'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            })

    return cleaned, kept_regions


def remove_padding(mask, pad_info):
    """Crop padding regions from mask to match original letterbox image."""
    top, bottom, left, right, _ = pad_info
    h, w = mask.shape
    return mask[top:h - bottom, left:w - right]


# ---------------------------------------------------------------------------
#  Overlapping crop ensemble (for large images)
# ---------------------------------------------------------------------------

def generate_crops(img_h, img_w, crop_h=TARGET_H, crop_w=TARGET_W,
                   overlap_h=256, overlap_w=128):
    """
    Generate overlapping crop windows that cover the full image.
    Returns list of (top, left, bottom, right).
    """
    crops = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            bottom = min(y + crop_h, img_h)
            right = min(x + crop_w, img_w)
            # Adjust start to ensure full crop size where possible
            top = max(0, bottom - crop_h)
            left = max(0, right - crop_w)
            crops.append((top, left, bottom, right))
            if right >= img_w:
                break
            x = right - overlap_w
        if bottom >= img_h:
            break
        y = bottom - overlap_h
    return crops


def ensemble_crop_inference(sess, full_img, mask_tensor, image_ph, crop_overlap=True):
    """
    Run inference on overlapping crops and stitch results.
    For images that fit in the model window, run single-pass.
    For larger images, run overlapping crops and max-pool the masks.
    """
    h, w = full_img.shape

    # If image fits within model window after slight resize, do single pass
    if h <= TARGET_H * 1.3 and w <= TARGET_W * 1.3:
        img_enhanced = apply_clahe(full_img)
        img_processed, pad_info = letterbox_resize(img_enhanced)
        img_input = np.array(img_processed[:, :, np.newaxis], dtype=np.float32)
        img_input = np.expand_dims(img_input, axis=0)
        mask_out = sess.run(mask_tensor, feed_dict={image_ph: img_input})
        # Mask is 160x64 (1/8 of input). Resize to input size first.
        mask_raw = (np.array(mask_out[0]).squeeze(2) * 255).astype(np.uint8)
        mask_letterbox = cv2.resize(mask_raw, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        # Remove padding then resize to original
        mask_raw = remove_padding(mask_letterbox, pad_info)
        mask_raw = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        return mask_raw

    # For larger images: sliding window with overlap
    overlap_h = TARGET_H // 4  # 320
    overlap_w = TARGET_W // 4  # 128

    crops = generate_crops(h, w, TARGET_H, TARGET_W, overlap_h, overlap_w)
    print("  Running {} overlapping crops...".format(len(crops)))

    # Accumulator for stitched mask
    mask_accum = np.zeros((h, w), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)

    for crop_idx, (top, left, bottom, right) in enumerate(crops):
        crop = full_img[top:bottom, left:right]
        # Process crop (with CLAHE)
        crop_enhanced = apply_clahe(crop)
        crop_resized, pad_info = letterbox_resize(crop_enhanced)

        crop_input = np.array(crop_resized[:, :, np.newaxis], dtype=np.float32)
        crop_input = np.expand_dims(crop_input, axis=0)

        mask_out = sess.run(mask_tensor, feed_dict={image_ph: crop_input})
        # Mask is 160x64 (1/8 of input). Resize to input size first.
        mask_raw = (np.array(mask_out[0]).squeeze(2) * 255).astype(np.float32)
        mask_letterbox = cv2.resize(mask_raw, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        # Remove padding then resize to crop dimensions
        mask_unpadded = remove_padding(mask_letterbox, pad_info)
        mask_resized = cv2.resize(mask_unpadded, (right - left, bottom - top),
                                  interpolation=cv2.INTER_LINEAR)

        # Cosine weight window (emphasis on center)
        ch, cw = bottom - top, right - left
        wy = np.sin(np.linspace(0, np.pi, ch))[:, np.newaxis]
        wx = np.sin(np.linspace(0, np.pi, cw))[np.newaxis, :]
        weight = wy * wx

        mask_accum[top:bottom, left:right] += mask_resized * weight
        weight_accum[top:bottom, left:right] += weight

        if (crop_idx + 1) % 10 == 0:
            print("    Crop {}/{}".format(crop_idx + 1, len(crops)))

    # Normalize
    valid = weight_accum > 0
    mask_accum[valid] /= weight_accum[valid]
    return mask_accum.astype(np.uint8)


# ---------------------------------------------------------------------------
#  Main inference
# ---------------------------------------------------------------------------

def run_inference_v2(image_dir, checkpoint_dir, output_dir,
                     use_clahe=True, use_crop_ensemble=True,
                     min_area_ratio=0.0001, morph_close=7, morph_open=5):
    """Run optimized defect detection on all images in a directory."""

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    if not image_files:
        print("No image files found in {}".format(image_dir))
        return

    print("Found {} images to process".format(len(image_files)))
    print("Config: CLAHE={}, crop_ensemble={}, morph_close={}, morph_open={}, min_area_ratio={}".format(
        use_clahe, use_crop_ensemble, morph_close, morph_open, min_area_ratio))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build model graph (same as original)
    tf.reset_default_graph()
    from tensorflow.contrib import slim
    from tensorflow.python.ops import math_ops

    def SegmentNet(input_tensor, scope, is_training, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm):
                net = slim.conv2d(input_tensor, 32, [5, 5], scope='conv1')
                net = slim.conv2d(net, 32, [5, 5], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool1')
                net = slim.conv2d(net, 64, [5, 5], scope='conv3')
                net = slim.conv2d(net, 64, [5, 5], scope='conv4')
                net = slim.conv2d(net, 64, [5, 5], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool2')
                net = slim.conv2d(net, 64, [5, 5], scope='conv6')
                net = slim.conv2d(net, 64, [5, 5], scope='conv7')
                net = slim.conv2d(net, 64, [5, 5], scope='conv8')
                net = slim.conv2d(net, 64, [5, 5], scope='conv9')
                net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool3')
                net = slim.conv2d(net, 1024, [15, 15], scope='conv10')
                features = net
                net = slim.conv2d(net, 1, [1, 1], activation_fn=None, scope='conv11')
                logits_pixel = net
                net = tf.sigmoid(net, name=None)
                mask = net
        return features, logits_pixel, mask

    def DecisionNet(feature, mask, scope, is_training, num_classes=2, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                padding='SAME',
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm):
                net = tf.concat([feature, mask], axis=3)
                net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool1')
                net = slim.conv2d(net, 8, [5, 5], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool2')
                net = slim.conv2d(net, 16, [5, 5], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], [2, 2], scope='pool3')
                net = slim.conv2d(net, 32, [5, 5], scope='conv3')
                vector1 = math_ops.reduce_mean(net, [1, 2], name='pool4', keepdims=True)
                vector2 = math_ops.reduce_max(net, [1, 2], name='pool5', keepdims=True)
                vector3 = math_ops.reduce_mean(mask, [1, 2], name='pool6', keepdims=True)
                vector4 = math_ops.reduce_max(mask, [1, 2], name='pool7', keepdims=True)
                vector = tf.concat([vector1, vector2, vector3, vector4], axis=3)
                vector = tf.squeeze(vector, axis=[1, 2])
                logits = slim.fully_connected(vector, num_classes, activation_fn=None)
                output = tf.argmax(logits, axis=1)
                return logits, output

    # Placeholder (batch_size=1 for simplicity)
    Image = tf.placeholder(tf.float32, shape=(1, TARGET_H, TARGET_W, 1), name='Image')
    is_training = tf.constant(False)
    features, logits_pixel, mask = SegmentNet(Image, 'segment', is_training)
    logits_class, output_class = DecisionNet(features, mask, 'decision', is_training)
    prob = tf.nn.softmax(logits_class)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt_state and ckpt_state.model_checkpoint_path:
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            print("Restored from {}\n".format(ckpt_state.model_checkpoint_path))
        else:
            print("ERROR: No checkpoint found in {}".format(checkpoint_dir))
            return

        results = []
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            print("[{}/{}] {}".format(idx + 1, len(image_files), image_file))

            # Read original image
            original_img = cv2.imread(image_path, 0)
            if original_img is None:
                print("  ERROR: Cannot read image")
                continue
            oh, ow = original_img.shape

            # ---------------------------------------------------------------
            # Step 1: Get raw mask (with preprocessing)
            # ---------------------------------------------------------------
            if use_crop_ensemble and (oh > TARGET_H * 1.3 or ow > TARGET_W * 1.3):
                mask_raw = ensemble_crop_inference(sess, original_img, mask, Image)
            else:
                img_enhanced = apply_clahe(original_img) if use_clahe else original_img
                img_processed, pad_info = letterbox_resize(img_enhanced)
                img_input = np.array(img_processed[:, :, np.newaxis], dtype=np.float32)
                img_input = np.expand_dims(img_input, axis=0)
                mask_out = sess.run(mask, feed_dict={Image: img_input})
                # Mask is 160x64 (1/8 of input). Resize to input size first.
                mask_raw = (np.array(mask_out[0]).squeeze(2) * 255).astype(np.uint8)
                mask_letterbox = cv2.resize(mask_raw, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
                mask_raw = remove_padding(mask_letterbox, pad_info)
                mask_raw = cv2.resize(mask_raw, (ow, oh), interpolation=cv2.INTER_LINEAR)

            # ---------------------------------------------------------------
            # Step 2: Post-process mask
            # ---------------------------------------------------------------
            mask_cleaned, defect_regions = postprocess_mask(
                mask_raw,
                min_area_ratio=min_area_ratio,
                morph_close_size=morph_close,
                morph_open_size=morph_open
            )

            # ---------------------------------------------------------------
            # Step 3: Classification (use the model's DecisionNet on a
            #          properly preprocessed single pass)
            # ---------------------------------------------------------------
            img_enhanced = apply_clahe(original_img) if use_clahe else original_img
            img_processed, pad_info = letterbox_resize(img_enhanced)
            img_input = np.array(img_processed[:, :, np.newaxis], dtype=np.float32)
            img_input = np.expand_dims(img_input, axis=0)

            class_out, prob_out = sess.run(
                [output_class, prob],
                feed_dict={Image: img_input}
            )

            pred_class = class_out[0]
            class_label = "NG (Defect)" if pred_class == 1 else "OK (No Defect)"
            conf_ok = prob_out[0][0]
            conf_ng = prob_out[0][1]

            print("  Result: {}  |  OK={:.4f}  NG={:.4f}".format(class_label, conf_ok, conf_ng))

            # Compute metrics
            total_px = oh * ow
            raw_defect_px = np.sum(mask_raw > 30)
            cleaned_defect_px = np.sum(mask_cleaned > 0)
            raw_ratio = raw_defect_px / total_px * 100
            cleaned_ratio = cleaned_defect_px / total_px * 100
            num_regions = len(defect_regions)

            print("  Raw defect area: {:.2f}%  ->  Cleaned: {:.2f}%  ({} regions)".format(
                raw_ratio, cleaned_ratio, num_regions))
            for j, r in enumerate(defect_regions[:5]):  # show top 5
                print("    Region {}: area={}px  bbox={}".format(
                    j + 1, r['area'], r['bbox']))

            # ---------------------------------------------------------------
            # Step 4: Save visualizations
            # ---------------------------------------------------------------
            basename = os.path.splitext(image_file)[0]

            # 4a. Original vs raw mask vs cleaned mask (3-panel comparison)
            # All must be same size for concatImage
            comparison = concatImage([original_img, mask_raw, mask_cleaned])
            comp_path = os.path.join(output_dir, "{}_comparison.png".format(basename))
            comparison.save(comp_path)

            # 4b. Overlay: original image with defect regions highlighted in red
            overlay = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
            mask_colored = np.zeros((oh, ow, 3), dtype=np.uint8)
            mask_colored[:, :, 2] = mask_cleaned  # Red channel for defects
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            overlay_path = os.path.join(output_dir, "{}_overlay.png".format(basename))
            cv2.imwrite(overlay_path, overlay)

            # 4c. Cleaned mask only
            mask_path = os.path.join(output_dir, "{}_mask.png".format(basename))
            cv2.imwrite(mask_path, mask_cleaned)

            # 4d. CLAHE-enhanced input for reference
            enhanced_path = os.path.join(output_dir, "{}_enhanced.png".format(basename))
            cv2.imwrite(enhanced_path, img_enhanced)

            results.append({
                'file': image_file,
                'class': class_label,
                'confidence_ok': float(conf_ok),
                'confidence_ng': float(conf_ng),
                'raw_defect_pct': raw_ratio,
                'cleaned_defect_pct': cleaned_ratio,
                'regions': num_regions,
            })

            print("  Saved: {}_comparison.png, {}_overlay.png, {}_mask.png, {}_enhanced.png".format(
                basename, basename, basename, basename))

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        ng_count = sum(1 for r in results if 'NG' in r['class'])
        ok_count = sum(1 for r in results if 'OK' in r['class'])
        print("Total images: {}".format(len(results)))
        print("Defects detected (NG): {}".format(ng_count))
        print("No defects (OK): {}".format(ok_count))
        print("=" * 60)
        print("\n{:<16s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s} {:>8s}".format(
            "File", "Result", "OK_conf", "NG_conf", "Raw_Def%", "Clean_Def%", "Regions"))
        print("-" * 72)
        for r in results:
            print("{:<16s} {:>8s} {:>8.4f} {:>8.4f} {:>9.2f}% {:>9.2f}% {:>8d}".format(
                r['file'], r['class'], r['confidence_ok'], r['confidence_ng'],
                r['raw_defect_pct'], r['cleaned_defect_pct'], r['regions']))

        print("\nAll outputs saved to: {}".format(output_dir))
        print("File types: *_comparison.png (orig|raw|cleaned), *_overlay.png (defect overlay),")
        print("           *_mask.png (cleaned binary mask), *_enhanced.png (CLAHE preprocessed)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Optimized defect detection inference')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images to test')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory with trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='inference_results_v2',
                        help='Directory to save results')
    parser.add_argument('--no_clahe', action='store_true',
                        help='Disable CLAHE preprocessing')
    parser.add_argument('--no_crop_ensemble', action='store_true',
                        help='Disable overlapping-crop ensemble for large images')
    parser.add_argument('--min_area_ratio', type=float, default=0.0001,
                        help='Minimum connected component area as fraction of total (default: 0.0001)')
    parser.add_argument('--morph_close', type=int, default=7,
                        help='Morphological closing kernel size (default: 7)')
    parser.add_argument('--morph_open', type=int, default=5,
                        help='Morphological opening kernel size (default: 5)')
    args = parser.parse_args()

    run_inference_v2(
        args.image_dir, args.checkpoint_dir, args.output_dir,
        use_clahe=not args.no_clahe,
        use_crop_ensemble=not args.no_crop_ensemble,
        min_area_ratio=args.min_area_ratio,
        morph_close=args.morph_close,
        morph_open=args.morph_open
    )


if __name__ == '__main__':
    main()
