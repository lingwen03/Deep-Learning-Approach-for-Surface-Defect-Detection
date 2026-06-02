"""
Inference script for defect detection on custom images.
Usage: python inference.py --image_dir <path_to_images>
"""
import argparse
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from config import IMAGE_SIZE
from utils import concatImage

# Suppress TF warnings
tf.logging.set_verbosity(tf.logging.ERROR)


def preprocess_image(image_path):
    """Load and preprocess image to match training format."""
    img = cv2.imread(image_path, 0)  # grayscale
    if img is None:
        print("ERROR: Cannot read image {}".format(image_path))
        return None, None
    original_shape = img.shape
    img_resized = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img_input = np.array(img_resized[:, :, np.newaxis], dtype=np.float32)
    img_input = np.expand_dims(img_input, axis=0)  # add batch dimension
    return img_input, original_shape


def run_inference(image_dir, checkpoint_dir, output_dir):
    """Run defect detection on all images in a directory."""

    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ])

    if not image_files:
        print("No image files found in {}".format(image_dir))
        return

    print("Found {} images to process".format(len(image_files)))

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build the model graph
    tf.reset_default_graph()

    # Rebuild the same model architecture
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

    # Placeholder
    Image = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1), name='Image')
    is_training = tf.constant(False)
    features, logits_pixel, mask = SegmentNet(Image, 'segment', is_training)
    logits_class, output_class = DecisionNet(features, mask, 'decision', is_training)
    prob = tf.nn.softmax(logits_class)

    # Load checkpoint
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore from checkpoint
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt_state and ckpt_state.model_checkpoint_path:
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            print("Restored from {}".format(ckpt_state.model_checkpoint_path))
        else:
            print("ERROR: No checkpoint found in {}".format(checkpoint_dir))
            return

        # Process each image
        results = []
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(image_dir, image_file)
            print("\n[{}/{}] Processing: {}".format(idx+1, len(image_files), image_file))

            img_input, original_shape = preprocess_image(image_path)
            if img_input is None:
                continue

            # Run inference
            mask_out, class_out, prob_out = sess.run(
                [mask, output_class, prob],
                feed_dict={Image: img_input}
            )

            # Get results
            pred_class = class_out[0]  # 0=OK, 1=NG
            if pred_class == 1:
                class_label = "NG (Defect)"
            else:
                class_label = "OK (No Defect)"

            print("  Result: {}".format(class_label))
            print("  Confidence: OK={:.4f}, NG={:.4f}".format(prob_out[0][0], prob_out[0][1]))

            # Generate visualization
            mask_img = (np.array(mask_out[0]).squeeze(2) * 255).astype(np.uint8)
            original_img = np.array(img_input[0]).squeeze(2).astype(np.uint8)

            # Save visualization
            viz = concatImage([original_img, mask_img])
            basename = os.path.splitext(image_file)[0]
            viz_path = os.path.join(output_dir, "{}_result.png".format(basename))
            viz.save(viz_path)
            print("  Visualization saved to: {}".format(viz_path))

            # Also save the mask separately (resized back to original dimensions)
            mask_save = cv2.resize(mask_img,
                                   (original_shape[1], original_shape[0]))
            mask_path = os.path.join(output_dir, "{}_mask.png".format(basename))
            cv2.imwrite(mask_path, mask_save)

            results.append({
                'file': image_file,
                'class': class_label,
                'confidence_ok': float(prob_out[0][0]),
                'confidence_ng': float(prob_out[0][1]),
            })

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        ng_count = sum(1 for r in results if 'NG' in r['class'])
        ok_count = sum(1 for r in results if 'OK' in r['class'])
        print("Total images: {}".format(len(results)))
        print("Defects detected (NG): {}".format(ng_count))
        print("No defects (OK): {}".format(ok_count))
        print("="*60)
        print("\nDetailed results:")
        for r in results:
            print("  {}: {} (OK={:.4f}, NG={:.4f})".format(
                r['file'], r['class'], r['confidence_ok'], r['confidence_ng']))

        print("\nAll visualizations saved to: {}".format(output_dir))

    return results


def main():
    parser = argparse.ArgumentParser(description='Defect detection inference')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images to test')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory with trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save results')
    args = parser.parse_args()

    run_inference(args.image_dir, args.checkpoint_dir, args.output_dir)


if __name__ == '__main__':
    main()
