from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import argparse
import pickle

import numpy as np
import PIL.Image as Image
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import scripts.retrain as retrain
from scripts.count_ops import load_graph


def evaluate_graph(graph_file_name, image_dir, n_classes=2):
    with load_graph(graph_file_name).as_default() as graph:
        ground_truth_input = tf.placeholder(
            tf.float32, [None, n_classes], name='GroundTruthInput')

        image_buffer_input = graph.get_tensor_by_name('input:0')
        final_tensor = graph.get_tensor_by_name('final_result:0')
        accuracy, _ = retrain.add_evaluation_step(
            final_tensor, ground_truth_input)

        logits = graph.get_tensor_by_name("final_training_ops/Wx_plus_b/add:0")
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input,
            logits=logits))

    # image_dir = 'tf_files/breast-cancer'
    # testing_percentage = 10
    # validation_percentage = 10
    # validation_batch_size = -1
    category = 'testing'

    # image_lists = retrain.create_image_lists(
    #     image_dir, testing_percentage,
    #     validation_percentage)
    with open(os.path.join(image_dir, 'image_lists.pickle'), 'rb') as f:
        image_lists = pickle.load(f)
    class_count = len(image_lists.keys())

    ground_truths = []
    filenames = []

    for label_index, label_name in enumerate(image_lists.keys()):
        for image_index, image_name in enumerate(image_lists[label_name][category]):
            image_name = retrain.get_image_path(
                image_lists, label_name, image_index, image_dir, category)
            ground_truth = np.zeros([1, class_count], dtype=np.float32)
            ground_truth[0, label_index] = 1.0
            ground_truths.append(ground_truth)
            filenames.append(image_name)

    accuracies = []
    xents = []
    with tf.Session(graph=graph) as sess:
        for filename, ground_truth in zip(filenames, ground_truths):
            image = Image.open(filename).resize((224, 224), Image.ANTIALIAS)
            image = np.array(image, dtype=np.float32)[None, ...]
            image = (image - 128) / 128.0

            feed_dict = {
                image_buffer_input: image,
                ground_truth_input: ground_truth}

            eval_accuracy, eval_xent = sess.run([accuracy, xent], feed_dict)

            accuracies.append(eval_accuracy)
            xents.append(eval_xent)

    return np.mean(accuracies), np.mean(xents)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--graph_file_name',
        type=str,
        default='/data/tf/tf_files/retrained_graph.pb',
        help='Path to graph.'
    )
    parser.add_argument(
        '--n_classes',
        type=int,
        default=2,
        help='number of classes.'
    )
    args = parser.parse_args()
    accuracy, xent = evaluate_graph(args.graph_file_name, args.image_dir)
    print('Accuracy: %g' % accuracy)
    print('Cross Entropy: %g' % xent)

'''
python -m scripts.evaluate \
    --image_dir=/data/combined_data/ \
    --graph_file_name=/data/tf/tf_files/retrained_graph.pb
'''
