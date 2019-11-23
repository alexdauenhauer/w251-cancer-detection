# %%
import argparse
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow.contrib.tensorrt

import scripts.retrain as retrain
from scripts.count_ops import load_graph
from scripts.label_image import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# %%


def labelImages(img_dir, model_file, label_file):
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"
    graph = load_graph(model_file)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results_dict = defaultdict(list)
        for dirname, subdirs, file in os.walk(img_dir):
            if not file.endswith('jpg'):
                continue
            file_name = os.path.join(dirname, file)
            results_dict['file_name'].append(file_name)
            results_dict['true_label'].append(dirname.split('/')[-1])
            t = read_tensor_from_image_file(file_name,
                                            input_height=input_height,
                                            input_width=input_width,
                                            input_mean=input_mean,
                                            input_std=input_std)
            start = time.time()
            results = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: t})
            end = time.time()
            results_dict['eval_time'].append(end - start)
            print('\nEvaluation time (1-image): {:.3f}s\n'.format(end - start))
            results = np.squeeze(results)
            top_k = results.argsort()[-5:][::-1]
            labels = load_labels(label_file)
            template = "{} (score={:0.5f})"
            for i in top_k:
                results_dict[labels[i]].append(results[i])
                print(template.format(labels[i], results[i]))
    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="path to top-level images folder")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    args = parser.parse_args()
    results_dict = labelImages(args.image_dir, args.graph, args.labels)
    with open('../results_dict.pickle', 'wb') as f:
        pickle.dump(results_dict, f, protocol=-1)
