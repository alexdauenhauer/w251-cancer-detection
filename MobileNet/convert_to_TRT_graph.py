# %%
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.framework import graph_util
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--graph_file_name',
    type=str,
    default='/data/tf/tf_files/retrained_graph.pb',
    help='Path to graph.'
)
parser.add_argument(
    '--output_graph_file_name',
    type=str,
    default='/data/tf/tf_files/trt_graph.pb',
    help='Path to optimized graph.'
)
args = parser.parse_args()

# %%
with tf.gfile.GFile(args.graph_file_name, 'rb') as f:
    frozen_graph = tf.GraphDef()
    frozen_graph.ParseFromString(f.read())
# %%
trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=['final_result:0'],
    max_batch_size=32,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)
with tf.gfile.FastGFile(args.output_graph_file_name, 'wb') as f:
    f.write(trt_graph.SerializeToString())
