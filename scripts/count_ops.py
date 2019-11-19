from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import sys
import tensorflow as tf

def load_graph(file_name):
  with open(file_name,'rb') as f:
    content = f.read()
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(content)
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')
  return graph

def count_ops(file_name, op_name = None):
  graph = load_graph(file_name)

  if op_name is None:
    return len(graph.get_operations())
  else:
    return sum(1 for op in graph.get_operations() 
               if op.name == op_name)

if __name__ == "__main__":
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  print(count_ops(*sys.argv[1:]))

