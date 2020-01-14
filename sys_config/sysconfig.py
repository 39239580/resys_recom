import tensorflow as tf

tf_version = tf.__version__

if tf_version == "1.14.0":
    tf_v = 1

elif tf_version == "2.0.0":
    tf_v = 2
