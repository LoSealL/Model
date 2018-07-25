import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

tf.flags.DEFINE_string('model', './model.pb', 'model path')
tf.flags.DEFINE_string('test_dir', './Data', 'test images path')
tf.flags.DEFINE_string('save_dir', './Save', 'output path')
FLAGS = tf.flags.FLAGS


def main(*args):
    with tf.gfile.FastGFile(FLAGS.model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='import')
    images = Path(FLAGS.test_dir).glob('*.png')
    images = [(Image.open(fp), fp.stem) for fp in images]
    images = [(np.asarray(img.convert('RGB'), dtype='float32'), name) for img, name in images]

    Path(FLAGS.save_dir).mkdir(exist_ok=True)

    with tf.Session() as sess:
        graph = tf.get_default_graph()
        outputs = graph.get_tensor_by_name('import/rdn/mul:0')
        inputs = graph.get_tensor_by_name('import/rdn/input/lr/gray:0')
        for img, name in images:
            img = np.expand_dims(img, 0)
            pred = sess.run(outputs, feed_dict={inputs: img})
            pred = pred[0].astype('uint8')
            Image.fromarray(pred, 'RGB').save(f'{FLAGS.save_dir}/{name}.png')


if __name__ == '__main__':
    tf.app.run(main)
