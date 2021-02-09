# -*- coding: utf-8 -*-

import copy
import pickle

import tensorflow as tf
import numpy as np
import math
import collections
import pickle as pkl
import os, logging, sys, json, gc
import time
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ''

_CSV_COLUMNS = ['train_batch', 'gds_inputs', 'gen_gds_inputs', 'l4_inputs', 'l3_inputs', 'l2_inputs',
                'big_brand_inputs', 'shop_inputs', 'sal_inputs']


def define_flags():
    flags = tf.app.flags
    tf.app.flags.DEFINE_string("task", "train", "train/dump/inference")
    # Flags Sina ML required
    tf.app.flags.DEFINE_string("data_dir", "sequence",
                               "Set local data path of train set. Coorpate with 'input-strategy DOWNLOAD'.")  # 16902295
    tf.app.flags.DEFINE_string("validate_dir", "valid",
                               "Set local data path of validate set. Coorpate with 'input-strategy DOWNLOAD'.")
    tf.app.flags.DEFINE_string("train_dir", "./model", "Set model save path. Not input path")
    # tf.app.flags.DEFINE_string("log_dir", "./log", "Set tensorboard even log path.")
    # Flags Sina ML required: for tf.train.ClusterSpec
    tf.app.flags.DEFINE_string("ps_hosts", "",
                               "Comma-separated list of hostname:port pairs, you can also specify pattern like ps[1-5].example.com")
    tf.app.flags.DEFINE_string("worker_hosts", "",
                               "Comma-separated list of hostname:port pairs, you can also specify worker[1-5].example.co")
    # Flags Sina ML required:Flags for defining the tf.train.Server
    tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job.Sina ML required arg.")

    flags.DEFINE_string("checkpoints_dir", "/home/predict/net_disk_project/liulu/graph/checkpoints_dir",
                        "Set checkpoints path.")
    # flags.DEFINE_string("model_dir", "./model_dir", "Set checkpoints path.")
    flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
    flags.DEFINE_string("hidden_units", "512,256,128",
                        "Comma-separated list of number of units in each hidden layer of the NN")
    flags.DEFINE_integer("num_epochs", 100, "Number of (global) training steps to perform, default 1000000")
    flags.DEFINE_integer("batch_size", 100, "Training batch size, default 512")
    flags.DEFINE_integer("shuffle_buffer_size", 10000, "dataset shuffle buffer size, default 10000")
    flags.DEFINE_float("learning_rate", 0.025, "Learning rate, default 0.01")
    flags.DEFINE_float("dropout_rate", 0.25, "Drop out rate, default 0.25")
    flags.DEFINE_integer("num_parallel_readers", 4, "number of parallel readers for training data, default 5")
    flags.DEFINE_integer("save_checkpoints_steps", 3000000, "Save checkpoints every this many steps, default 5000")
    flags.DEFINE_boolean("run_on_cluster", False,
                         "Whether the cluster info need to be passed in as input, default False")
    flags.DEFINE_integer("embedding_size", 160, "100/200")
    flags.DEFINE_integer("num_gpus", 0, "1/2")
    flags.DEFINE_integer("num_sampled", 5, "number of negative sample")
    flags.DEFINE_integer("win_len", 5, "behavior embedding dim, 32/64")

    FLAGS = flags.FLAGS
    return FLAGS


FLAGS = define_flags()


def parse_argument():
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    os.environ["TF_ROLE"] = FLAGS.job_name
    os.environ["TF_INDEX"] = str(FLAGS.task_index)

    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = {"worker": worker_spec, "ps": ps_spec}
    os.environ["TF_CLUSTER_DEF"] = json.dumps(cluster)


def set_tfconfig_environ():
    if "TF_CLUSTER_DEF" in os.environ:
        cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
        task_index = int(os.environ["TF_INDEX"])
        task_type = os.environ["TF_ROLE"]

        tf_config = dict()
        worker_num = len(cluster["worker"])
        if FLAGS.job_name == "ps":
            tf_config["task"] = {"index": task_index, "type": task_type}
            FLAGS.job_name = "ps"
            FLAGS.task_index = task_index
        else:
            if task_index == 0:
                tf_config["task"] = {"index": 0, "type": "chief"}
            else:
                tf_config["task"] = {"index": task_index - 1, "type": task_type}
            FLAGS.job_name = "worker"
            FLAGS.task_index = task_index

        if worker_num == 1:
            cluster["chief"] = cluster["worker"]
            del cluster["worker"]
        else:
            cluster["chief"] = [cluster["worker"][0]]
            del cluster["worker"][0]

        tf_config["cluster"] = cluster
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        print("TF_CONFIG", json.loads(os.environ["TF_CONFIG"]))


def my_model(features, labels, mode, params):
    with tf.name_scope('dict'):
        blank_paddings = tf.zeros_like(features['train_batch'], tf.int32)
        gds_embedding_dict = tf.Variable(
            tf.random_uniform([feature_size['gds_inputs'] + 1, FLAGS.embedding_size], -1.0, 1.0,
                              dtype=tf.float32)
        )

        gen_gds_embedding_dict = tf.Variable(
            tf.random_uniform([feature_size['gen_gds_inputs'] + 1, FLAGS.embedding_size], -1.0, 1.0,
                              dtype=tf.float32)
        )

        l4_embedding_dict = tf.Variable(
            tf.random_uniform([feature_size['l4_inputs'], FLAGS.embedding_size], -1.0, 1.0,
                              dtype=tf.float32)
        )

        l3_embedding_dict = tf.Variable(
            tf.random_uniform([feature_size['l3_inputs'], FLAGS.embedding_size], -1.0, 1.0,
                              dtype=tf.float32)
        )

        l2_embedding_dict = tf.Variable(
            tf.random_uniform([feature_size['l2_inputs'], FLAGS.embedding_size], -1.0, 1.0,
                              dtype=tf.float32)
        )

        big_brand_embedding_dict = tf.Variable(
            tf.random_uniform([feature_size['big_brand_inputs'], FLAGS.embedding_size], -1.0, 1.0,
                              dtype=tf.float32)
        )

        sal_embedding_dict = tf.Variable(
            tf.random_uniform([feature_size['sal_inputs'], FLAGS.embedding_size], -1.0, 1.0,
                              dtype=tf.float32)
        )

        shop_embedding_dict = tf.Variable(
            tf.random_uniform([feature_size['shop_inputs'], FLAGS.embedding_size], -1.0, 1.0,
                              dtype=tf.float32)
        )

        weight_dict = tf.Variable(
            tf.random_uniform([vocab_size, len(_CSV_COLUMNS) - 1], -1.0, 1.0, dtype=tf.float32)
        )

        gds_embedding = tf.nn.embedding_lookup(
            gds_embedding_dict, tf.where(tf.less(features['gds_inputs'], feature_size['gds_inputs']),
                                         features['gds_inputs'], blank_paddings)
        )
        gen_gds_embedding = tf.nn.embedding_lookup(
            gen_gds_embedding_dict, tf.where(tf.less(features['gen_gds_inputs'], feature_size['gen_gds_inputs']),
                                             features['gen_gds_inputs'], blank_paddings)
        )
        l4_embedding = tf.nn.embedding_lookup(l4_embedding_dict, features['l4_inputs'])
        l3_embedding = tf.nn.embedding_lookup(l3_embedding_dict, features['l3_inputs'])
        l2_embedding = tf.nn.embedding_lookup(l2_embedding_dict, features['l2_inputs'])
        big_brand_embedding = tf.nn.embedding_lookup(big_brand_embedding_dict, features['big_brand_inputs'])
        sal_splits = tf.string_split(features['sal_inputs'], '|')
        sal_value = tf.SparseTensor(indices=sal_splits.indices,
                                    values=tf.string_to_number(sal_splits.values, tf.int32),
                                    dense_shape=[sal_splits.dense_shape[0], 10])
        sal_embedding = tf.nn.embedding_lookup_sparse(sal_embedding_dict, sp_ids=sal_value, sp_weights=None)
        shop_embedding = tf.nn.embedding_lookup(
            shop_embedding_dict, tf.where(tf.less(features['shop_inputs'], feature_size['shop_inputs']),
                                          features['shop_inputs'], blank_paddings))

        embedding_list = [gds_embedding, gen_gds_embedding, shop_embedding, l4_embedding, l3_embedding,
                          l2_embedding, big_brand_embedding, sal_embedding]
        weight = tf.exp(tf.nn.embedding_lookup(weight_dict, features['train_batch']))
        weight_sum = tf.reduce_sum(weight, axis=1, keep_dims=True)
        normlized_weight = tf.truediv(weight, weight_sum, name='normalized_weight')

    with tf.name_scope('embeddings'):
        embed = tf.reduce_sum(
            tf.expand_dims(normlized_weight, -1) * tf.stack(embedding_list, axis=1), axis=1,
            name='embed')

    _weight = tf.ones_like(normlized_weight, tf.float32)  # B*F
    _gds_weight = tf.where(tf.less(features['gds_inputs'], feature_size['gds_inputs']),
                           tf.ones_like(features['train_batch'], tf.float32),
                           tf.zeros_like(features['train_batch'], tf.float32))
    _gen_gds_weight = tf.where(tf.less(features['gen_gds_inputs'], feature_size['gen_gds_inputs']),
                               tf.ones_like(features['train_batch'], tf.float32),
                               tf.zeros_like(features['train_batch'], tf.float32))
    ges_weight = tf.concat([tf.stack([_gds_weight, _gen_gds_weight], axis=1),
                            tf.slice(_weight, [0, 2], [-1, -1])], axis=1)  # B*F
    feature_len = tf.reduce_sum(ges_weight, axis=1, keepdims=True)  # B*1
    avg_embed = tf.reduce_sum(tf.expand_dims(ges_weight, -1) * tf.stack(embedding_list, axis=1), axis=1) / feature_len
    avg_normalized_embed = tf.reshape(avg_embed, [FLAGS.batch_size, -1], name='avg_embed')

    with tf.name_scope('weights'):
        nce_weight = tf.Variable(tf.truncated_normal([vocab_size, FLAGS.embedding_size],
                                                     stddev=1.0 / math.sqrt(FLAGS.embedding_size)))
    with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocab_size]))

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weight,
                biases=nce_biases,
                labels=labels,
                inputs=embed,
                num_sampled=FLAGS.num_sampled,
                num_classes=vocab_size
            )
        )

    tf.summary.scalar('loss', loss)
    avg = tf.metrics.mean(loss)

    metrics = {'avg': avg}
    hook_dict = {"loss": avg[1]}
    logging_hook = tf.train.LoggingTensorHook(hook_dict, every_n_iter=20)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, training_hooks=[logging_hook])

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    # AdagradOptimizer/AdamOptimizer
    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss,
                                                                               global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics,
                                      training_hooks=[logging_hook])


def build_estimator(model_dir):
    """Build an estimator appropriate for the given model type."""
    set_tfconfig_environ()

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig(
        session_config=tf.ConfigProto(device_count={'GPU': FLAGS.num_gpus, 'CPU': 40},
                                      inter_op_parallelism_threads=0,
                                      intra_op_parallelism_threads=0),
        save_checkpoints_secs=FLAGS.save_checkpoints_steps,  # 300
        keep_checkpoint_max=3,
        model_dir=model_dir)

    model = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'learning_rate': FLAGS.learning_rate,
            'dropout_rate': FLAGS.dropout_rate
        },
        config=run_config
    )

    return model


def ctx_idxx(target_idx, window_size, tokens):
    """Return positions of context words."""
    ctx_range = tf.range(start=tf.maximum(tf.constant(0, dtype=tf.int32),
                                          target_idx - window_size),
                         limit=tf.minimum(tf.size(input=tokens, out_type=tf.int32),
                                          target_idx + window_size + 1),
                         delta=1, dtype=tf.int32)
    idx = tf.case({tf.less_equal(target_idx, window_size): lambda: target_idx,
                   tf.greater(target_idx, window_size): lambda: window_size},
                  exclusive=True)
    t0 = lambda: tf.constant([], dtype=tf.int32)
    t1 = lambda: ctx_range[idx + 1:]
    t2 = lambda: ctx_range[0:idx]
    t3 = lambda: tf.concat([ctx_range[0:idx], ctx_range[idx + 1:]], axis=0)
    c1 = tf.logical_and(tf.equal(idx, 0),
                        tf.less(idx + 1, tf.size(input=ctx_range, out_type=tf.int32)))
    c2 = tf.logical_and(tf.greater(idx, 0),
                        tf.equal(idx + 1, tf.size(input=ctx_range, out_type=tf.int32)))
    c3 = tf.logical_and(tf.greater(idx, 0),
                        tf.less(idx + 1, tf.size(input=ctx_range, out_type=tf.int32)))
    return tf.case({c1: t1, c2: t2, c3: t3}, default=t0, exclusive=True)


# pylint: disable=R1710
def concat_to_features_and_labels(tokens, train_mode, window_size):
    """Concatenate features and labels into Tensor."""

    def internal_func(features, labels, target_idx):
        if train_mode not in ['cbow', 'skipgram']:
            raise Exception('Unsupported Word2Vec mode \'{}\''
                            .format(train_mode))
        ctxs = ctx_idxx(target_idx, window_size, tokens)
        if train_mode == 'cbow':
            ctx_features = tf.gather(tokens, ctxs)
            diff = 2 * window_size - tf.size(input=ctx_features)
            ctx_features = tf.reshape(ctx_features, [1, -1])
            paddings = tf.concat(
                [tf.constant([[0, 0]]),
                 tf.concat([tf.constant([[0]]), [[diff]]], axis=1)], axis=0)
            padded_ctx_features = tf.pad(tensor=ctx_features, paddings=paddings,
                                         constant_values='_CBOW#_!MASK_')
            label = tf.reshape(tokens[target_idx], [1, -1])
            return tf.concat([features, padded_ctx_features], axis=0), \
                   tf.concat([labels, label], axis=0), target_idx + 1
        if train_mode == 'skipgram':
            label = tf.reshape(tf.gather(tokens, ctxs), [-1, 1])
            feature = tf.fill([tf.size(input=label)], tokens[target_idx])
            return tf.concat([features, feature], axis=0), \
                   tf.concat([labels, label], axis=0), target_idx + 1

    return internal_func


def extract_examples(tokens, train_mode, window_size, p_num_threads):
    """Extract (features, labels) examples from a list of tokens."""
    if train_mode not in ['cbow', 'skipgram']:
        raise Exception('Unsupported Word2Vec mode \'{}\''
                        .format(train_mode))
    features_dict_one = tf.reshape(features_array_one, [tf.shape(features_array_one)[0], len(_CSV_COLUMNS) - 2])
    features_dict_multi = tf.reshape(features_array_multi, [tf.shape(features_array_multi)[0], 1])
    train_batch = dict()
    if train_mode == 'cbow':
        features = tf.constant([], shape=[0, 2 * window_size], dtype=tf.int32)
    elif train_mode == 'skipgram':
        features = tf.constant([], dtype=tf.int32)

    labels = tf.constant([], shape=[0, 1], dtype=tf.int32)
    target_idx = tf.constant(0, dtype=tf.int32)
    concat_func = concat_to_features_and_labels(tokens, train_mode,
                                                window_size)
    max_size = tf.size(input=tokens, out_type=tf.int32)
    idx_below_tokens_size = lambda w, x, idx: tf.less(idx, max_size)
    if train_mode == 'cbow':
        result = tf.while_loop(
            cond=idx_below_tokens_size,
            body=concat_func,
            loop_vars=[features, labels, target_idx],
            shape_invariants=[tf.TensorShape([None, 2 * window_size]),
                              tf.TensorShape([None, 1]),
                              target_idx.get_shape()],
            parallel_iterations=p_num_threads)
    elif train_mode == 'skipgram':
        result = tf.while_loop(
            cond=idx_below_tokens_size,
            body=concat_func,
            loop_vars=[features, labels, target_idx],
            shape_invariants=[tf.TensorShape([None]),
                              tf.TensorShape([None, 1]),
                              target_idx.get_shape()],
            parallel_iterations=p_num_threads)
    node_features_one = tf.nn.embedding_lookup(features_dict_one, result[0])
    node_features_multi = tf.nn.embedding_lookup(features_dict_multi, result[0])
    train_batch['train_batch'] = result[0]
    train_batch['gds_inputs'] = tf.cast(tf.reshape(tf.slice(node_features_one, [0, 0], [-1, 1]), [-1]), tf.int32)
    train_batch['gen_gds_inputs'] = tf.cast(tf.reshape(tf.slice(node_features_one, [0, 1], [-1, 1]), [-1]), tf.int32)
    train_batch['l4_inputs'] = tf.cast(tf.reshape(tf.slice(node_features_one, [0, 2], [-1, 1]), [-1]), tf.int32)
    train_batch['l3_inputs'] = tf.cast(tf.reshape(tf.slice(node_features_one, [0, 3], [-1, 1]), [-1]), tf.int32)
    train_batch['l2_inputs'] = tf.cast(tf.reshape(tf.slice(node_features_one, [0, 4], [-1, 1]), [-1]), tf.int32)
    train_batch['big_brand_inputs'] = tf.cast(tf.reshape(tf.slice(node_features_one, [0, 5], [-1, 1]), [-1]), tf.int32)
    train_batch['shop_inputs'] = tf.cast(tf.reshape(tf.slice(node_features_one, [0, 6], [-1, 1]), [-1]), tf.int32)
    train_batch['sal_inputs'] = tf.reshape(tf.slice(node_features_multi, [0, 0], [-1, 1]), [-1])

    return train_batch, result[1]


def input_fn(training_data_filepath, train_mode, window_size,
             batch_size, num_epochs, p_num_threads, shuffling_buffer_size):
    """Generate a Tensorflow Dataset for a Word2Vec model."""
    """reference: https://github.com/akb89/word2vec"""
    # word_count_table = vocab_utils.get_tf_word_count_table(words, counts)
    files = tf.data.Dataset.list_files(training_data_filepath)
    # Extract lines from input files using the Dataset API.
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=5, sloppy=True)
    )
    dataset = dataset.map(tf.strings.strip, num_parallel_calls=p_num_threads) \
        .filter(lambda x: tf.not_equal(tf.strings.length(input=x), 0)) \
        .map(lambda x: tf.string_to_number(tf.string_split([x], delimiter=' ').values, out_type=tf.int32),
             num_parallel_calls=p_num_threads) \
        .filter(lambda x: tf.greater(tf.size(input=x), 1)) \
        .map(lambda x: extract_examples(x, train_mode, window_size,
                                        p_num_threads),
             num_parallel_calls=p_num_threads) \
        .flat_map(lambda features, labels: \
                      tf.data.Dataset.from_tensor_slices((features, labels))) \
        .shuffle(buffer_size=shuffling_buffer_size,
                 reshuffle_each_iteration=False) \
        .repeat(num_epochs) \
        .batch(batch_size, drop_remainder=True).prefetch(batch_size)
    # we need drop_remainder to statically know the batch dimension
    # this is required to get features.get_shape()[0] in
    # w2v_estimator.avg_ctx_features_embeddings
    input_tensor_map = dict()
    dataset_iter = dataset.make_initializable_iterator()  # .make_one_shot_iterator()
    features, labels = dataset_iter.get_next()

    # print('features:' + str(features))
    for input_name, tensor in features.items():
        input_tensor_map[input_name] = tensor.name

    with open(os.path.join(FLAGS.checkpoints_dir, 'input_tensor_map.pickle'), 'wb') as f:
        pickle.dump(input_tensor_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_iter.initializer)

    return features, labels


def eval_input_fn(training_data_filepath, train_mode, window_size, batch_size,
                  p_num_threads):
    # Extract lines from input files using the Dataset API.
    files = tf.data.Dataset.list_files(training_data_filepath)
    # Extract lines from input files using the Dataset API.
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TextLineDataset, cycle_length=1, sloppy=True)
    )
    dataset = dataset.map(tf.strings.strip, num_parallel_calls=p_num_threads) \
        .filter(lambda x: tf.not_equal(tf.strings.length(input=x), 0)) \
        .map(lambda x: tf.string_to_number(tf.string_split([x], delimiter=' ').values, out_type=tf.int32),
             num_parallel_calls=p_num_threads) \
        .filter(lambda x: tf.greater(tf.size(input=x), 1)) \
        .map(lambda x: extract_examples(x, train_mode, window_size, p_num_threads),
             num_parallel_calls=p_num_threads) \
        .flat_map(lambda features, labels: \
                      tf.data.Dataset.from_tensor_slices((features, labels))) \
        .batch(batch_size, drop_remainder=True)

    return dataset


def main(unused_argv):
    # Clean up the model directory if present
    # shutil.rmtree(FLAGS.checkpoints_dir, ignore_errors=True)

    model = build_estimator(FLAGS.checkpoints_dir)
    if isinstance(FLAGS.data_dir, str) and os.path.isdir(FLAGS.data_dir):
        train_files = [FLAGS.data_dir + '/' + x for x in os.listdir(FLAGS.data_dir)] if os.path.isdir(
            FLAGS.data_dir) else FLAGS.data_dir
    else:
        train_files = FLAGS.data_dir
    if isinstance(FLAGS.validate_dir, str) and os.path.isdir(FLAGS.validate_dir):
        eval_files = [FLAGS.validate_dir + '/' + x for x in os.listdir(FLAGS.validate_dir)] if os.path.isdir(
            FLAGS.validate_dir) else FLAGS.validate_dir
    else:
        eval_files = FLAGS.validate_dir

    print('train files: ' + str(train_files))
    print('eval files: ' + str(eval_files))

    # train process
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(training_data_filepath=train_files,
                                  train_mode='skipgram',
                                  window_size=FLAGS.win_len,
                                  batch_size=FLAGS.batch_size,
                                  num_epochs=FLAGS.num_epochs,
                                  p_num_threads=FLAGS.num_parallel_readers,
                                  shuffling_buffer_size=FLAGS.shuffle_buffer_size),
        max_steps=int(vocab_size // 2)
    )
    input_fn_for_eval = lambda: eval_input_fn(training_data_filepath=eval_files,
                                              train_mode='skipgram',
                                              window_size=FLAGS.win_len,
                                              batch_size=FLAGS.batch_size,
                                              p_num_threads=FLAGS.num_parallel_readers, )
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=300)

    print("before train and evaluate")
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    print("after train and evaluate")


def load_model(output_node_names, sd_path, item_emb_path, id_path):
    """Extract the sub graph defined by the output nodes and convert
        all its variables into constant
        Args:
            model_dir: the root folder containing the checkpoint state file
            output_node_names: a string, containing all the output node's names,
                                comma separated
        """
    if (output_node_names is None):
        output_node_names = 'loss'

    if not tf.gfile.Exists(my_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % my_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(my_dir)
    print(checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    # absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    # output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        with open(os.path.join(my_dir, 'input_tensor_map.pickle'), 'rb') as f:
            input_tensor_map = pickle.load(f)

        batch_size = FLAGS.batch_size
        info_fp = open('/home/predict/net_disk_project/liulu/graph/datainfo', 'r')
        vocab_size = int(info_fp.readline().strip())
        info_fp.close()
        sd_fp = open(sd_path, 'r')
        item_fp = open(item_emb_path, 'w')
        id_fp = open(id_path, 'w')

        with open('/home/predict/net_disk_project/liulu/graph/feature_dict.pickle', 'rb') as f:
            feature_dict = pickle.load(f)

        feature_size = {}
        for k in _CSV_COLUMNS[1:]:
            feature_size[k] = len(feature_dict[k])

        end_of_file = False
        empty_X = {}
        id = []
        for k in _CSV_COLUMNS:
            empty_X[k] = []
        print("Begin inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        while True:
            X_validate = copy.deepcopy(empty_X)
            read_line_num = 0

            while True:
                line = sd_fp.readline().strip().split()
                if len(line) < len(_CSV_COLUMNS):
                    end_of_file = True
                    break

                arr = []
                gds_cd = line[1]
                shop = line[7]
                arr.append(int(line[0]))
                for i in range(1, len(_CSV_COLUMNS)):
                    if _CSV_COLUMNS[i] == 'sal_inputs':
                        arr.append('|'.join([str(feature_dict[_CSV_COLUMNS[i]][x]) for x in line[i].split('|')]))
                    else:
                        arr.append(int(feature_dict[_CSV_COLUMNS[i]][line[i]]))

                if len(arr) != len(_CSV_COLUMNS):
                    break

                id.append(gds_cd + '_' + shop + '\n')

                for i in range(0, len(_CSV_COLUMNS)):
                    X_validate[_CSV_COLUMNS[i]].append(arr[i])

                read_line_num += 1
                if read_line_num == batch_size:
                    break

            if read_line_num < batch_size:
                vacant_line = batch_size - read_line_num
                for v in range(0, vacant_line):
                    for i in range(0, len(_CSV_COLUMNS) - 1):
                        X_validate[_CSV_COLUMNS[i]].append(0)
                    X_validate[_CSV_COLUMNS[8]].append('0')

            input_feed = dict()
            for key, tensor_name in input_tensor_map.items():
                # print("--------------tensor name---------"+tensor_name)
                tensor = sess.graph.get_tensor_by_name(tensor_name)
                # print(str(key)+'\t'+str(tensor_name)+'\t'+str(tensor))
                input_feed[tensor] = X_validate[key]

            embed = tf.get_default_graph().get_operation_by_name(output_node_names).outputs[-1]
            item_embedding = sess.run(embed, feed_dict=input_feed)
            norm = np.linalg.norm(item_embedding, axis=1, keepdims=True)
            emb = item_embedding/norm
            np.savetxt(item_fp, emb[:read_line_num], delimiter=',', fmt='%s')

            if end_of_file:
                break
        print("Inference finished")
        id_fp.writelines(id)
        sd_fp.close()
        item_fp.close()
        id_fp.close()


def load_model_v2(output_node_names, sd_path, item_emb_path, id_path):
    """Extract the sub graph defined by the output nodes and convert
        all its variables into constant
        Args:
            model_dir: the root folder containing the checkpoint state file
            output_node_names: a string, containing all the output node's names,
                                comma separated
        """
    if (output_node_names is None):
        output_node_names = 'loss'

    if not tf.gfile.Exists(my_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % my_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(my_dir)
    print(checkpoint)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    # absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    # output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        with open(os.path.join(my_dir, 'input_tensor_map.pickle'), 'rb') as f:
            input_tensor_map = pickle.load(f)

        batch_size = FLAGS.batch_size
        info_fp = open('/home/predict/net_disk_project/liulu/graph/datainfo', 'r')
        vocab_size = int(info_fp.readline().strip())
        info_fp.close()
        sd_fp = open(sd_path, 'r')
        item_fp = open(item_emb_path, 'w')
        id_fp = open(id_path, 'w')

        with open('/home/predict/net_disk_project/liulu/graph/feature_dict.pickle', 'rb') as f:
            feature_dict = pickle.load(f)

        feature_size = {}
        for k in _CSV_COLUMNS[1:]:
            feature_size[k] = len(feature_dict[k])

        end_of_file = False
        empty_X = {}
        id = []
        for k in _CSV_COLUMNS:
            empty_X[k] = []
        print("Begin inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        while True:
            X_validate = copy.deepcopy(empty_X)
            read_line_num = 0

            while True:
                line = sd_fp.readline().strip().split()
                if len(line) < len(_CSV_COLUMNS):
                    end_of_file = True
                    break

                sal_flag = False
                sal_inputs = []
                for s in line[8].split('|'):
                    if feature_dict[_CSV_COLUMNS[8]].get(s):
                        sal_flag = True
                        sal_inputs.append(feature_dict[_CSV_COLUMNS[8]].get(s))

                if feature_dict[_CSV_COLUMNS[3]].get(line[3]) and feature_dict[_CSV_COLUMNS[4]].get(line[4]) and \
                        feature_dict[_CSV_COLUMNS[5]].get(line[5]) and feature_dict[_CSV_COLUMNS[6]].get(
                    line[6]) and sal_flag and feature_dict[_CSV_COLUMNS[7]].get(line[7]):
                    arr = []
                    gds_cd = line[1]
                    shop = line[7]
                    if int(line[0]) >= 0 and int(line[0]) < vocab_size:
                        arr.append(int(line[0]))
                    else:
                        arr.append(vocab_size)

                    for i in range(1, len(_CSV_COLUMNS)):
                        if _CSV_COLUMNS[i] in ['gds_inputs', 'gen_gds_inputs']:
                            arr.append(int(feature_dict[_CSV_COLUMNS[i]][line[i]])) if feature_dict[
                                _CSV_COLUMNS[i]].get(
                                line[i]) else arr.append(feature_size[_CSV_COLUMNS[i]])
                        elif _CSV_COLUMNS[i] == 'sal_inputs':
                            arr.append('|'.join([str(x) for x in sal_inputs]))
                        else:
                            arr.append(int(feature_dict[_CSV_COLUMNS[i]][line[i]]))

                    if len(arr) != len(_CSV_COLUMNS):
                        break

                    id.append(gds_cd + '_' + shop + '\n')

                    for i in range(0, len(_CSV_COLUMNS)):
                        X_validate[_CSV_COLUMNS[i]].append(arr[i])

                    read_line_num += 1
                    if read_line_num == batch_size:
                        break

            if read_line_num < batch_size:
                vacant_line = batch_size - read_line_num
                for v in range(0, vacant_line):
                    for i in range(0, len(_CSV_COLUMNS) - 1):
                        X_validate[_CSV_COLUMNS[i]].append(0)
                    X_validate[_CSV_COLUMNS[8]].append('0')

            input_feed = dict()
            for key, tensor_name in input_tensor_map.items():
                # print("--------------tensor name---------"+tensor_name)
                tensor = sess.graph.get_tensor_by_name(tensor_name)
                # print(str(key)+'\t'+str(tensor_name)+'\t'+str(tensor))
                input_feed[tensor] = X_validate[key]

            embed = tf.get_default_graph().get_operation_by_name(output_node_names).outputs[-1]
            item_embedding = sess.run(embed, feed_dict=input_feed)
            norm = np.linalg.norm(item_embedding, axis=1, keepdims=True)
            emb = item_embedding / norm
            np.savetxt(item_fp, emb[:read_line_num], delimiter=',', fmt='%s')

            if end_of_file:
                break
        print("Inference finished")
        id_fp.writelines(id)
        sd_fp.close()
        item_fp.close()
        id_fp.close()


if __name__ == '__main__':
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    if FLAGS.run_on_cluster: parse_argument()
    tf.logging.set_verbosity(tf.logging.INFO)

    my_dir = FLAGS.checkpoints_dir
    if FLAGS.task == 'inference':
        load_model('embeddings/embed', '/home/predict/net_disk_project/liulu/graph/side_info_avail', \
                   '/home/predict/net_disk_project/liulu/graph/embedding', \
                   '/home/predict/net_disk_project/liulu/graph/itemlist')
    elif FLAGS.task == 'inference_syt_eges':
        load_model('embeddings/embed', 'syt_gds1', 'syt_embedding1', 'syt_id1')
    elif FLAGS.task == 'inference_syt_ges':
        load_model_v2('avg_embed', 'syt_gds2', 'syt_embedding2', 'syt_id2')
    elif FLAGS.task == 'inference_syt_seed':
        load_model('embeddings/embed', 'syt_seed', 'syt_seed_embedding', 'syt_seed_id')
    elif FLAGS.task == 'inference_for_atten':
        load_model('embeddings/embed', '/home/predict/net_disk_project/liulu/graph/side_info', \
                   '/home/predict/net_disk_project/liulu/graph/embedding_atten', \
                   '/home/predict/net_disk_project/liulu/graph/node')
    else:
        sd_fp = open('/home/predict/net_disk_project/liulu/graph/side_info', 'r')
        feature_set = {}
        for k in _CSV_COLUMNS[1:]:
            feature_set[k] = set()
        while True:
            line = sd_fp.readline().strip().split()
            if len(line) < len(_CSV_COLUMNS):
                break
            for i in range(1, len(_CSV_COLUMNS)):
                if _CSV_COLUMNS[i] == 'sal_inputs':
                    feature_set[_CSV_COLUMNS[i]].update(line[i].split('|'))
                else:
                    feature_set[_CSV_COLUMNS[i]].add(line[i])
        sd_fp.close()
        feature_dict = {}
        for k in _CSV_COLUMNS[1:]:
            feature_dict[k] = {key: index for index, key in enumerate(feature_set[k])}
        global feature_size
        feature_size = {}
        for k in _CSV_COLUMNS[1:]:
            feature_size[k] = len(feature_dict[k])

        with open('/home/predict/net_disk_project/liulu/graph/feature_dict.pickle', 'wb') as f:
            pickle.dump(feature_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        sd_fp = open('/home/predict/net_disk_project/liulu/graph/side_info', 'r')
        side_info_list_one = []
        side_info_list_multi = []
        while True:
            line = sd_fp.readline().strip().split()
            if len(line) < len(_CSV_COLUMNS):
                break
            for i in range(1, len(_CSV_COLUMNS)):
                if _CSV_COLUMNS[i] == 'sal_inputs':
                    line[i] = '|'.join([str(feature_dict[_CSV_COLUMNS[i]][x]) for x in line[i].split('|')])
                else:
                    line[i] = int(feature_dict[_CSV_COLUMNS[i]][line[i]])
            side_info_list_one.append(line[1:len(_CSV_COLUMNS) - 1])
            side_info_list_multi.append(line[len(_CSV_COLUMNS) - 1:len(_CSV_COLUMNS)])

        sd_fp.close()
        print("Load sequence ...")
        info_fp = open('/home/predict/net_disk_project/liulu/graph/datainfo', 'r')
        num_nodes = int(info_fp.readline().strip())
        info_fp.close()
        print(num_nodes, ' gds in total')
        global vocab_size
        vocab_size = num_nodes
        global features_array_one
        features_array_one = side_info_list_one
        global features_array_multi
        features_array_multi = side_info_list_multi
        del feature_dict
        del feature_set
        del side_info_list_one
        del side_info_list_multi
        gc.collect()
        tf.app.run(main=main)