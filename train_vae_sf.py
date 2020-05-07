import datetime
import os
import shutil

import bottleneck as bn
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from scipy import sparse

from model import MultiVAESF

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'data/ml-20m', 'data dir')
flags.DEFINE_string('exp_dir', 'exp_out', 'experiment output dir')
flags.DEFINE_string('timestamp', None, 'timestamp')
flags.DEFINE_integer('n_epochs', 20, 'num epochs')

logging.set_verbosity(logging.INFO)

def train():

    if FLAGS.timestamp is None:
        now = datetime.datetime.now()
        ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second)
    else:
        ts = FLAGS.timestamp

    exp_out_dir = os.path.join(FLAGS.exp_dir, ts)
    os.makedirs(exp_out_dir, exist_ok=True)

    pro_dir = os.path.join(FLAGS.data_dir, 'pro_sg')

    unique_sid = list()
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)

    logging.info('n_items %s', n_items)

    def load_train_data(csv_file):
        tp = pd.read_csv(csv_file)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                  (rows, cols)), dtype='float64',
                                 shape=(n_users, n_items))
        return data



    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'))


    def load_tr_te_data(csv_file_tr, csv_file_te):
        tp_tr = pd.read_csv(csv_file_tr)
        tp_te = pd.read_csv(csv_file_te)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                     (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                     (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
        return data_tr, data_te



    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'),
                                               os.path.join(pro_dir, 'validation_te.csv'))

    # Set up training hyperparameters


    N = train_data.shape[0]
    idxlist = [x for x in range(N)]

    # training batch size
    batch_size = 500
    batches_per_epoch = int(np.ceil(float(N) / batch_size))

    N_vad = vad_data_tr.shape[0]
    idxlist_vad = [x for x in range(N_vad)]

    # validation batch size (since the entire validation set might not fit into GPU memory)
    batch_size_vad = 2000

    # the total number of gradient updates for annealing
    total_anneal_steps = 200000
    # largest annealing parameter
    anneal_cap = 0.2

    # Evaluate function: Normalized discounted cumulative gain (NDCG@k) and Recall@k

    def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
        '''
        normalized discounted cumulative gain@k for binary relevance
        ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
        '''
        batch_users = X_pred.shape[0]
        idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
        topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                           idx_topk_part[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
        # topk predicted score
        idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
        # build the discount template
        tp = 1. / np.log2(np.arange(2, k + 2))

        DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                             idx_topk].toarray() * tp).sum(axis=1)
        IDCG = np.array([(tp[:min(n, k)]).sum()
                         for n in heldout_batch.getnnz(axis=1)])
        return DCG / IDCG

    def Recall_at_k_batch(X_pred, heldout_batch, k=100):
        batch_users = X_pred.shape[0]

        idx = bn.argpartition(-X_pred, k, axis=1)
        X_pred_binary = np.zeros_like(X_pred, dtype=bool)
        X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

        X_true_binary = (heldout_batch > 0).toarray()
        tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
            np.float32)
        recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
        return recall


    # ### Train a Multi-VAE^{PR}

    # For ML-20M dataset, we set both the generative function $f_\theta(\cdot)$ and the inference model $g_\phi(\cdot)$ to be 3-layer multilayer perceptron (MLP) with symmetrical architecture.

    # The generative function is a [200 -> 600 -> n_items] MLP, which means the inference function is a [n_items -> 600 -> 200] MLP. Thus the overall architecture for the Multi-VAE^{PR} is [n_items -> 600 -> 200 -> 600 -> n_items].

    #         -3      -2        -1
    # 200 --> 600 --> 200 --> n_items
    p_dims = [200, 600, 200, n_items]
    q_dims = [n_items, 600, 200]

    tf.reset_default_graph()
    vae = MultiVAESF(p_dims, q_dims, lam=0.0, random_seed=98765)

    saver, logits_var, loss_var, train_op_var, merged_var = vae.build_graph()

    ndcg_var = tf.Variable(0.0)
    ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
    ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
    ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
    merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])

    # Set up logging and checkpoint directory
    #
    # - Change all the logging directory and checkpoint directory to somewhere of your choice
    # - Monitor training progress using tensorflow by: `tensorboard --logdir=$log_dir`


    arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))



    log_dir = '{}/log/ml-20m/VAESF_anneal{}K_cap{:1.1E}/{}'.format(exp_out_dir,
        total_anneal_steps / 1000, anneal_cap, arch_str)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    logging.info("log directory: %s" % log_dir)
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    chkpt_dir = '{}/chkpt/ml-20m/VAESF_anneal{}K_cap{:1.1E}/{}'.format(exp_out_dir,
        total_anneal_steps / 1000, anneal_cap, arch_str)

    if not os.path.isdir(chkpt_dir):
        os.makedirs(chkpt_dir)

    logging.info("chkpt directory: %s" % chkpt_dir)

    n_epochs = FLAGS.n_epochs

    ndcgs_vad = []

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        best_ndcg = -np.inf

        update_count = 0.0

        for epoch in range(n_epochs):
            logging.info('[train] epoch %s', epoch)
            np.random.shuffle(idxlist)
            # train for one epoch
            for bnum, st_idx in enumerate(range(0, N, batch_size)):
                end_idx = min(st_idx + batch_size, N)
                X = train_data[idxlist[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')

                if total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                else:
                    anneal = anneal_cap

                feed_dict = {vae.input_ph: X,
                             vae.keep_prob_ph: 0.5,
                             vae.anneal_ph: anneal,
                             vae.is_training_ph: 1}
                sess.run(train_op_var, feed_dict=feed_dict)

                if bnum % 100 == 0:
                    summary_train = sess.run(merged_var, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_train,
                                               global_step=epoch * batches_per_epoch + bnum)

                update_count += 1

            # compute validation NDCG
            ndcg_dist = []
            for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
                end_idx = min(st_idx + batch_size_vad, N_vad)
                X = vad_data_tr[idxlist_vad[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')

                pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                ndcg_dist.append(NDCG_binary_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))

            ndcg_dist = np.concatenate(ndcg_dist)
            ndcg_ = ndcg_dist.mean()
            ndcgs_vad.append(ndcg_)
            merged_valid_val = sess.run(merged_valid, feed_dict={ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
            summary_writer.add_summary(merged_valid_val, epoch)

            # update the best model (if necessary)
            if ndcg_ > best_ndcg:
                logging.info('[validation] epoch %s new best ndcg %s > %s', epoch, ndcg_, best_ndcg)
                saver.save(sess, '{}/model'.format(chkpt_dir))
                best_ndcg = ndcg_

    # plt.figure(figsize=(12, 3))
    # plt.plot(ndcgs_vad)
    # plt.ylabel("Validation NDCG@100")
    # plt.xlabel("Epochs")
    # pass

    # ### Load the test data and compute test metrics
    logging.info('[test] running test evaluation!')

    test_data_tr, test_data_te = load_tr_te_data(
        os.path.join(pro_dir, 'test_tr.csv'),
        os.path.join(pro_dir, 'test_te.csv'))

    N_test = test_data_tr.shape[0]
    idxlist_test = [x for x in range(N_test)]

    batch_size_test = 2000

    tf.reset_default_graph()
    vae = MultiVAESF(p_dims, q_dims, lam=0.0)
    saver, logits_var, _, _, _ = vae.build_graph()

    # Load the best performing model on the validation set

    chkpt_dir = '{}/chkpt/ml-20m/VAESF_anneal{}K_cap{:1.1E}/{}'.format(exp_out_dir,
        total_anneal_steps / 1000, anneal_cap, arch_str)
    logging.info("[test] chkpt directory: %s" % chkpt_dir)


    n100_list, r20_list, r50_list = [], [], []

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        saver.restore(sess, '{}/model'.format(chkpt_dir))

        for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, N_test)
            X = test_data_tr[idxlist_test[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')

            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: X})
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
            r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
            r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    logging.info("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    logging.info("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    logging.info("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))

def main(argv):
    logging.info('Running with args %s', str(argv))
    train()


if __name__ == "__main__":
    app.run(main)
