import os
import sys

import numpy as np
import pandas as pd
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'data/ml-20m', 'Where to store the data.')

logging.set_verbosity(logging.INFO)

# ## Data preprocessing

# We load the data and create train/validation/test splits following strong generalization:
#
# - We split all users into training/validation/test sets.
#
# - We train models using the entire click history of the training users.
#
# - To evaluate, we take part of the click history from held-out (validation and test) users to learn the necessary user-level representations for the model and then compute metrics by looking at how well the model ranks the rest of the unseen click history from the held-out users.

# First, download the dataset at http://files.grouplens.org/datasets/movielens/ml-20m.zip

def download():
    if not os.path.exists(FLAGS.data_dir):
        parent_dir = "/".join(os.path.split(FLAGS.data_dir)[0:-1])
        logging.info('Downloading the data to %s', parent_dir)
        os.system("wget --directory-prefix=%s http://files.grouplens.org/datasets/movielens/ml-20m.zip" % parent_dir)
        os.system('unzip %s -d %s' % (os.path.join(parent_dir, 'ml-20m.zip'), FLAGS.data_dir))
    else:
        logging.info('Data already exists! %s', FLAGS.data_dir)


def process():
    data_dir = FLAGS.data_dir

    raw_data = pd.read_csv(os.path.join(data_dir, 'ratings.csv'), header=0)

    # binarize the data (only keep ratings >= 4)
    raw_data = raw_data[raw_data['rating'] > 3.5]

    raw_data.head()


    # ### Data splitting procedure

    # - Select 10K users as heldout users, 10K users as validation users, and the rest of the users for training
    # - Use all the items from the training users as item set
    # - For each of both validation and test user, subsample 80% as fold-in data and the rest for prediction

    def get_count(tp, id):
        playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
        count = playcount_groupbyid.size()
        return count

    def filter_triplets(tp, min_uc=5, min_sc=0):
        # Only keep the triplets for items which were clicked on by at least min_sc users.
        if min_sc > 0:
            itemcount = get_count(tp, 'movieId')
            tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]

        # Only keep the triplets for users who clicked on at least min_uc items
        # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
        if min_uc > 0:
            usercount = get_count(tp, 'userId')
            tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]

        # Update both usercount and itemcount after filtering
        usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
        return tp, usercount, itemcount


    # Only keep items that are clicked on by at least 5 users
    raw_data, user_activity, item_popularity = filter_triplets(raw_data)

    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])

    logging.info("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" %
          (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))


    unique_uid = user_activity.index

    np.random.seed(98765)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    # create train/validation/test users
    n_users = unique_uid.size
    n_heldout_users = 10000

    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]

    unique_sid = pd.unique(train_plays['movieId'])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pro_dir = os.path.join(data_dir, 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)


    def split_train_test_proportion(data, test_prop=0.2):
        data_grouped_by_user = data.groupby('userId')
        tr_list, te_list = list(), list()

        np.random.seed(98765)

        for i, (_, group) in enumerate(data_grouped_by_user):
            n_items_u = len(group)

            if n_items_u >= 5:
                idx = np.zeros(n_items_u, dtype='bool')
                idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

                tr_list.append(group[np.logical_not(idx)])
                te_list.append(group[idx])
            else:
                tr_list.append(group)

            if i % 1000 == 0:
                logging.info("%d users sampled" % i)
                sys.stdout.flush()

        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)

        return data_tr, data_te

    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    def numerize(tp):
        uid = list(map(lambda x: profile2id[x], tp['userId']))
        sid = list(map(lambda x: show2id[x], tp['movieId']))
        return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

    train_data = numerize(train_plays)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_plays_te)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)


def main(argv):
    logging.info('Running with args %s', str(argv))
    download()
    process()


if __name__ == "__main__":
    app.run(main)
