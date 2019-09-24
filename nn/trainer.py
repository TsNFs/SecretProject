##############################################
# train part don't place it in the predictor #
##############################################
from nn.nn_model import NN
import tensorflow as tf

training_batch_size = 256
valid_batch_size = 256
embedding_size = 128
iteration = 100000
##

import pickle
import util.generate_batch as generator

if __name__ == '__main__':
    print("reading data from training set...")
    try:
        with open('./dump_data/nn/dump_train_x.txt', 'rb') as f:
            train_data_x = pickle.load(f)

        with open('./dump_data/nn/dump_train_y_label.txt', 'rb') as f:
            train_data_y = pickle.load(f)

        with open('./dump_data/nn/dump_valid_x.txt', 'rb') as f:
            valid_data_x = pickle.load(f)

        with open('./dump_data/nn/dump_valid_y_label.txt', 'rb') as f:
            valid_data_y = pickle.load(f)

        with open('./dump_data/nn/dump_test_x.txt', 'rb') as f:
            test_data_x = pickle.load(f)

        with open('./dump_data/nn/dump_test_y_label.txt', 'rb') as f:
            test_data_y = pickle.load(f)
    except:
        print("No dump file read original file! Please wait... "
              "If u want to accelerate this process, please see read_me -> transform_data_to_feature_and_dump")
        word_dict, embedding, reverse_dictionary = generator.get_dictionary_and_embedding()
        train_data_x, train_data_y = generator.build_data_to_x_y(embedding, word_dict)

    print("reading complete!")
    # just test generate_accu_batch
    x, y = generator.generate_batch(train_data_x, train_data_y)
    print(x.shape)

    print("data load complete")
    print("The model begin here")

    print(len(train_data_y[0]))

    model = NN()
    # run part
    with model.graph.as_default():
        with tf.Session() as sess:
            # 初始化变量
            sess.run(tf.global_variables_initializer())
            # 保存参数所用的保存器
            saver = tf.train.Saver(max_to_keep=1)
            # get latest file
            ckpt = tf.train.get_checkpoint_state('./xkf_nn_model')
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # 可视化部分
            tf.summary.scalar("loss", model.loss)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter("./xkf_nn_logs", sess.graph)

            # training part
            for i in range(iteration):
                x, y = generator.generate_batch(train_data_x, train_data_y)

                x_valid, y_valid = generator.generate_batch(train_data_x, train_data_y)
                # x_test, y_test = generator.generate_batch(training_batch_size, test_data_x, test_data_y)

                if i % 1000 == 0:
                    print("step:", i, "train:",
                          sess.run([model.loss], feed_dict={model.x: x, model.y: y, model.keep_prob: 1}))
                    # train_accuracy = sess.run(accuracy, feed_dict={xs: x, ys: y})
                    valid_x, valid_y = generator.generate_batch(train_data_x, train_data_y)
                    print("step:", "valid:",
                          sess.run([model.loss], feed_dict={model.x: valid_x, model.y: valid_y, model.keep_prob: 1}))
                    # valid_accuracy = sess.run(accuracy, feed_dict={xs: valid_x, ys: valid_y})
                    # print("step %d, training accuracy %g" % (i, train_accuracy))
                    # print("step %d, valid accuracy %g" % (i, valid_accuracy))
                    #
                    # y_label_result, y_true_result = sess.run([y_label, y_true], feed_dict={xs: valid_x, ys: valid_y})
                    # print("f1_score", sk.metrics.f1_score(y_label_result, y_true_result, average = "weighted"))
                    # exit(0)
                    # print(y_label)
                    # print(_index)

                    saver.save(sess, "./xkf_nn_model/nn", global_step=i)

                _, summary = sess.run([model.train_op, merged], feed_dict={model.x: x, model.y: y, model.keep_prob: 1})
                writer.add_summary(summary, i)
                _, summary = sess.run([model.train_op, merged],
                                      feed_dict={model.x: x_valid, model.y: y_valid, model.keep_prob: 1})
                writer.add_summary(summary, i)
                # _, summary = sess.run([model.train_op, merged], feed_dict={model.x: x_test, model.y: y_test, model.keep_prob: 1})
                # writer.add_summary(summary, i)
