from optparse import OptionParser
import tensorflow as tf
import time
import numpy as np
from datetime import datetime
import sys
import cv2
import os

from yolo.dataset.text_dataset import TextDataSet
from net.yolo_tiny_net import YoloTinyNet
from yolo.utils.Config import Config


classes_name = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


class Runner():

    def __init__(self, conf_file, data_set=TextDataSet, net=YoloTinyNet):

        common_params, dataset_params, net_params, solver_params = Config.process_config_for_train(conf_file)

        # data
        self.dataset = data_set(common_params, dataset_params)
        self.height = int(common_params['image_size'])
        self.width = int(common_params['image_size'])

        # net
        self.net = net(common_params, net_params)
        self.moment = float(solver_params['moment'])
        self.learning_rate = float(solver_params['learning_rate'])
        self.batch_size = int(common_params['batch_size'])
        self.max_objects = int(common_params['max_objects_per_image'])
        self.pretrain_path = str(solver_params['pretrain_model_path'])
        self.train_dir = str(solver_params['train_dir'])
        self.model_name = str(solver_params['model_name'])
        self.max_iterators = int(solver_params['max_iterators'])

        # construct graph
        self.construct_graph()

        # TODO:  很棒的实现， eval 好好研究一下
        # net = eval(net_params['name'])(common_params, net_params)
        # dataset = eval(dataset_params['name'])(common_params, dataset_params)
        # solver = eval(；solver_params['name'])(dataset, net, common_params, solver_params)
        pass

    # construct graph
    def construct_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 3))
        self.labels = tf.placeholder(tf.float32, (self.batch_size, self.max_objects, 5))
        self.objects_num = tf.placeholder(tf.int32, (self.batch_size))

        self.predicts = self.net.inference(self.images)
        self.total_loss, self.nilboy = self.net.loss(self.predicts, self.labels, self.objects_num)

        tf.summary.scalar('loss', self.total_loss)

        opt = tf.train.MomentumOptimizer(self.learning_rate, self.moment)
        self.train_op = opt.apply_gradients(opt.compute_gradients(self.total_loss), global_step=self.global_step)

    def solve(self):
        saver1 = tf.train.Saver(self.net.pretrained_collection, write_version=1)
        saver2 = tf.train.Saver(self.net.trainable_collection, write_version=1)

        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("load trained model")
                saver2.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("load pretrained model")
                saver1.restore(sess, self.pretrain_path)

            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

            for step in range(self.max_iterators):
                start_time = time.time()
                np_images, np_labels, np_objects_num = self.dataset.batch()

                _, loss_value, nilboy = sess.run([self.train_op, self.total_loss, self.nilboy],
                                                 feed_dict={self.images: np_images, self.labels: np_labels,
                                                            self.objects_num: np_objects_num})

                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 1000 == 0:
                    num_examples_per_step = self.dataset.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    print('{}: step {}, loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'
                          .format(datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

                    sys.stdout.flush()
                if step % 100 == 0:
                    summary_str = sess.run(summary_op, feed_dict={self.images: np_images, self.labels: np_labels,
                                                                  self.objects_num: np_objects_num})
                    summary_writer.add_summary(summary_str, step)
                if step % 10000 == 0:
                    saver2.save(sess, '{}/{}.ckpt'.format(self.train_dir, self.model_name), global_step=step)

    @staticmethod
    def process_predicts(predicts):
        # 每个grid有30维，这30维中，8维是回归box的坐标，2维是box的confidence，还有20维是类别
        p_classes = predicts[0, :, :, 0:20]
        box_confidence = predicts[0, :, :, 20:22]
        coordinate = predicts[0, :, :, 22:]

        p_classes = np.reshape(p_classes, (7, 7, 1, 20))
        box_confidence = np.reshape(box_confidence, (7, 7, 2, 1))

        P = box_confidence * p_classes

        index = np.argmax(P)
        # 如果index是每个维度对应的元素个数为p.shape的，那么它将是返回值的index维度的一个向量
        index = np.unravel_index(index, P.shape)

        class_num = index[3]

        coordinate = np.reshape(coordinate, (7, 7, 2, 4))

        max_coordinate = coordinate[index[0], index[1], index[2], :]

        xcenter = max_coordinate[0]
        ycenter = max_coordinate[1]
        w = max_coordinate[2]
        h = max_coordinate[3]

        xcenter = (index[1] + xcenter) * (448 / 7.0)
        ycenter = (index[0] + ycenter) * (448 / 7.0)

        w = w * 448
        h = h * 448

        xmin = xcenter - w / 2.0
        ymin = ycenter - h / 2.0

        xmax = xmin + w
        ymax = ymin + h

        return xmin, ymin, xmax, ymax, class_num

    @staticmethod
    def predict(test_files, conf_file, net=YoloTinyNet):

        # param
        common_params, net_params, solver_params, predict_params = Config.process_config_for_predict(conf_file)

        net = net(common_params, net_params, test=True)

        image_size = int(common_params["image_size"])
        image = tf.placeholder(tf.float32, (1, image_size, image_size, 3))
        predicts = net.inference(image)

        for test_file in test_files:

            np_img = cv2.imread(test_file)
            resized_img = cv2.resize(np_img, (image_size, image_size))
            np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

            np_img = np_img.astype(np.float32)

            np_img = np_img / 255.0 * 2 - 1
            np_img = np.reshape(np_img, (1, image_size, image_size, 3))

            saver = tf.train.Saver(net.trainable_collection)

            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(solver_params['train_dir'])
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                np_predict = sess.run(predicts, feed_dict={image: np_img})

                xmin, ymin, xmax, ymax, class_num = Runner.process_predicts(np_predict)
                class_name = classes_name[class_num]
                cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
                cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
                cv2.imwrite('{}/{}'.format(predict_params['output_dir'], os.path.split(test_file)[-1]), resized_img)

        pass

    def run(self):
        self.solve()
        pass

    pass



if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-c", "--conf", dest="configure", help="configure filename", default="../conf/train.cfg")
    (options, args) = parser.parse_args()
    conf_file = None
    if options.configure:
        conf_file = str(options.configure)
    else:
        print('please sspecify --conf configure filename')
        exit(0)

    runner = Runner(conf_file=conf_file)
    runner.run()

    # Runner.predict(['../data/cat.jpg'], conf_file)

    pass
