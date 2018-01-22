import random
import cv2
import numpy as np
from queue import Queue
from threading import Thread

from yolo.dataset.dataset import DataSet


# text file format: image_path xmin1 ymin1 xmax1 ymax1 class1 xmin2 ymin2 xmax2 ymax2 class2
class TextDataSet(DataSet):

    def __init__(self, common_params, dataset_params):
        # data
        self.data_path = str(dataset_params['path'])
        self.width = int(common_params['image_size'])
        self.height = int(common_params['image_size'])
        self.batch_size = int(common_params['batch_size'])
        self.thread_num = int(dataset_params['thread_num'])
        self.max_objects = int(common_params['max_objects_per_image'])

        # record and image_label queue
        self.record_queue = Queue(maxsize=10000)
        self.image_label_queue = Queue(maxsize=512)

        self.record_list = self._read_records()

        self.record_point = 0
        self.record_number = len(self.record_list)

        self.num_batch_per_epoch = int(self.record_number / self.batch_size)

        # 生产者
        t_record_producer = Thread(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()
        # 消费者
        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start()

    def _read_records(self):
        record_list = []
        with open(self.data_path, 'r') as input_file:
            for line in input_file:
                ss = line.strip().split(' ')
                ss[1:] = [float(num) for num in ss[1:]]
                record_list.append(ss)
        return record_list
        pass

    def record_producer(self):
        while True:
            if self.record_point % self.record_number == 0:
                random.shuffle(self.record_list)
                self.record_point = 0
            self.record_queue.put(self.record_list[self.record_point])
            self.record_point += 1

    # 一张图片和对应的标签
    def record_process(self, record):
        image = cv2.imread(record[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        width_rate = self.width * 1.0 / image.shape[1]
        height_rate = self.height * 1.0 / image.shape[0]

        image = cv2.resize(image, (self.height, self.width))
        # labels: 2-D list [self.max_objects, 5] (xcenter, ycenter, w, h, class_num)
        labels = [[0, 0, 0, 0, 0]] * self.max_objects
        i = 1
        object_num = 0
        while i < len(record):
            xmin = record[i]
            ymin = record[i + 1]
            xmax = record[i + 2]
            ymax = record[i + 3]
            class_num = record[i + 4]

            xcenter = (xmin + xmax) * 1.0 / 2 * width_rate
            ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

            box_w = (xmax - xmin) * width_rate
            box_h = (ymax - ymin) * height_rate

            labels[object_num] = [xcenter, ycenter, box_w, box_h, class_num]
            object_num += 1
            i += 5
            if object_num >= self.max_objects:
                break
        return [image, labels, object_num]

    # record queue's customer
    def record_customer(self):
        while True:
            item = self.record_queue.get()
            out = self.record_process(item)
            self.image_label_queue.put(out)

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 3]
          labels: 3-D ndarray [batch_size, max_objects, 5]
          objects_num: 1-D ndarray [batch_size]
        """
        images = []
        labels = []
        objects_num = []
        for i in range(self.batch_size):
            image, label, object_num = self.image_label_queue.get()
            images.append(image)
            labels.append(label)
            objects_num.append(object_num)
        images = np.asarray(images, dtype=np.float32)
        images = images / 255 * 2 - 1
        labels = np.asarray(labels, dtype=np.float32)
        objects_num = np.asarray(objects_num, dtype=np.int32)
        return images, labels, objects_num
