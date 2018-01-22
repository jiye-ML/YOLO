import tensorflow as tf
import numpy as np

from yolo.net.net import Net


class YoloTinyNet(Net):
    def __init__(self, common_params, net_params, test=False):
        super(YoloTinyNet, self).__init__(common_params, net_params)
        # data
        self.image_size = int(common_params['image_size'])
        self.num_classes = int(common_params['num_classes'])
        self.batch_size = int(common_params['batch_size'])
        self.cell_size = int(net_params['cell_size'])
        # net
        self.boxes_per_cell = int(net_params['boxes_per_cell'])
        self.weight_decay = float(net_params['weight_decay'])
        self.conv_num = 0

        if not test:
            # 用于 loss
            self.object_scale = float(net_params['object_scale'])
            self.noobject_scale = float(net_params['noobject_scale'])
            self.class_scale = float(net_params['class_scale'])
            self.coord_scale = float(net_params['coord_scale'])

    def inference(self, images):
        """Build the yolo model
    
        Args:
          images:  4-D tensor [batch_size, image_height, image_width, channels]
        Returns:
          predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        """
        temp_conv = self.conv2d(images, [3, 3, 3, 16], stride=1) # 448x448x16
        temp_pool = self.max_pool(temp_conv, [2, 2], 2) # 224x224x16
        temp_conv = self.conv2d(temp_pool, [3, 3, 16, 32], stride=1) # 224x224x32
        temp_pool = self.max_pool(temp_conv, [2, 2], 2) # 112x112x32
        temp_conv = self.conv2d(temp_pool, [3, 3, 32, 64], stride=1) # 112x112x64
        temp_conv = self.max_pool(temp_conv, [2, 2], 2) # 56x56x64
        temp_conv = self.conv2d(temp_conv, [3, 3, 64, 128], stride=1) #56x56x128
        temp_conv = self.max_pool(temp_conv, [2, 2], 2) # 28x28x128
        temp_conv = self.conv2d(temp_conv, [3, 3, 128, 256], stride=1) # 28x28x256
        temp_conv = self.max_pool(temp_conv, [2, 2], 2) # 14x14x256
        temp_conv = self.conv2d(temp_conv,[3, 3, 256, 512], stride=1) # 14x14x512
        temp_conv = self.max_pool(temp_conv, [2, 2], 2) #7x7x512
        temp_conv = self.conv2d(temp_conv,[3, 3, 512, 1024], stride=1) # 7x7x1024
        temp_conv = self.conv2d(temp_conv,[3, 3, 1024, 1024], stride=1) # 7x7x1024
        temp_conv = self.conv2d(temp_conv, [3, 3, 1024, 1024], stride=1) # 7x7x1024
        temp_conv = tf.transpose(temp_conv, (0, 3, 1, 2)) # 1x1024x7x7

        # Fully connected layer
        local1 = self.fully_connection('local1', temp_conv, self.cell_size * self.cell_size * 1024, 256) # 1x256
        local2 = self.fully_connection('local2', local1, 256, 4096) # 1x4096
        local3 = self.fully_connection('local3', local2, 4096,
                            self.cell_size * self.cell_size * (self.num_classes + self.boxes_per_cell * 5), leaky=False,
                            pretrain=False, train=True) # 1x1470(7x7x(20+2*5))

        # 每个grid有30维，这30维中，8维是回归box的坐标，2维是box的confidence，还有20维是类别
        n1 = self.cell_size * self.cell_size * self.num_classes # 980=7x7x20
        n2 = n1 + self.cell_size * self.cell_size * self.boxes_per_cell # 1078=7x7x2
        class_probs = tf.reshape(local3[:, 0:n1], (-1, self.cell_size, self.cell_size, self.num_classes)) # 1x7x7x20
        scales = tf.reshape(local3[:, n1:n2], (-1, self.cell_size, self.cell_size, self.boxes_per_cell)) # 1x7x7x2
        boxes = tf.reshape(local3[:, n2:], (-1, self.cell_size, self.cell_size, self.boxes_per_cell * 4)) # 1x7x7x8

        return tf.concat([class_probs, scales, boxes], 3) # 1x7x7x30

    def iou(self, boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                           boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
        boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                           boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])

        # calculate the left up point
        lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
        rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

        # intersection
        intersection = rd - lu

        inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

        mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

        inter_square = mask * inter_square

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
        square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        return inter_square / (square1 + square2 - inter_square + 1e-6)

    def cond1(self, num, object_num, loss, predict, label, nilboy):
        return num < object_num

    def body1(self, num, object_num, loss, predict, labels, nilboy):
        """
        calculate loss
        Args:
          predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
          labels : [max_objects, 5]  (x_center, y_center, w, h, class)
        """
        # 同一个格子预测num个box， 但预测属于同一类
        label = labels[num:num + 1, :]
        label = tf.reshape(label, [-1])

        # calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        min_x = (label[0] - label[2] / 2) / (self.image_size / self.cell_size)
        max_x = (label[0] + label[2] / 2) / (self.image_size / self.cell_size)

        min_y = (label[1] - label[3] / 2) / (self.image_size / self.cell_size)
        max_y = (label[1] + label[3] / 2) / (self.image_size / self.cell_size)

        min_x = tf.floor(min_x)
        min_y = tf.floor(min_y)

        max_x = tf.ceil(max_x)
        max_y = tf.ceil(max_y)

        temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
        objects = tf.ones(temp, tf.float32)

        temp = tf.cast(tf.stack([min_y, self.cell_size - max_y, min_x, self.cell_size - max_x]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        objects = tf.pad(objects, temp, "CONSTANT")

        # calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        # calculate responsible tensor [CELL_SIZE, CELL_SIZE]
        center_x = label[0] / (self.image_size / self.cell_size)
        center_x = tf.floor(center_x)

        center_y = label[1] / (self.image_size / self.cell_size)
        center_y = tf.floor(center_y)

        response = tf.ones([1, 1], tf.float32)

        temp = tf.cast(tf.stack([center_y, self.cell_size - center_y - 1, center_x, self.cell_size - center_x - 1]),tf.int32)
        temp = tf.reshape(temp, (2, 2))
        response = tf.pad(response, temp, "CONSTANT")
        # objects = response

        # calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        predict_boxes = predict[:, :, self.num_classes + self.boxes_per_cell:]

        predict_boxes = tf.reshape(predict_boxes, [self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        predict_boxes = predict_boxes * [self.image_size / self.cell_size, self.image_size / self.cell_size,
                                         self.image_size, self.image_size]

        base_boxes = np.zeros([self.cell_size, self.cell_size, 4])

        for y in range(self.cell_size):
            for x in range(self.cell_size):
                # nilboy
                base_boxes[y, x, :] = [self.image_size / self.cell_size * x, self.image_size / self.cell_size * y, 0, 0]
        base_boxes = np.tile(np.resize(base_boxes, [self.cell_size, self.cell_size, 1, 4]),
                             [1, 1, self.boxes_per_cell, 1])

        predict_boxes = base_boxes + predict_boxes

        iou_predict_truth = self.iou(predict_boxes, label[0:4])
        # calculate C [cell_size, cell_size, boxes_per_cell]
        C = iou_predict_truth * tf.reshape(response, [self.cell_size, self.cell_size, 1])

        # calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        I = iou_predict_truth * tf.reshape(response, (self.cell_size, self.cell_size, 1))

        max_I = tf.reduce_max(I, 2, keep_dims=True)

        I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (self.cell_size, self.cell_size, 1))

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        no_I = tf.ones_like(I, dtype=tf.float32) - I

        p_C = predict[:, :, self.num_classes:self.num_classes + self.boxes_per_cell]

        # calculate truth x,y,sqrt_w,sqrt_h 0-D
        x = label[0]
        y = label[1]

        sqrt_w = tf.sqrt(tf.abs(label[2]))
        sqrt_h = tf.sqrt(tf.abs(label[3]))
        # sqrt_w = tf.abs(label[2])
        # sqrt_h = tf.abs(label[3])

        # calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        p_x = predict_boxes[:, :, :, 0]
        p_y = predict_boxes[:, :, :, 1]

        # p_sqrt_w = tf.sqrt(tf.abs(predict_boxes[:, :, :, 2])) * ((tf.cast(predict_boxes[:, :, :, 2] > 0, tf.float32) * 2) - 1)
        # p_sqrt_h = tf.sqrt(tf.abs(predict_boxes[:, :, :, 3])) * ((tf.cast(predict_boxes[:, :, :, 3] > 0, tf.float32) * 2) - 1)
        # p_sqrt_w = tf.sqrt(tf.maximum(0.0, predict_boxes[:, :, :, 2]))
        # p_sqrt_h = tf.sqrt(tf.maximum(0.0, predict_boxes[:, :, :, 3]))
        # p_sqrt_w = predict_boxes[:, :, :, 2]
        # p_sqrt_h = predict_boxes[:, :, :, 3]
        p_sqrt_w = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(tf.minimum(self.image_size * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
        # calculate truth p 1-D tensor [NUM_CLASSES]
        P = tf.one_hot(tf.cast(label[4], tf.int32), self.num_classes, dtype=tf.float32)

        # calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
        p_P = predict[:, :, 0:self.num_classes]

        # class_loss
        class_loss = tf.nn.l2_loss(
            tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale
        # class_loss = tf.nn.l2_loss(tf.reshape(response, (self.cell_size, self.cell_size, 1)) * (p_P - P)) * self.class_scale

        # object_loss
        object_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale
        # object_loss = tf.nn.l2_loss(I * (p_C - (C + 1.0)/2.0)) * self.object_scale

        # noobject_loss
        # noobject_loss = tf.nn.l2_loss(no_I * (p_C - C)) * self.noobject_scale
        noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale

        # coord_loss
        coord_loss = (tf.nn.l2_loss(I * (p_x - x) / (self.image_size / self.cell_size)) +
                      tf.nn.l2_loss(I * (p_y - y) / (self.image_size / self.cell_size)) +
                      tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w)) / self.image_size +
                      tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h)) / self.image_size) * self.coord_scale

        nilboy = I

        return num + 1, object_num, [loss[0] + class_loss, loss[1] + object_loss, loss[2] + noobject_loss,
                                     loss[3] + coord_loss], predict, labels, nilboy

    def loss(self, predicts, labels, objects_num):
        """Add Loss to all the trainable variables
    
        Args:
          predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
          ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
          labels  : 3-D tensor of [batch_size, max_objects, 5]
          objects_num: 1-D tensor [batch_size]
        """
        class_loss = tf.constant(0, tf.float32)
        object_loss = tf.constant(0, tf.float32)
        noobject_loss = tf.constant(0, tf.float32)
        coord_loss = tf.constant(0, tf.float32)
        loss = [0, 0, 0, 0]
        for i in range(self.batch_size):
            predict = predicts[i, :, :, :]
            label = labels[i, :, :]
            object_num = objects_num[i]
            nilboy = tf.ones([7, 7, 2])
            tuple_results = tf.while_loop(self.cond1, self.body1, [tf.constant(0), object_num,
                                                                   [class_loss, object_loss, noobject_loss, coord_loss],
                                                                   predict, label, nilboy])
            for j in range(4):
                loss[j] = loss[j] + tuple_results[2][j]
            nilboy = tuple_results[5]

        tf.add_to_collection('losses', (loss[0] + loss[1] + loss[2] + loss[3]) / self.batch_size)

        tf.summary.scalar('class_loss', loss[0] / self.batch_size)
        tf.summary.scalar('object_loss', loss[1] / self.batch_size)
        tf.summary.scalar('noobject_loss', loss[2] / self.batch_size)
        tf.summary.scalar('coord_loss', loss[3] / self.batch_size)
        tf.summary.scalar('weight_loss', tf.add_n(tf.get_collection('losses')) - (loss[0] + loss[1] + loss[2] + loss[3]) / self.batch_size)

        return tf.add_n(tf.get_collection('losses'), name='total_loss'), nilboy
