import tensorflow as tf
import numpy as np
import samples.utils.tensorflow_yolo_ext as yolo_helper
from samples.utils.box_convert import bbox_to_anbox


class TINY_YOLO_v2():
    anchors = [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]

    def __init__(self, height=416, width=416, channels=3, num_classes=80, output_op_name="output", lr=1e-3,
                 momentum=0.9, threshold=0.6, decay=0.0005, is_train=False, format="NCHW", batch_size=10,
                 true_boxes_num=30):
        if format == "NCHW":
            self.input_shape = [None, channels, height, width]
        elif format == "NHWC":
            self.input_shape = [None, height, width, channels]
        else:
            self.input_shape = [None]
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.output_size = (num_classes + 5) * 5
        self.output_shape = [None, (num_classes + 5) * 5]
        self.output_op_name = output_op_name
        self.block_size = 32
        self.grid_w = width // self.block_size
        self.grid_h = height // self.block_size
        self.batch_size = batch_size
        self.true_boxes_num = true_boxes_num

        self.COORD = 1.0
        self.NO_COORD = 0.1
        self.OBJ = 5.0
        self.NO_OBJ = 1.0
        self.CLASS = 1.0

        self.threshold =threshold
        self.num_anchor = len(self.anchors)

        self.lr = lr
        self.momentum = momentum
        self.decay = decay

        tiny_yolo2_graph = tf.Graph()

        with tiny_yolo2_graph.as_default():
            inputs = tf.placeholder(tf.float32, self.input_shape, name="inputs")
            conv_1 = yolo_helper.darknetconv2d(inputs, output_size=16, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_1 = yolo_helper.darknetpool(conv_1, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, name="maxpool")
            conv_2 = yolo_helper.darknetconv2d(pool_1, output_size=32, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_2 = yolo_helper.darknetpool(conv_2, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, name="maxpool")
            conv_3 = yolo_helper.darknetconv2d(pool_2, output_size=64, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_3 = yolo_helper.darknetpool(conv_3, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, name="maxpool")
            conv_4 = yolo_helper.darknetconv2d(pool_3, output_size=128, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_4 = yolo_helper.darknetpool(conv_4, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, name="maxpool")
            conv_5 = yolo_helper.darknetconv2d(pool_4, output_size=256, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_5 = yolo_helper.darknetpool(conv_5, h_kernel=2, w_kernel=2, h_stride=2, w_stride=2, name="maxpool")
            conv_6 = yolo_helper.darknetconv2d(pool_5, output_size=512, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            pool_6 = yolo_helper.darknetpool(conv_6, h_kernel=2, w_kernel=2, h_stride=1, w_stride=1, name="maxpool")
            conv_7 = yolo_helper.darknetconv2d(pool_6, output_size=1024, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            conv_8 = yolo_helper.darknetconv2d(conv_7, output_size=512, h_kernel=3, w_kernel=3, h_stride=1, w_stride=1,
                                               activation="leaky_relu", batch_normalization=True, name="convolutional")
            output = yolo_helper.darknetconv2d(conv_8, output_size=self.output_size, h_kernel=1, w_kernel=1, h_stride=1,
                                               w_stride=1, activation="linear", batch_normalization=True,
                                               name=self.output_op_name)

            if is_train:
                with tf.name_scope("train"):
                    y_true = tf.placeholder(tf.float32,
                                            [None, self.grid_h, self.grid_w, self.num_anchor, self.num_classes + 5],
                                            name="y_true")
                    y_mask = tf.placeholder(tf.float32, [None, self.grid_h, self.grid_w, self.num_anchor, 1])
                    y_true_nogrid = tf.placeholder(tf.float32, [None, self.true_boxes_num, 4])
                    total_loss, object_loss, coord_loss, class_loss = self.loss(output, y_true, y_mask, y_true_nogrid)

                    tf.losses.add_loss(class_loss)
                    tf.losses.add_loss(object_loss)
                    tf.losses.add_loss(coord_loss)

                    self.class_loss = class_loss
                    self.object_loss = object_loss
                    self.coord_loss = coord_loss
                    self.global_step = tf.train.create_global_step()
                    self.total_loss = tf.losses.get_total_loss()
                    self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, 100000, 0.96,
                                                                    staircase=True)
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                                momentum=self.momentum)
                    self.minimize = self.optimizer.minimize(loss=self.total_loss)
                    self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    self.train_op = tf.group([self.minimize, self.extra_update_ops])
                    self.y_true = y_true
                    self.y_mask = y_mask
                    self.y_true_nogrid = y_true_nogrid
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            pb_saver = tf.train
            init = tf.global_variables_initializer()

        self.graph = tiny_yolo2_graph
        self.inputs = inputs
        self.output = output
        self.init = init
        self.saver = saver
        self.pb_saver = pb_saver

    def loss(self, y_pred, y_adjust_true, y_mask, y_true_not_grid):
        y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 1])
        y_pred_shape = tf.shape(y_pred)

        batch_size = y_pred_shape[0]

        yx_offset = self.get_offset_yx(grid_w=self.grid_w, grid_h=self.grid_h)

        y_pred = tf.reshape(y_pred, [batch_size, self.grid_h, self.grid_w, self.num_anchor, self.num_classes + 5])
        # pred feature
        xy_pred, wh_pred, conf_pred, boxes_classes_pred = tf.split(y_pred, [2, 2, 1, self.num_classes], axis=-1)
        xy_pred_sigmoid = tf.sigmoid(xy_pred)
        xy_pred_box = (xy_pred_sigmoid + yx_offset) / [self.grid_w, self.grid_h]
        wh_pred_box = (tf.exp(wh_pred) * self.anchors) / [self.grid_w, self.grid_h]
        boxes_pred = tf.concat((xy_pred_box, wh_pred_box), axis=-1)
        conf_pred = tf.sigmoid(conf_pred)

        # true label
        boxes_true, conf_true, classes_true = tf.split(y_adjust_true, [4, 1, self.num_classes], axis=-1)
        xy_true = boxes_true[..., 0:2]
        wh_true_box = boxes_true[..., 2:4]
        wh_true_box = wh_true_box / [self.grid_w, self.grid_h]
        # get ground truth no obj mask
        not_grid_iou = self.ground_truth_iou_batch(boxes_pred, y_true_not_grid)
        not_grid_iou = tf.expand_dims(not_grid_iou, axis=-1)
        obj_mask = tf.to_float(not_grid_iou > self.threshold)
        no_obj_mask = 1 - obj_mask

        # get shift best iou and truth object item
        shift_iou = self.box_iou_shift(wh_pred_box, wh_true_box)
        best_shift_iou = tf.argmax(shift_iou, axis=-1)
        obj_detector_mask = tf.to_float(tf.one_hot(best_shift_iou, depth=self.num_anchor, axis=-1))
        obj_detector_mask = tf.expand_dims(obj_detector_mask, axis=-1) * y_mask


        no_obj_detector_mask = 1 - obj_detector_mask
        # obj_mask = y_mask

        no_obj_delta = self.NO_OBJ * no_obj_mask * no_obj_detector_mask * tf.square(0 - conf_pred)
        obj_delta = self.OBJ * obj_detector_mask * tf.square(1 - conf_pred)

        full_obj_delta = obj_delta + no_obj_delta
        obj_loss = tf.reduce_sum(full_obj_delta)

        xy_delta = (tf.square(obj_detector_mask * (xy_pred_sigmoid - xy_true))) * self.COORD
        wh_delta = (tf.square(obj_detector_mask * (wh_pred_box - wh_true_box))) * self.COORD

        xy_loss = tf.reduce_sum(xy_delta)
        wh_loss = tf.reduce_sum(wh_delta)
        coord_loss = xy_loss + wh_loss

        obj_class_pred = obj_detector_mask * boxes_classes_pred
        obj_class_true = obj_detector_mask * classes_true

        class_loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=obj_class_true, logits=obj_class_pred)) * self.CLASS

        total_loss = obj_loss + coord_loss + class_loss

        return total_loss, obj_loss, coord_loss, class_loss

    def box_iou(self, boxes_pred, boxes_true):

        b1_xy = boxes_pred[..., :2]
        b1_wh = boxes_pred[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]

        b2_xy = boxes_true[..., :2]
        b2_wh = boxes_true[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]

        intersect_mins = tf.maximum(b1_mins, b2_mins)
        intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        iou = tf.divide(intersect_area, (b1_area + b2_area - intersect_area))
        return iou

    def box_iou_shift(self, box_wh_pred, box_wh_true):

        pred_wh_half = tf.divide(box_wh_pred, 2)
        pred_wh_maxes = pred_wh_half
        pred_wh_mins = -pred_wh_half
        pred_area = box_wh_pred[..., 0] * box_wh_pred[..., 1]

        true_wh_half = tf.divide(box_wh_true, 2)
        true_wh_maxes = true_wh_half
        true_wh_mins = -true_wh_half
        true_area = box_wh_true[..., 0] * box_wh_true[..., 1]

        intersect_mins = tf.maximum(pred_wh_mins, true_wh_mins)
        intersect_maxes = tf.minimum(pred_wh_maxes, true_wh_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        iou = tf.divide(intersect_area, (pred_area + true_area - intersect_area))
        return iou

    def ground_truth_iou_batch(self, pred_boxes_batch, true_boxes_batch):
        ever_iou = tf.map_fn(lambda x: self.ground_truth_iou(x[0], x[1]), (pred_boxes_batch, true_boxes_batch),
                             dtype=tf.float32)
        return ever_iou

    def ground_truth_iou(self, pred_boxes, true_boxes):
        def cal_true_boxes_iou(true_box):
            pred_xy = pred_boxes[..., :2]
            pred_wh = pred_boxes[..., 2:4]
            perd_wh_half = pred_wh / 2
            pred_maxes = pred_xy + perd_wh_half
            pred_mins = pred_xy - perd_wh_half
            pred_area = pred_wh[..., 0] * pred_wh[..., 1]

            true_xy = true_box[..., :2]
            true_wh = true_box[..., 2:4]
            true_wh_half = true_wh / 2
            true_maxes = true_xy + true_wh_half
            true_mins = true_xy - true_wh_half
            true_area = pred_wh[..., 0] * pred_wh[..., 1]

            intersect_mins = tf.maximum(pred_mins, true_mins)
            intersect_maxes = tf.minimum(pred_maxes, true_maxes)
            intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

            iou = tf.divide(intersect_area, (pred_area + true_area - intersect_area))
            return iou

        ground_truth_iou = tf.map_fn(cal_true_boxes_iou, true_boxes)
        ground_truth_best_iou = tf.reduce_max(ground_truth_iou, axis=0)
        return ground_truth_best_iou

    def pre_process_true_boxes_normal(self, label_list, grid_w, grid_h):
        y_true_grid = []
        y_mask = []
        y_true_no_grid = []
        for labels in label_list:
            true_box_grid = np.zeros([grid_h, grid_w, self.num_anchor, self.num_classes + 5], dtype=np.float32)
            true_box_mask = np.zeros([grid_h, grid_w, self.num_anchor, 1], dtype=np.float32)
            true_box_no_grid = np.zeros([self.true_boxes_num, 4])
            for label_index, label in enumerate(labels):
                if label_index < self.true_boxes_num:
                    bbox = label[0:4]
                    classes = label[4:]
                    anbox = bbox_to_anbox(bbox)  # [x1 y1 x2 y2] to [x y w h]
                    anbox_grid = anbox * np.array([grid_w, grid_h, grid_w, grid_h])
                    i = np.floor(anbox_grid[0]).astype(np.int)
                    j = np.floor(anbox_grid[1]).astype(np.int)

                    if not(i < grid_w and j < grid_h):
                        continue
                    
                    true_box_no_grid[label_index] = anbox

                    for k, anchor in enumerate(self.anchors):
                            true_box_mask[j, i, k, :] = 1
                            shift_true_box = [anbox_grid[0] - i, anbox_grid[1] - j, anbox_grid[2], anbox_grid[3]]
                            true_box_grid[j, i, k, 0:4] = shift_true_box
                            true_box_grid[j, i, k, 4] = 1
                            true_box_grid[j, i, k, 5:] = classes
                else:
                    pass

            y_true_grid.append(true_box_grid)
            y_mask.append(true_box_mask)
            y_true_no_grid.append(true_box_no_grid)
        return y_true_grid, y_mask, y_true_no_grid

    def get_offset_xy(self, grid_w, grid_h):
        grid_x = np.arange(grid_w)
        grid_y = np.arange(grid_h)
        x, y = np.meshgrid(grid_x, grid_y)
        x = np.reshape(x, (grid_w, grid_h, -1))
        y = np.reshape(y, (grid_w, grid_h, -1))
        x_y_offset = np.concatenate((x, y), -1)
        x_y_offset = np.reshape(x_y_offset, [grid_w, grid_h, 1, 2])
        return x_y_offset

    def get_offset_yx(self, grid_h, grid_w):
        grid_x = np.arange(grid_w)
        grid_y = np.arange(grid_h)
        x, y = np.meshgrid(grid_y, grid_x)
        x = np.reshape(x, (grid_h, grid_w, -1))
        y = np.reshape(y, (grid_h, grid_w, -1))
        x_y_offset = np.concatenate((y, x), -1)
        x_y_offset = np.reshape(x_y_offset, [grid_h, grid_w, 1, 2])
        return x_y_offset
