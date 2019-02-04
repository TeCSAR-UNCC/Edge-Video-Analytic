# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]


def darknet53(inputs):
    """
    Builds Darknet-53 model.
    """
    inputs = _conv2d_fixed_padding(inputs, 32, 3)
    inputs = _conv2d_fixed_padding(inputs, 64, 3, strides=2)
    inputs = _darknet53_block(inputs, 32)
    inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2)

    for i in range(2):
        inputs = _darknet53_block(inputs, 64)

    inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 128)

    route_1 = inputs
    inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 256)

    route_2 = inputs
    inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2)

    for i in range(4):
        inputs = _darknet53_block(inputs, 512)

    return route_1, route_2, inputs


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def _darknet53_block(inputs, filters):
    shortcut = inputs
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)

    inputs = inputs + shortcut
    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.
    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if kwargs['data_format'] == 'NCHW':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]], mode='CONSTANT')
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
    return padded_inputs


def _yolo_block(inputs, filters):
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    route = inputs
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    return route, inputs


def _get_size(shape, data_format):
    if len(shape) == 4:
        shape = shape[1:]
    return shape[1:3] if data_format == 'NCHW' else shape[0:2]


def _detection_layer(inputs, num_classes, anchors, img_size, data_format):
    num_anchors = len(anchors)
    predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1, stride=1, normalizer_fn=None,
                              activation_fn=None, biases_initializer=tf.zeros_initializer())
    feature_map=predictions
    shape = predictions.get_shape().as_list()
    grid_size = _get_size(shape, data_format)
    #print(grid_size)
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes
    #print(shape)
    #print(inputs.shape)
    #print(predictions[:4])

    if data_format == 'NCHW':
        predictions = tf.reshape(predictions, [-1, num_anchors * bbox_attrs, dim])
        predictions = tf.transpose(predictions, [0, 2, 1])

    predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])
    #print(predictions[:4])

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])

    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)

    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))

    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    anchors = tf.tile(anchors, [dim, 1])
    anchors=tf.cast(anchors,tf.float32)
    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride

    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

    classes = tf.nn.sigmoid(classes)
    predictions = tf.concat([detections, classes], axis=-1)
    return predictions, feature_map


def _upsample(inputs, out_shape, data_format='NCHW'):
    # tf.image.resize_nearest_neighbor accepts input in format NHWC
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    if data_format == 'NCHW':
        new_height = out_shape[3]
        new_width = out_shape[2]
    else:
        new_height = out_shape[2]
        new_width = out_shape[1]

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    # back to NCHW if needed
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = tf.identity(inputs, name='upsampled')
    return inputs


def yolo_v3(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v3 model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], data_format=data_format, reuse=reuse):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
            with tf.variable_scope('darknet-53'):
                route_1, route_2, inputs = darknet53(inputs)
                darknet_output=inputs
            with tf.variable_scope('yolo-v3'):
                route, inputs = _yolo_block(inputs, 512)
                detect_1,feature_map1 = _detection_layer(inputs, num_classes, _ANCHORS[6:9], img_size, data_format)
                detect_1 = tf.identity(detect_1, name='detect_1')
        

                inputs = _conv2d_fixed_padding(route, 256, 1)
                upsample_size = route_2.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size, data_format)
                inputs = tf.concat([inputs, route_2], axis=1 if data_format == 'NCHW' else 3)

                route, inputs = _yolo_block(inputs, 256)

                detect_2,feature_map2 = _detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                detect_2 = tf.identity(detect_2, name='detect_2')

                inputs = _conv2d_fixed_padding(route, 128, 1)
                upsample_size = route_1.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size, data_format)
                inputs = tf.concat([inputs, route_1], axis=1 if data_format == 'NCHW' else 3)

                _, inputs = _yolo_block(inputs, 128)

                detect_3,feature_map3 = _detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
                detect_3 = tf.identity(detect_3, name='detect_3')

                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                detections = tf.identity(detections, name='detections')
                return [detect_1,detect_2,detect_3],[feature_map1,feature_map2,feature_map3]


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def detections_boxes(detections,feature_map):
    """
    Converts center x, center y, width and height values to coordinates of top left and bottom right points.
    :param detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
    :return: converted detections of same shape as input
    """
    box_det=[]
    for detection in detections:
        center_x, center_y, width, height, attrs = tf.split(detection, [1, 1, 1, 1, -1], axis=-1)
        w2 = width / 2
        h2 = height / 2
        x0 = center_x - w2
        y0 = center_y - h2
        x1 = center_x + w2
        y1 = center_y + h2
        #print(detections.shape)
        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        detection = tf.concat([boxes, attrs], axis=-1)
        box_det.append(detection)
    return box_det, feature_map


def _iou(box1, box2):
    """
    Computes Intersection over Union value for 2 bounding boxes
    
    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

def convert_to_original_size(box, size, original_size):
    x_scale=float(original_size[0])/float(size[0])
    y_scale=float(original_size[1])/float(size[1])
    #print(x_scale,y_scale)
    x_upper_left=int(np.round(box[0]*x_scale))
    y_upper_left=int(np.round(box[1]*y_scale))
    x_lower_right=int(np.round(box[2]*x_scale))
    y_lower_right=int(np.round(box[3]*y_scale))
    x_center=float(x_lower_right-x_upper_left)/2.0
    y_center=float(y_lower_right-y_upper_left)/2.0
    return [x_upper_left,y_upper_left,x_lower_right,y_lower_right]
    
def non_max_suppression(predictions_with_boxes,detection_size,original_size, confidence_threshold, iou_threshold=0.4):
    """
    Applies Non-max suppression to prediction boxes.
    :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    results=[]
    obj_num_frame=0
    for predictions_with_box in predictions_with_boxes:
        conf_mask = np.expand_dims((predictions_with_box[:, :, 4] > confidence_threshold), -1)
        predictions = predictions_with_box * conf_mask
    
        result = {}
       
        for i, image_pred in enumerate(predictions):
            shape = image_pred.shape
            non_zero_idxs = np.nonzero(image_pred)
            #print(image_pred[np.unique(non_zero_idxs[0]),:5])
            relevant_indices=np.unique(non_zero_idxs[0])
            #all_boxes_array=np.zeros([len(relevant_indices),86])
            all_boxes_array=image_pred[relevant_indices]
            #print(all_boxes_array.shape)
            all_boxes_array=np.column_stack([relevant_indices,all_boxes_array])
            #for i in range(len(relevant_indices)):
                #all_boxes_array[i][0]=relevant_indices[i]
                #all_boxes_array[i][1:]=image_pred[relevant_indices[i]]
            #print(all_boxes_array)
            
            #image_pred = image_pred[non_zero_idxs]
            
            #print(image_pred.shape)
            #print(shape)
            #print(non_zero_idxs)
            
            #image_pred = image_pred.reshape(-1, shape[-1])
            #print(relevant_indices)
            #print(image_pred.shape)
            bbox_attrs = all_boxes_array[:, :6]
            classes = all_boxes_array[:, 6:]
            
            classes = np.argmax(classes, axis=-1)
            #print(classes)
            
            unique_classes = list(set(classes.reshape(-1)))
            #print(unique_classes)
            for cls in unique_classes:
                cls_mask = classes == cls
                cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
                
                cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
                #print(cls_boxes)
                cls_scores = cls_boxes[:, -1]
                cl_boxes = cls_boxes[:, 1:5]
                box_idx=cls_boxes[:,0]
                #print(cls_scores)
                #print(cl_boxes)
                #print(box_idx)
    
                while len(cl_boxes) > 0:
                    box = cl_boxes[0]
                    score = cls_scores[0]
                    box_id=box_idx[0]
                    if not cls in result:
                        result[cls] = []
                    result[cls].append((convert_to_original_size(box,detection_size,original_size), score,box_id))
                    cl_boxes = cl_boxes[1:]
                    box_idx=box_idx[1:]
                    ious = np.array([_iou(box, x) for x in cl_boxes])
                    #print(ious)
                    iou_mask = ious < iou_threshold
                    #print(iou_mask)
                    cl_boxes = cl_boxes[np.nonzero(iou_mask)]
                    cls_scores = cls_scores[np.nonzero(iou_mask)]
                    box_idx=box_idx[np.nonzero(iou_mask)]
		    if cls==0:	
                    	obj_num_frame+=1
                    #print(box_idx)
                    #print(cls_scores)
                    #print(cls_boxes)
                    #print(result)
    #for i in range(len(result)):
        results.append(result)                

    return results,obj_num_frame

