"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, concatenate, MaxPooling2D, Activation
from keras.layers import LeakyReLU, Softmax, ReLU #.advanced_activations
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from eval.predict.yolo3.utils import compose

from keras.preprocessing import image

from keras.layers import Input, Convolution2D, Dropout, GlobalAveragePooling2D # RandomNormal,



@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])



def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3))
            )(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1))
            )(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1))
            )(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2)
            )(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1))
            )([x2,x1])

    return Model(inputs, [y1,y2])

# a building block of the SqueezeNet architecture
def fire_module(number, x, squeeze, expand, weight_decay=None, trainable=True):
    
    module_name = 'fire' + number
    
    if trainable and weight_decay is not None:
        kernel_regularizer = l2(weight_decay) 
    else:
        kernel_regularizer = None
    
    x = Convolution2D(
        squeeze, (1, 1), 
        name=module_name + '/' + 'squeeze',
        trainable=trainable, 
        kernel_regularizer=kernel_regularizer
    )(x)
    x = Activation('relu')(x)

    a = Convolution2D(
        expand, (1, 1),
        name=module_name + '/' + 'expand1x1',
        trainable=trainable, 
        kernel_regularizer=kernel_regularizer
    )(x)
    a = Activation('relu')(a)

    b = Convolution2D(
        expand, (3, 3), padding='same',
        name=module_name + '/' + 'expand3x3',
        trainable=trainable, 
        kernel_regularizer=kernel_regularizer
    )(x)
    b = Activation('relu')(b)

    return concatenate([a, b])


def squeezenet_body(weight_decay=1e-4, input_tensor=Input(shape=(416, 416, 3))):

    image = input_tensor 

    x = ZeroPadding2D(((1,0),(1,0)))(image) #417, 417, 3

    x = Convolution2D(
        64, (3, 3), strides=(2, 2), name='conv1', 
        trainable=True
    )(x) #  208, 208, 64
    
    x = Activation('relu')(x)
    x = ZeroPadding2D(((1,0),(1,0)))(x) # 209, 209, 64
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 104, 104, 64

    x = fire_module('2', x, squeeze=16, expand=64) # 104, 104, 128
    x = fire_module('3', x, squeeze=16, expand=64) # 104, 104, 128
    x = ZeroPadding2D(((1,0),(1,0)))(x) # 105, 105, 128
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 52, 52, 128

    x = fire_module('4', x, squeeze=32, expand=128) # 52, 52, 256
    x = fire_module('5', x, squeeze=32, expand=128) # 52, 52, 256
    x = ZeroPadding2D(((1,0),(1,0)))(x) # 53, 53, 256
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 26, 26, 256
    
    x = fire_module('6', x, squeeze=48, expand=192) # 26, 26, 384
    x = fire_module('7', x, squeeze=48, expand=192) # 26, 26, 384
    x = ZeroPadding2D(((1,0),(1,0)))(x) # 27, 27, 384
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 13, 13, 384

    x = fire_module('8', x, squeeze=64, expand=256) # 13, 13, 512
    x = fire_module('9', x, squeeze=64, expand=256) # 13, 13, 512
    x = fire_module('10', x, squeeze=128, expand=320) # 13, 13, 640
    x = fire_module('11', x, squeeze=128, expand=320) # 13, 13, 640
    
    '''
    x = Dropout(0.5)(x)
    x = Convolution2D(
        320, (1, 1), name='conv10',
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    )(x) # 13, 13, 256
    '''
    #x = Activation('relu')(x)
    #logits = GlobalAveragePooling2D()(x) # 256
    #probabilities = Activation('softmax')(logits)
    
    model = Model(image, x )
       # load weights
    #weights_path = get_file(
    #    'squeezenet_weights.h5',
    #    TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models'
    #)
    #model.load_weights(weights_path, skip_mismatch=True)
    
    return model

def squeezenet_yolo_body(inputs, num_anchors, num_classes):
    #net, endpoint = inception_v2.inception_v2(inputs)
    squeezenet = squeezenet_body(input_tensor=inputs)

    # input: 416 x 416 x 3
    # contatenate_10 :13 x 13 x 640
    # contatenate_8 :13 x 13 x 512
    # contatenate_6 :26 x 26 x 384
    # contatenate_4 : 52 x 52 x 256

    f1 = squeezenet.get_layer('concatenate_10').output
    # f1 :13 x 13 x 640
    x, y1 = make_last_layers(f1, 512, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f2 = squeezenet.get_layer('concatenate_6').output
    # f2: 26 x 26 x 384
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)

    f3 = squeezenet.get_layer('concatenate_4').output
    # f3 : 52 x 52 x 256
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              model_name=None):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_yolo_heads = 3 #yolo_outputs.shape[0] #len(yolo_outputs)
    if 'tiny' in model_name:
        num_yolo_heads -= 1
    #elif model_name in ['tiny_yolo_infusion_hydra']:
    #    num_yolo_heads -= 2

    # anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_yolo_heads==3 else [[3,4,5], [1,2,3]] # default setting -> # BUG
    num_anchors_per_head = len(anchors) // num_yolo_heads
    anchor_mask = np.arange(len(anchors)).reshape(-1,num_anchors_per_head)[::-1] # dynamic mask

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_yolo_heads):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes, model_name=None, num_yolo_heads=None):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    # num_layers = len(anchors)//3 # default setting #TO-DO #TO-FIX
    # anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # BUG
    num_anchors_per_head = len(anchors) // num_yolo_heads
    anchor_mask = np.arange(len(anchors)).reshape(-1,num_anchors_per_head)[::-1].tolist() # dynamic mask

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    if model_name == 'yolo_small_objs':
        grid_shapes = [input_shape//{0:32, 1:16, 2:4}[l] for l in range(num_yolo_heads)] #small-objs
    elif model_name == 'tiny_yolo_small_objs':
        grid_shapes = [input_shape//{0:32, 1:8}[l] for l in range(num_yolo_heads)] #small-objs
    else:
        grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_yolo_heads)]

    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_yolo_heads)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_yolo_heads):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, model_name=None, num_yolo_heads=None, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_yolo_heads = len(anchors)//3 # 3 -> default number of anchors per cell
    num_outputs = num_yolo_heads
    #old style
    # if model_name in ['tiny_yolo_infusion','yolo_infusion']:
    #     #add segmentation output layer.
    #     num_outputs = num_heads + 1
    # elif model_name in ['tiny_yolo_infusion_hydra']:
    #     #add segmentation output layer.
    #     num_outputs = num_heads + 2
    #new style: keep commented.

    #args => head_a, head_b, seg, input_a, input_b
    yolo_outputs = args[:num_outputs] #head_a, head_b, seg_output
    y_true = args[num_outputs:] #input_a,input_b, input_seg

    # anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_heads==3 else [[3,4,5], [1,2,3]] # -> BUG
    # anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_heads==3 else [[3,4,5], [0,1,2]]
    num_anchors_per_head = len(anchors) // num_yolo_heads
    anchor_mask = np.arange(len(anchors)).reshape(-1,num_anchors_per_head)[::-1] # dynamic mask

    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_yolo_heads)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))
    # print('k.eval',K.eval(mf))

    for l in range(num_yolo_heads):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')

    #old style
    # if model_name in ['tiny_yolo_infusion','yolo_infusion']:
    #     #calc seg loss
    #     raw_true_seg = y_true[num_outputs-1] #seg is always the last output.
    #     raw_pred_seg = yolo_outputs[num_outputs-1]
    #     print('raw_true_seg, raw_pred_seg',raw_true_seg, raw_pred_seg)
    #     # seg_loss = K.binary_crossentropy(raw_true_seg, output=raw_pred_seg, from_logits=False) #requires sigmoid activation
    #     seg_loss = K.categorical_crossentropy(raw_true_seg, raw_pred_seg, from_logits=False) #requires softmax activation
    #     # loss += seg_loss
    #     seg_loss = K.sum(seg_loss) / mf
    #     loss += seg_loss * 2.0
    #     # loss += K.sum(seg_loss) / mf #mf seems to be the batch size.
    #     if print_loss:
    #         loss = tf.Print(loss, [loss, seg_loss], message="loss (seg): ")
    # elif model_name in ['tiny_yolo_infusion_hydra']:
    #     #calc seg loss
    #     raw_true_seg_1 = y_true[num_outputs-2] #seg is always the last output.
    #     raw_pred_seg_1 = yolo_outputs[num_outputs-2]
    #     print('raw_true_seg_1, raw_pred_seg_1',raw_true_seg_1, raw_pred_seg_1)
    #     raw_true_seg_2 = y_true[num_outputs-1] #seg is always the last output.
    #     raw_pred_seg_2 = yolo_outputs[num_outputs-1]
    #     print('raw_true_seg_2, raw_pred_seg_2',raw_true_seg_2, raw_pred_seg_2)
    #     seg_loss_1 = K.categorical_crossentropy(raw_true_seg_1, raw_pred_seg_1, from_logits=False) #requires softmax activation
    #     seg_loss_2 = K.categorical_crossentropy(raw_true_seg_2, raw_pred_seg_2, from_logits=False) #requires softmax activation
    #     # loss += seg_loss
    #     seg_loss_1 = K.sum(seg_loss_1) / mf
    #     loss += seg_loss_1
    #
    #     seg_loss_2 = K.sum(seg_loss_2) / mf
    #     loss += seg_loss_2
    #     # loss += K.sum(seg_loss) / mf #mf seems to be the batch size.
    #     if print_loss:
    #         loss = tf.Print(loss, [loss, seg_loss_1, seg_loss_2], message="loss (seg): ")
    #new style: keep commented.

    return loss
