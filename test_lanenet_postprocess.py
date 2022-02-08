#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import os.path as ops
import argparse
import time
import math
import sys

#sys.path = ['', '/root/root_dit_atlas/home/cjcramer/lane_lines/LaneNet', '/root/anaconda3/lib/python36.zip', '/root/anaconda3/lib/python3.6', '/root/anaconda3/lib/python3.6/lib-dynload', '/root/anaconda3/lib/python3.6/site-packages']
import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2

import lanenet_merge_model
import lanenet_cluster
import lanenet_postprocess
import global_config

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]

DO_POST_PROCESS = True

# Need more points, because poly is overfitting
ORDER = 3
Y_POLY_POINT_STEP = 1
EXTEND_LANE_DOWN_BY = 30

lane_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 255], [255, 255, 255]]

#lane_colors=[255,255,255]
# HSV ranges (THESE ARE HSV VALUES)
lower_blu = np.array([118, 240, 10])
upper_blu = np.array([122, 255, 255])

lower_red = np.array([0, 240, 10])
upper_red = np.array([6, 255, 255])

lower_grn = np.array([54, 240, 10])
upper_grn = np.array([66, 255, 255])

lower_cya = np.array([84, 240, 10])
upper_cya = np.array([96, 255, 132])


# Function of lane width varied over height
# May need to fidget with this if the resolution differernt than CULane
def lane_width_px(y):
    if y <= 260:
        return 1
    return (int) (y * 0.1 - 25)


# @each row in @points should be of form (x,y)
# @lane_number is the number to be placed in the grayscale image, which represents the class of the lane line
# @points should be going from the top of the image downwards, the drawing will stop when the lane line goes off screen
def draw_lane(img, points, lane_number):
    # Draw a line in for each point
    for i in range(points.shape[0]):
        point = points[i, :]
        lane_width = lane_width_px(point[1])
        if point[1] < 0 or point[1] > img.shape[0] or point[0] < lane_width // 2 or point[0] > img.shape[1] + lane_width // 2:
            return
        cv2.line(img, (point[0] - lane_width // 2, point[1]), (point[0] + lane_width // 2, point[1]), lane_number, 1) # Thickness = 1


# @point should be (x,y) form of point, top left corner is origin point
# @ width, height should be the height and width of the frame
def angle_from_top_center(point, width, height):
    dx = (width / 2) - point[0]
    dy = point[1] - 0.25 * height   # Take center point y dimension to be closer to vanishing point
    return math.atan2(dy, dx)


def post_process_output(mask):
    # Threshold different color lines out
    mask_hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)    # Convert to HSV
    # Get each lane
    red_res = cv2.inRange(mask_hsv, lower_red, upper_red)
    blu_res = cv2.inRange(mask_hsv, lower_blu, upper_blu)
    grn_res = cv2.inRange(mask_hsv, lower_grn, upper_grn)
    cya_res = cv2.inRange(mask_hsv, lower_cya, upper_cya)

    lanes = [{"color": "red", "lane": red_res}, {"color": "blue", "lane": blu_res},
             {"color": "green", "lane": grn_res}, {"color": "cyan", "lane": cya_res}]

    # Get centers first
    for i in range(len(lanes)):
        lane = lanes[i]["lane"]
        M = cv2.moments(lane)
        # calculate x,y coordinate of center
        if M["m00"] == 0:
            lanes[i]["center"] = None
            lanes[i]["DNE"] = True
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #cv2.circle(lane, (cX, cY), 40, [255], -1)
            lanes[i]["center"] = (cX, cY)

    # Get angles
    for i in range(len(lanes)):
        if "DNE" not in lanes[i].keys():
            lanes[i]["angle"] = angle_from_top_center(lanes[i]["center"], mask.shape[1], mask.shape[0])
        else:
            lanes[i]["angle"] = 0    # Assume left most lane

    # Sort the lanes according to the angle between the centers of each lane and the top center
    def sort_by_angle(d):
        return d["angle"]
    lanes = sorted(lanes, key = sort_by_angle)

    post_processed = np.zeros((lanes[0]["lane"].shape[0], lanes[0]["lane"].shape[1], 3), np.uint8)
    # Calculate 6th order polynomial for each
    for i in range(len(lanes)):
        if "DNE" in lanes[i].keys():
            continue
        points = np.argwhere(lanes[i]["lane"] != 0)
        # Polynomial regression, Evaluate polynomial, Draw polynomial
        if points.shape[0] > 0:
            coeefs = np.polyfit(points[:,0], points[:,1], ORDER)  # As a function of Y
            y_vals = list(range(min(points[:,0]), max(points[:,0]) + EXTEND_LANE_DOWN_BY, Y_POLY_POINT_STEP))
            x_vals = [np.polyval(coeefs, y) for y in y_vals]
            poly_points = np.int32( np.column_stack((x_vals, y_vals)) )
            # Draw the polynomial
            draw_lane(post_processed, poly_points, lane_colors[i])

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)   # So we can see it properly when plotting
    cv2.circle(mask, (mask.shape[1] // 2, int(mask.shape[0] * 0.25)), 20, [255], -1)

    return post_processed


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet_batch(image_dir, weights_path, batch_size, use_gpu, save_dir=None):
    """

    :param image_dir:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param save_dir:
    :return:
    """

    tf.reset_default_graph()    # So we can run this multiple times

    print("save_dir in function: ", save_dir)

    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    log.info('Start getting the path to the image file....')
    image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)

    print("NUMBER OF IMAGES IN IMAGE_PATH: ", len(image_path_list))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        print("SUCCESSFULLY RESTORED FROM SESSION")

        epoch_nums = int(math.ceil(len(image_path_list) / batch_size))

        print("EPOCH_NUMS = ", epoch_nums)

        for epoch in range(epoch_nums):
            log.info('[Epoch:{:d}] Start image reading and preprocessing...'.format(epoch))
            t_start = time.time()
            image_path_epoch = image_path_list[epoch * batch_size:(epoch + 1) * batch_size]
            image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
            image_vis_list = image_list_epoch
            image_list_epoch = [cv2.resize(tmp, (512, 256), interpolation=cv2.INTER_LINEAR)
                                for tmp in image_list_epoch]
            image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] Preprocessing {:d} images, total time: {:.5f}s, average time per image: {:.5f}'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            t_start = time.time()
            binary_seg_images, instance_seg_images = sess.run(
                [binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: image_list_epoch})
            t_cost = time.time() - t_start
            log.info('[Epoch:{:d}] Predicting lane lines for {:d} images, total time: {:.5f}s, average time per image: {:.5f}s'.format(
                epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))


            cluster_time = []
            for index, binary_seg_image in enumerate(binary_seg_images):
                t_start = time.time()
                binary_seg_image = postprocessor.postprocess(binary_seg_image)
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                   instance_seg_ret=instance_seg_images[index])
                cluster_time.append(time.time() - t_start)
                mask_image = cv2.resize(mask_image, (image_vis_list[index].shape[1],
                                                     image_vis_list[index].shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

                if save_dir is None:
                    plt.ion()
                    plt.figure('mask_image')
                    plt.imshow(mask_image[:, :, (2, 1, 0)])
                    plt.figure('src_image')
                    plt.imshow(image_vis_list[index][:, :, (2, 1, 0)])
                    plt.pause(3.0)
                    plt.show()
                    plt.ioff()

                if save_dir is not None:
                    # Mask is the predicted lane line pixels with colors on a black background
                    if DO_POST_PROCESS == True:
                        mask_image = post_process_output(mask_image)
                    # After this line it will layer the lines on the actual image
                    # Comment it out to save just the predictions
                    mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)

                    image_name = ops.split(image_path_epoch[index])[1]
                    image_save_path = ops.join(save_dir, image_name).replace("\\","/")
                    cv2.imshow(image_save_path, mask_image)
                    print("SAVED MASK IMAGE TO ", image_save_path)

            log.info('[Epoch:{:d}]Perform lane line clustering of {:d} images, total time: {:.5f}s, average time per image: {:.5f}'.format(
                epoch, len(image_path_epoch), np.sum(cluster_time), np.mean(cluster_time)))

    sess.close()

    return


if __name__ == '__main__':
    # init args

    test_lanenet_batch(image_dir="test_set/",
                       weights_path="model/culane_lanenet_2/culane_lanenet_2_vgg_2022-01-04-21-25-04.ckpt-9250",
                       save_dir="C:/Users/talka/OneDrive/Dokumenty/Studia/2stopien/WK/Projekt/output_frames2/",
                       use_gpu=1,
                       batch_size=1)
    print("All done)")