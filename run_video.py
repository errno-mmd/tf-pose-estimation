import argparse
import logging
import time

import cv2
import numpy as np
import os

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', action='store_true',
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--no_bg', action='store_true', help='show skeleton only.')
    parser.add_argument('--output_json', type=str, default='/tmp/', help='writing output json dir')
    parser.add_argument('--no_display', action='store_true', help='disable showing image')
    parser.add_argument('--resize_out_ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed')
    parser.add_argument('--number_people_max', type=int, default=1, help='maximum number of people')
    parser.add_argument('--frame_first', type=int, default=0, help='maximum number of people')
    parser.add_argument('--write_video', type=str, default=None, help='output video file')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.write_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.write_video, fourcc, 30.0, (width, height))
    
    frame = 0
    while cap.isOpened():
        ret_val, image = cap.read()
        if not ret_val:
            break
        if frame < args.frame_first:
            frame += 1
            continue
        
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        del humans[args.number_people_max:]
        if args.no_bg:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False, frame=frame, output_json_dir=args.output_json)
        frame += 1
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if not args.no_display:
            cv2.imshow('tf-pose-estimation result', image)
        if args.write_video is not None:
            out.write(image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    if args.write_video is not None:
        out.release()

    cv2.destroyAllWindows()
logger.debug('finished+')
