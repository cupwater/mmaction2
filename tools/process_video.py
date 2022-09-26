'''
Author: Peng Bo
Date: 2022-09-26 17:14:57
LastEditTime: 2022-09-26 21:18:50
Description: 

'''

import cv2
import os
import argparse
import pdb


def process_video(video_path, output, sample_step=1, sample_duration=3, tgt_size=(320, 240)):
    cap = cv2.VideoCapture(video_path)

    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_counter / fps

    processed_imgs_list = []
    while True:
        _, ori_image = cap.read()
        if ori_image is None:
            break
        h, w, _ = ori_image.shape
        if 1.0*h/w >= 1.0*tgt_size[1]/tgt_size[0]:
            new_w = tgt_size[0]
            new_h = int(h*tgt_size[0]/w)
            ori_image = cv2.resize(ori_image, (new_w, new_h))
            new_image = ori_image[int(
                new_h/2-tgt_size[1]/2):int(new_h/2+tgt_size[1]/2), :, :]
        else:

            new_w = int(w*tgt_size[1]/h)
            new_h = tgt_size[1]
            ori_image = cv2.resize(ori_image, (new_w, new_h))
            new_image = ori_image[:, int(
                new_w/2-tgt_size[0]/2):int(new_w/2+tgt_size[0]/2), :]
        new_image = cv2.resize(ori_image, tgt_size)
        processed_imgs_list.append(new_image)
    cap.release()

    # generate a sample every 'sample_step' second
    for idx in range(0, int(duration-1), sample_step):
        start_idx = int(idx*fps)
        end_idx = int(start_idx + sample_duration*fps)
        end_idx = end_idx if end_idx < frame_counter else frame_counter
        folder_path, file_name = os.path.dirname(
            video_path), os.path.basename(video_path)
        write_name = os.path.splitext(file_name)[0] + '_' + str(idx) + '.mp4'
        out_path = os.path.join(output, folder_path, write_name)
        print(out_path)
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        videoWriter = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, tgt_size)
        for frame in processed_imgs_list[start_idx:end_idx]:
            videoWriter.write(frame)
        videoWriter.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='split a video into several snippets')
    parser.add_argument('--input', type=str,
                        help='The source root of input videos')
    parser.add_argument('--output', type=str,
                        help='The target root of output videos')
    parser.add_argument('-s', '--sample-step', type=int,
                        default=1, help='The step of sampler')
    parser.add_argument('-d', '--sample-duration', type=int,
                        default=3, help='The duration of a snippet')
    args = parser.parse_args()

    for parent_root, dirs, flist in os.walk(args.input):
        for f in flist:
            if 'mp4' in f:
                print(os.path.join(parent_root, f))
                process_video(os.path.join(parent_root, f), 
                        args.output,
                        sample_step=args.sample_step, 
                        sample_duration=args.sample_duration)
