import sys
sys.path.insert(0, 'yolov5')
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from nms import nms, scale_coords_numpy
from torch.backends import cudnn
from pathlib import Path
import numpy as np
import argparse
import os
import platform
import shutil
import time
import cv2
import torch


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def bbox_left_top_right_bottom(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_right = max([xyxy[0].item(), xyxy[2].item()])
    bbox_bottom = max([xyxy[1].item(), xyxy[3].item()])

    return bbox_left, bbox_top, bbox_right, bbox_bottom


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, cls_ids=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{},{:d}'.format(cls_ids[i], id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.img_size, opt.evaluate

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights,
                     repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')

    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(
        model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None

    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    counting_folder = './counting'
    if not os.path.exists(counting_folder):
        os.makedirs(counting_folder)

    counting_path = os.path.join(counting_folder, txt_file_name + '.txt')

    counting_dict = {
        1: [], 2: [], 3: [], 4: []
    }
    count_accumulate = [0, 0, 0, 0]

    with open(txt_path, 'w') as f:
        with open(counting_path, 'w') as f_c:

            for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
                flag = True
                if frame_idx % 3 != 0:
                    flag = False                    

                if flag:                    
                    main_fps = vid_cap.get(cv2.CAP_PROP_FPS)

                    if frame_idx % 180 == 0:
                        for i in range(4):
                            count_accumulate[i] += len(set(counting_dict[i+1]))
                        counting_dict = {
                            1: [], 2: [], 3: [], 4: []
                        }

                    _, img_h, img_w = img.shape
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=opt.augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(
                        pred, opt.conf_thres, opt.iou_thres,
                        classes=opt.classes, agnostic=opt.agnostic_nms)

                    if isinstance(pred, list):
                        continue

                    pred = pred.detach().cpu().numpy()

                    pred[:, [0, 2]] /= img_w
                    pred[:, [1, 3]] /= img_h

                    bboxes, scores, labels = nms([pred[:, :4]], [pred[:, 4]], [
                        pred[:, 5]], iou_thr=opt.iou_thres)
                    bboxes[:, [0, 2]] *= img_w
                    bboxes[:, [1, 3]] *= img_h

                    pred = np.zeros((len(bboxes), 6))
                    pred[:, :4] = bboxes
                    pred[:, 4] = scores
                    pred[:, 5] = labels

                    t2 = time_synchronized()
                    # Process detections
                    for i, det in enumerate([pred]):  # detections per image

                        p, s, im0 = path, '', im0s

                        s += '%gx%g ' % img.shape[2:]  # print string
                        save_path = str(Path(out) / Path(p).name)

                        if det is not None and len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords_numpy(
                                img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in np.unique(det[:, -1]):
                                n = np.sum(det[:, -1] == c)  # detections per class
                                # add to string
                                s += '%g %ss, ' % (n, names[int(c)])

                            bbox_xywh, confs, bbox_ltrb, cls_list = [], [], [], []

                            # Adapt detections to deep sort input format
                            for *xyxy, conf, cls in det:

                                x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                                obj = [x_c, y_c, bbox_w, bbox_h]
                                bbox_xywh.append(obj)
                                confs.append([conf.item()])

                                bbox_ltrb.append(
                                    list(map(int, bbox_left_top_right_bottom(*xyxy))))
                                cls_list.append(int(cls.item() + 1))

                            xywhs = torch.Tensor(bbox_xywh)
                            confss = torch.Tensor(confs)

                            # Pass detections to deepsort
                            outputs = deepsort.update(xywhs, confss, im0)

                            update_label_list = []
                            have_detect_list = []
                            # Write MOT compliant results to file
                            if save_txt and len(outputs) != 0:
                                for j, output in enumerate(outputs):
                                    bbox_left, bbox_top, bbox_right, bbox_bottom, identity = output
                                    label = -1

                                    for idx, det_item in enumerate(bbox_ltrb):
                                        det_l, det_t, det_r, det_b = det_item
                                        if abs(det_l-bbox_left) < 20 and \
                                                abs(det_t-bbox_top) < 20 and \
                                                abs(det_r - bbox_right) < 20 and \
                                                abs(det_b - bbox_bottom) < 20:

                                            label = cls_list[idx]
                                            have_detect_list.append(output)
                                            update_label_list.append(label)
                                            counting_dict[label].append(identity)
                                            break

                                    if label == -1:
                                        continue

                                    f.write(('%g ' * 7 + '\n') % (frame_idx + 1, label, identity,
                                                                bbox_left, bbox_top, bbox_right, bbox_bottom))  # label format

                            if len(have_detect_list) > 0:
                                draw_boxes(im0,
                                        np.array(have_detect_list)[:, :4],
                                        np.array(have_detect_list)[:, -1],
                                        update_label_list)

                        else:
                            deepsort.increment_ages()

                        print('%sDone. (%.3fs)' % (s, t2 - t1))

                        # Save results (image with detections)

                    if save_vid:
                        if dataset.mode == 'images':
                            cv2.imwrite(save_path, im0)
                        else:
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer

                                fps = vid_cap.get(cv2.CAP_PROP_FPS)//3
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                vid_writer = cv2.VideoWriter(
                                    save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                            vid_writer.write(im0)

                if (frame_idx+1) % int(main_fps * 60) == 0:
                    # count per minute

                    f_c.write(' '.join(list(map(str, count_accumulate))) + '\n')
                    count_accumulate = [0, 0, 0, 0]

            if sum(count_accumulate) > 0:
                f_c.write(' '.join(list(map(str, count_accumulate))) + '\n')

        f_c.close()
    f.close()

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str,
                        default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--evaluate', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
