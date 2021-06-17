import logging
from pathlib import Path

import numpy as np
import torch
import cv2 as cv2
import matplotlib.pyplot as plt

from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model

logger = logging.getLogger()

def createVideo(pred_sum, video_path, output_name):
  #print(pred_sum.size)
  #original_frame = np.arange(0, pred_sum.size)
  cap = cv2.VideoCapture(video_path)
  frame_height = int(cap.get(4))
  frame_width = int(cap.get(3))
  out = cv2.VideoWriter(output_name + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
  index = -1
  segmentIndex = 0
  print(pred_sum.size)

  while (cap.isOpened()):
    ret, frame = cap.read()
    
    if (index >= pred_sum.size):
      break
    
    if (ret and pred_sum[index]):
      index += 1
      out.write(frame)
    elif (ret):
      index +=1
      continue
  
  length = int(out.get(cv2.CAP_PROP_FRAME_COUNT))
  print( length )

  cap.release()
  out.release()

def evaluate(model, val_loader, nms_thresh, device):
    model.eval()
    stats = data_helper.AverageMeter('fscore', 'diversity')

    with torch.no_grad():
        for test_key, seq, _, cps, n_frames, nfps, picks, user_summary in val_loader:
            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

            pred_cls, pred_bboxes = model.predict(seq_torch)

            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)
            pred_summ = vsumm_helper.bbox2summary(
                seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)
            print(pred_summ)
    return pred_summ


def main():
    args = init_helper.get_arguments()

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)

    logger.info(vars(args))
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        stats = data_helper.AverageMeter('fscore', 'diversity')

        for split_idx, split in enumerate(splits):
            split_path_test = 'summe.yml'
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path_test, split_idx)
            state_dict = torch.load(str(ckpt_path),
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

            val_set = data_helper.VideoDataset(split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)

            pred_sum = evaluate(model, val_loader, args.nms_thresh, args.device)
            
    createVideo(pred_sum, args.video_path, args.output_name)
    bar_img = np.zeros((120, len(pred_sum)), np.float32)
    for i in range(0, len(pred_sum)):
      #print(pred_sum[i])
      if (pred_sum[i] == True):
        cv2.line(bar_img, (i, 0), (i, 120), (100, 0, 0), 1)
      else:
        cv2.line(bar_img, (i, 0), (i, 120), (255, 100, 100), 1)
    #print(bar_img)
    #plt.imshow(bar_img)
    barImgBorder= cv2.copyMakeBorder(bar_img,2,2,2,2,cv2.BORDER_CONSTANT,value=[0,0,0])

    cv2.imwrite('bar_img.png', barImgBorder)

if __name__ == '__main__':
    main()

