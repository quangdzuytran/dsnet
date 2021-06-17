# DSNet: A Flexible Detect-to-Summarize Network for Video Summarization
## Project Team
- 18120022 - Trần Quang Duy
- 18120512 - Lê Đặng Thiên Phúc

## Getting started
If you are using conda, the environment can be installed using the [dsnet_env.yml](https://github.com/quangdzuytran/dsnet/blob/c99ac0be913602d486ddbb549078876b4a2ad9cd/dsnet_env.yml) file.\
To train and evaluate models, follows the guide from [https://github.com/li-plus/DSNet](https://github.com/li-plus/DSNet). (use **--base-model transformer** to use the transformer encoder as base model)\
Google Colab notebook used to train models are available at [https://colab.research.google.com/drive/1wPRUsrLR2C3pfIyW2GyUARZIBPSJJlwO?usp=sharing](https://colab.research.google.com/drive/1wPRUsrLR2C3pfIyW2GyUARZIBPSJJlwO?usp=sharing).\
Trained models are available at [https://drive.google.com/drive/folders/1G4M7HWXvP-1F9O2f2TdEcliJCjB7iHT5?usp=sharing](https://drive.google.com/drive/folders/1G4M7HWXvP-1F9O2f2TdEcliJCjB7iHT5?usp=sharing).


# How to create summarized video for Youtube dataset

First you have to have the original video of the dataset, most of the used datasets are preprocessed under the file .h5, so check out this link of Vsumm dataset (the original of OVP and Youtube dataset): https://sites.google.com/site/vsummsite/download

To run the export module,  you could use the ovp dataset instead of youtube dataset, NumOfVideo is for the index of the video, you want to test, the result of the file is the original filename append with ".export_demo":
'''
!python export_demo_shots.py --dataset '../datasets/eccv16_dataset_youtube_google_pool5.h5' --video-number video_#NumOfVideo
'''

Then create the yaml file for the above eccv16_dataset_youtube_google_pool5.h5.export_demo
'''
!python make_split.py --dataset ../datasets/eccv16_dataset_youtube_google_pool5.h5.export_demo --save-path ../splits/testYT.yml --num-splits 1 --train-ratio 0
'''

Run these two lines to create video for the video with the anchor-based method, the result video will be in the file type of '.avi'
'''
!python export_video.py anchor-based --model-dir ../models/ab_basic/ --splits ../splits/testYT.yml --video-path video_#NumOfVideo.avi --output-name output
'''

Or you could export video that used the anchor-free method:
'''
!python export_video.py anchor-free --model-dir ../models/af_basic/ --splits ../splits/testYT.yml --video-path video_#NumOfVideo.avi --output-name output --nms-thresh 0.4
'''

## Acknowledgments
This code is edited from [DSNet](https://github.com/li-plus/DSNet).
We gratefully thank the below open-source repo, which greatly boost our research.
+ Thank [KTS](https://github.com/pathak22/videoseg/tree/master/lib/kts) for the effective shot generation algorithm.
+ Thank [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) for the pre-processed public datasets.
+ Thank [VASNet](https://github.com/ok1zjf/VASNet) for the training and evaluation pipeline.

## Citation
```
@article{zhu2020dsnet,
  title={DSNet: A Flexible Detect-to-Summarize Network for Video Summarization},
  author={Zhu, Wencheng and Lu, Jiwen and Li, Jiahao and Zhou, Jie},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={948--962},
  year={2020}
}
```
