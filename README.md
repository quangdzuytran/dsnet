# DSNet: A Flexible Detect-to-Summarize Network for Video Summarization
## Project Team:
- 18120022 - Trần Quang Duy
- 18120512 - Lê Đặng Thiên Phúc

## How to run
If you are using conda, the environment can be installed using the [dsnet_env.yml](https://github.com/quangdzuytran/dsnet/blob/c99ac0be913602d486ddbb549078876b4a2ad9cd/dsnet_env.yml) file.\
To train and evaluate models, follows the guide from [https://github.com/li-plus/DSNet](https://github.com/li-plus/DSNet). (use **--base-model transformer** to use the transformer encoder as base model)\
Google Colab notebook used to train models are available at [https://colab.research.google.com/drive/1wPRUsrLR2C3pfIyW2GyUARZIBPSJJlwO?usp=sharing](https://colab.research.google.com/drive/1wPRUsrLR2C3pfIyW2GyUARZIBPSJJlwO?usp=sharing).\
Trained models are available at [https://drive.google.com/drive/folders/1G4M7HWXvP-1F9O2f2TdEcliJCjB7iHT5?usp=sharing](https://drive.google.com/drive/folders/1G4M7HWXvP-1F9O2f2TdEcliJCjB7iHT5?usp=sharing).

## Acknowledgments
This code is edited from [DSNet](https://github.com/li-plus/DSNet).\
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
