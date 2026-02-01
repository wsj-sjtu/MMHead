# MMHead: Towards Fine-grained Multi-modal 3D Facial Animation (ACM MM 2024)
## [Paper](https://arxiv.org/pdf/2410.07757) | [Project Page](https://wsj-sjtu.github.io/MMHead/) | [Video](https://www.youtube.com/watch?v=nnggJZhiEW4) | [Dataset](https://huggingface.co/datasets/Human-X/MMHead)

<img src="assets/teaser.png" /> 

## TODO
- [x] Release the codes for calculating the metrics.
- [x] Release the MMHead dataset.

## MMHead Dataset
### Download
The dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/Human-X/MMHead).

Please note that by requesting the dataset, you confirm that you have read, understood, and agree to be bound by the terms of the agreement.


### Overview
- The MMHead dataset is a multi-modal 3D facial animation dataset with hierarchical text annotations: (1) abstract action descriptions, (2) abstract emotions descriptions, (3) fine-grained expressions descriptions, (4) fine-grained head pose descriptions, and (5) emotion scenarios. The 3D facial motion is represented by 56-dimensional [FLAME](https://flame.is.tue.mpg.de/) parameters (50 expression + 3 neck pose + 3 jaw pose).
- MMHead dataset contains a total of 35903 facial motions, which is divided into two subsets for two tasks: (1) MMHead Subset I (28466 facial motions) for text-induced talking head animation, and (2) MMHead Subset II (7937 facial motions) for text-to-3D facial motion generation.

### Data Structure
```
MMHead
├── talking_subset.json
├── t2m_subset.json
├── audio.zip
│   ├── CELEBVHQ_--uyzf7X_0c_0.wav
│   └── ...
├── facial_motion.zip
│   ├── CELEBVHQ_--uyzf7X_0c_0.pkl
│   └── ...
└── text_annotations
    ├── action
    │    ├── CELEBVHQ_--uyzf7X_0c_0.txt
    │    └── ...
    ├── emotion
    │    ├── CELEBVHQ_--uyzf7X_0c_0.txt
    │    └── ...
    ├── detail_expression
    │    ├── CELEBVHQ_--uyzf7X_0c_0.txt
    │    └── ...
    ├── detail_head_pose
    │    ├── CELEBVHQ_--uyzf7X_0c_0.txt
    │    └── ...
    └── emotion_scenario
         ├── CELEBVHQ_--uyzf7X_0c_0.txt
         └── ...


• talking_subset.json contains the data list of MMHead Subset I, alone with its training, validation, and testing set splits.
• t2m_subset.json contains the data list of MMHead Subset II, alone with its training, validation, and testing set splits.
```


## Evaluation metrics
First, download 


## Citation
If you use this dataset, please consider citing
```
@inproceedings{wu2024mmhead,
  title={MMHead: Towards Fine-grained Multi-modal 3D Facial Animation},
  author={Wu, Sijing and Li, Yunhao and Yan, Yichao and Duan, Huiyu and Liu, Ziwei and Zhai, Guangtao},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={7966--7975},
  year={2024}
}
```

## Acknowledgement
Our code is developed with reference to [TMR](https://github.com/Mathux/TMR) and [text-to-motion](https://github.com/EricGuo5513/text-to-motion). We thank all the authors for their great work and repos.

## Contact
- Sijing Wu [(wusijing@sjtu.edu.cn)](wusijing@sjtu.edu.cn)
