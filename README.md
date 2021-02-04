# Go-CaRD - A Generic, Optical Car Part Recognition and Detection System

Go-CaRD is an automatic recognition and detection system of automotive parts e.g. to study interactions between human and the vehicle.
![interaction](https://github.com/lstappen/GoCaRD/blob/main/img/interaction.gif)
![in-out](https://github.com/lstappen/GoCaRD/blob/main/img/in-out.gif)
![cockpit](https://github.com/lstappen/GoCaRD/blob/main/img/cockpit.gif)
![distance-to-close](https://github.com/lstappen/GoCaRD/blob/main/img/distance-to-close.gif)
Some examples on random videos. Note: Modell is not trained on consecutive frames (e.g. videos). 
<!---[GoCarD Example](https://github.com/lstappen/GoCarD/blob/main/img/example_MuSe_CaR.png?raw=true))-->

## Results
### Recognition 

Results of **CLOSE-CaR** using the FULL head on the devel(opment) and test set reported in F1. 
Best model trained on [devel/test]:
- inside and outside classes: RESNET50 90.23 / 93.76
- outside classes: DENSENET201 97.13 / 93.08 
- inside classes: MOBILENETV2 87.00 / 93.60

### Detection 

Results using a dataset for training (T1) with optional data injections in % of the second training set (T2); considering three levels of IoU fit for reporting mAP in  % on the test set.

| T1  | Inj.% | T2   | dev/test | > 0.2 | > 0.4 | > 0.5 |
|-----|----------|------|----------|-------|-------|-------|
|<td colspan=7>Darknet-backbone
| mix | --       | --   | mix      | **58.20** | 56.66 | 54.60 |
| mix | --       | --   | part     | 17.39 | 15.89 | 14.46 |
| mix | 100    | part | part     | 41.07 | 38.60 | 35.56 |
|<td colspan=7>TinyDarknet-backbone   |
| mix | --       | --   | mix      | 40.89 | 26.43 | 24.41 |
| mix | 100    | part | part     | 28.24 | 14.44 | 12.51 |
|<td colspan=7>SqueezeNet-backbone   |
| mix | --       | --   | mix      | 46.03 | 44.14 | 42.29 |
| mix | 100    | part | part     | 22.99 | 21.00 | 19.07 |

A detailed version in the paper.

## Pretrained models ([download here](https://zenodo.org/record/4453520)) 

The purpose of these models are to support research in the field of automatic recognition and detection of automotive parts in a natural context. It provides predictions for 29 interior and exterior vehicle regions during human-vehicle interaction. It also enables benchmarking and cross-corpus transfer learning, as demonstrated in GoCarD (A Generic, Optical Car Part Recognition and Detection).


## MuSe-CaR-Part dataset ([download here](https://zenodo.org/record/4450468)) 

This dataset is a subset of 74 videos from the multimodal in-the-wild dataset MuSe-CAR. It contains 1 124 video frames showing human-vehicle interactions across all MuSe topics and 6 146 labels (bounding boxes). The pre-defined training, development and test partitions are also provided. 

The purpose of this dataset is to support research in the field of automatic recognition and detection of automotive parts in a natural context. It provides labels for 29 interior and exterior vehicle regions during human-vehicle interaction. It also enables benchmarking and cross-corpus transfer learning, as demonstrated in GoCarD (A Generic, Optical Car Part Recognition and Detection). The footage captures many "in-the-wild" characteristics, including a range of shot sizes, camera motion, moving objects, a wide variety of backgrounds and different interactions. 

## GoCaRD evaluation and feature extraction code 
The code in this repo is only exemplary and shows a) how images could be predicted from MuSe-CaR-Part using the model weights b) transform the prediction to a fixed sized feature set as used in [1], [2]. It is not intended to work out-of-the-box e.g. on videos. However, the weights can be used for such an application.

## Usage: For Research Purposes Only

Any data, models, and weights derived from data contained in **MuSe-CaR** may only be used for scientific, non-commercial applications. The Academic Parties have the right to use these components for Academic and Research Purposes and Academic Publication is permitted. Commercial applications include, but are not limited to: Proving the efficiency of commercial systems, Testing commercial systems, Using screenshots of subjects from the database in advertisements, Selling data from the database. Please download and fill out the **EULA - End User License Agreement** before requesting the data or models. We will review your application and get in touch as soon as possible. Thank you.


## References
**[1]** L Stappen , G Rizos, B Schuller. (2020). X-AWARE: ConteXt-AWARE human-environment attention fusion for driver gaze prediction in the wild. In Proceedings of the 22nd International Conference on Multimodal Interaction (ICMI). ACM.<br>
**[2]** L Stappen, A Baird, G Rizos, P Tzirakis, X Du, F Hafner, L Schumann, A Mallol-Ragolta, BW Schuller, I Lefter, E Cambria. (2020). MuSe 2020 challenge and workshop: Multimodal sentiment analysis, emotion-target engagement and trustworthiness detection in real-life media: Emotional car reviews in-the-wild. In Proceedings of the 1st International on Multimodal Sentiment Analysis in Real-life Media Challenge and Workshop 2020 (MuSe). ACM.
