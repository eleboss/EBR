# Neuromorphic Synergy for Video Binarization (Academic Use Only)
## [Paper](https://google.com) | [Video: Youtube](https://youtu.be/fbdvowAHkn0) | [Video: Bilibili](https://www.bilibili.com/video/BV1ws4y1X7Sh/)
## This work is currently under reviewing, we will release the code once ready.

Binary images play a crucial role in computer vision and robotics for distilling essential information for efficient downstream processing of widely used bimodal objects, like visual markers. However, real-world applications often have complex motion, resulting in motion-degraded images that existing binarization techniques cannot process. Although recent advances in neuromorphic cameras offer promising capabilities for alleviating motion blur, no previous event-based attempts have been able to generate sharp binary images directly and efficiently. Most previous attempts focused on intensity reconstruction, lacking the essential information distillation process for binary images. In this paper, for the first time, it is demonstrated that simultaneous motion deblurring and video binarization can be achieved without the need to solve the intricate problem of intensity reconstruction, which is limited by the time-consuming event-wise double integral. Instead, the binarization is decomposed into two sub-problems that can be efficiently solved in linear time in event space and image space, respectively, enabling the high frame-rate reconstruction of binary video. Moreover, a novel method is designed that naturally fused events and images to provide the distillation process in binarization, enabling unsupervised and parameter-free identification of the threshold and significantly relaxing the need for accurate contrast estimation. The whole pipeline is asynchronous and can operate in real-time on CPU-only devices. Extensive evaluations on various datasets demonstrate the state-of-the-art performance of the methods.

![Demo](./figures/image.png)



## Download model and data
In our paper, we conduct experiments on three types of data:
- **HQF** contains synthetic blurry images and real-world events from [HQF](https://timostoff.github.io/20ecnn), where blurry images are generated using the same manner as GoPro.
- **Reblur** contains real-world blurry images and real-world events from [Reblur](https://github.com/AHupuJR/EFNet).
- **EBT** is our collected dataset, it contains simulated event sequence using ESIM and real-world test data [EBT](https://google.com).

## EBT- Event-based Bimodal Target dataset
Examples of the real sequences of EBT dataset:
![Example of real sequence of EBT dataset](./figures/example.png)

Examples of the synthetic sequences of EBT dataset:
![Example of real sequence of EBT dataset](./figures/sim_example.png)

