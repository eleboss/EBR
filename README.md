# Neuromorphic Synergy for Video Binarization (Academic Use Only)
## [Paper](https://google.com) | [Youtube](https://youtu.be/miR-nFxa35g)
## This work is currently under reviewing, we will release the code once ready.

Binary images are widely used in computer vision and robotics applications due to their ability to provide essential bimodal information for efficient downstream processing. However, motion blur caused by relative camera and target motion in dynamic scenarios can significantly degrade the performance of image binarization. Recent advances in bio-inspired vision sensors called event cameras offer promising capabilities for alleviating motion blur in binary images. However, previous event-based image reconstruction methods are not directly applicable to motion deblurring for binary images, and are computationally intensive, limiting their real-time applicability. In this paper, we propose a novel event-based binary reconstruction (EBR) framework that integrates events and images to directly produce sharp binary images. We demonstrate that motion deblurring for binary images can be decomposed into two subproblems that can be efficiently solved in linear time, without the need for solving the intricate problem of intensity reconstruction. Our proposed method leverages a novel representation that efficiently fuses events and images for unsupervised and unparameterized motion-invariant recovery of the sharp binary image. We further propose an efficient method for producing high-rate sharp binary videos under various types of complex motion, allowing for asynchronous video generation. Extensive evaluations on various datasets demonstrate that our proposed method outperforms the current state-of-the-art methods.

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

