# Required enviroment
    Tensorflow 1.13.1
    Or
    Pytorch


# A-computation-efficient-CNN-system-for-high-quality-brain-tumor-segmentation

[1] Yanming Sun and Chunyan Wang, A computation-efficient CNN system for high-quality brain tumor segmentation, Biomedical Signal Processing and Control, vol. 74, 2022, 103475.


Abstract

In this paper, a reliable computation-efficient system of Convolutional Neural Network (CNN) is proposed for brain tumor segmentation. It consists of a segmentation-CNN, a pre-CNN block for data reduction and a refinement block. The unique CNN is custom-designed, following the proposed paradigm of ASCNN (Application Specific CNN), to perform mono-modality and cross-modality feature extractions, tumor localization and pixel classification. It features modality-wise normalization to improve the input data quality, depthwise convolution, combined with instance normalization, for the mono-modality feature extraction, bilinear upsampling for dimension expansion without introducing randomness, and weighted data addition for signal modulation. The proposed activation function Full-ReLU helps to halve the number of kernels in convolution layers of high-pass filtering without degrading processing quality. In this specific design context, the CNN is structured to have 7 convolution layers, requiring only 108 kernels and 20,308 trainable parameters in total. The number of kernels in each layer is made just-sufficient for its task, instead of exponentially growing over the layers, with a view to a higher information density in data channels and lower randomness in network training. Extensive experiments with BRATS2018 dataset have been conducted to confirm the high-level processing quality and reproducibility of the system. The mean-dice-scores for enhancing-tumor, whole-tumor and tumor-core are 77.2%, 89.2% and 76.3%, respectively. Testing each patient case requires only 29.07G Flops, a tiny fraction of what found in literature. The simple structure and reliable high processing quality of the proposed system will facilitate its implementation and medical applications.
