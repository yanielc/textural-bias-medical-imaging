# Textural bias medical imaging

## MRI artifacts

During data acquistion and image reconstruction different types of artifacts can make their way into the final MR images. In this repository I have implemented transforms that model three such artifacts: Gibbs (or truncation) artifact, spikes (herringbone) artifact, and wraparound (alasing) artifact. Each transform is implement in the k-space version of the data.

More details on the physics of the artifacts can be found in the following publications:

[AAPM/RSNA physics tutorial for residents: fundamental physics of MR imaging.](https://pubmed.ncbi.nlm.nih.gov/16009826)

[Body MRI artifacts in clinical practice: A physicist's and radiologist's perspective.](https://doi.org/10.1002/jmri.24288)

[An Image-based Approach to Understanding the Physics of MR Artifacts.](https://pubs.rsna.org/doi/full/10.1148/rg.313105115)

### Examples and visualizations
For an overview of the three transforms and visualizations of their applications on the data refer to the Jupyter notebook: __artifacts_transforms_visualizations.ipynb__.

### MONAI contributions

I have contributed the Gibbs filter to the codebase of [MONAI](https://monai.io/) as various transfroms:

1. Related to Gibbs artifacts: 
     * [GibbsNoise](https://docs.monai.io/en/latest/transforms.html?highlight=GibbsNoise#gibbsnoise): applies artifact directly on the image. 
     * [RandGibbsNoise](https://docs.monai.io/en/latest/transforms.html?highlight=RandGibbsNoise#randgibbsnoise): to apply randomly on images with uniform sampling of filter's intensity.
     * [GibbsNoised](https://docs.monai.io/en/latest/transforms.html?highlight=GibbsNoised#gibbsnoised): to apply on group data; dictionary-style version of ``GibbsNoise``.
     * [RandGibbsNoised](https://docs.monai.io/en/latest/transforms.html?highlight=RandGibbsNoised#monai.transforms.RandGibbsNoised): dictionary-style of ``RandGibbsNoise``.

2. Related to Spikes (herringbone) artifacts:
     * [KSpaceSpikeNoise](https://docs.monai.io/en/latest/transforms.html#kspacespikenoise): applies spike artifacts directly on the image.
     * [RandKSpaceSpikeNoise](https://docs.monai.io/en/latest/transforms.html#kspacespikenoise): applies the artifacts randomly on images while sampling in the intensity of the artifacts.
     * [KSpaceSpikeNoised](https://docs.monai.io/en/latest/transforms.html#kspacespikenoised): allows to apply transform both on the image and/or label; dictionary-style version of ``KSpaceSpikeNoise``.
     * [RandKSpaceSpikeNoised](https://docs.monai.io/en/latest/transforms.html#randkspacespikenoised): dictionary-version of ``RandKSpaceSpikeNoise``.
 


## Working with textural filters and DCNNs.
