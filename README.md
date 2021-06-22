# Textural bias medical imaging

## MRI artifacts

During data acquistion and image reconstruction different types of artifacts can make their way into the final MR images. In this repository I have implemented transforms that deal with three such artifacts: Gibbs (or truncation) artifact, spikes (herringbone) artifact, and wraparound (alasing) artifact. Each transform is implement in the k-space version of the data.

More details on the physics of the artifacts can be found in the following publications:

[AAPM/RSNA physics tutorial for residents: fundamental physics of MR imaging.](https://pubmed.ncbi.nlm.nih.gov/16009826)

[Body MRI artifacts in clinical practice: A physicist's and radiologist's perspective.](https://doi.org/10.1002/jmri.24288)

[An Image-based Approach to Understanding the Physics of MR Artifacts.](https://pubs.rsna.org/doi/full/10.1148/rg.313105115)

For an overview of the three transforms and visualizations of their applications on the data refer to the Jupyter notebook:


Working with textural filters and DCNNs.
