# Compressed-Sensing-Image-Recovery
This is a project for my machine learning course (ECE 580), where we used machine learning techniques to recover corrupted images. For detailed information about the project, see the pdf included in the respository: `Compressed-Sensing-Image-Recovery.pdf`

## Problem and Modeling Approach
Given a corrupted image, we would like to recover the original image. 

To do this, we compute the discrete cosine transform (DCT) on blocks of pixels within the image. This provides us the DCT coefficients for the different basis functions. To find the original pixel values, we are now left with an underdetermined linear system. This means there are infinite solutions so we must introduce additional constraints. We introduced the constraint of sparsity, since generally the DCT coefficients tend to be sparse in local areas within an image.

This modeling approach was evaluated with cross validation and with a variety of different parameter settings and post-processing filters. The results are shown and discussed the in the attached PDF.

## Repo Structure
- `regression_compressive_sensing.py`: Contains code to perform image recovery experiments.
- `data/`: contains bitmap image files for experimentation
- `Matlab/`: contains original matlab files for reading and showing images
- `Compressed-Sensing-Image-Recovery.pdf`: explains project motivation, mathematical formulation, results, and discussion of results
