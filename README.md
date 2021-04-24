# IMPR

Projects related to Introduction to Image Processing course.

## Code_Sample_1 - Image Representations, Intensity Transformations, Quantization
The main purpose of this program is to get acquainted with NumPy and some image proccessing facillities.

### The program covers:
- Loading grayscale and RGB image representations.
- Displaying figures and images.
- Transforming RGB images back and forth from the YIQ color space.
- Performing intensity transformations with histogram equalization.
- Performing optimal quantizations.

### Related Files
- **sol1.py**: The program file.
- **requirements_1.txt**: The requirements file for the program. 

### Notes
The project assumes valid inputs.<br />
 <br />
 
## Code_Sample_2 - Fourier Transform & Convolution
The main purpose of this program is to understand (hands-on) the concept of the frequency domain by performing simple manipulations on sounds and images.

### The program covers:
- Implementing DFT (Discrete Fourier Transform) on 1D and 2D signals.
- Performing sound fast forward.
- Performing image derivative.

### Related Files
- **sol2.py**: The program file.
- **requirements_2.txt**: The requirements file for the program. 

### Notes
- The project assumes valid inputs.
- Methods and variables names are not consistent due to the program API's requirements. <br />


 <br />
 
## Code_Sample_3 - Pyramid Blending
The main purpose of this program is to deal with the applications of pyramids, low-pass and band-pass filtering in image proccessing.

### The program covers:
- Construction of Gaussian and Laplacian pyramids.
- Ipmlementing pyramid blending.

### Related Files
- **sol3.py**: The program file.
- **requirements_3.txt**: The requirements file for the program. 
- **externals**: "catan512.jpg" "example_outcome.jpg" "trump_big_512.jpg" "catan_trump_mask.jpg"; being used in the blending example.

### Program's Outcomes
**Input Images**:
<img src="Code_Sample_3/externals/trump_big_512.jpg" alt="input1" width="200"/>
![input1](Code_Sample_3/externals/trump_big_512.jpg)
<img src="Code_Sample_3/externals/catan512.jpg" alt="input2" width="200"/>
![input2](Code_Sample_3/externals/catan512.jpg)

**Blending Output**:
<img src="Code_Sample_3/externals/example_outcome.jpg" alt="outcome" width="300"/>



### Notes
- The project assumes valid inputs.

