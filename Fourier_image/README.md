### 1. Fourier Transform Represents Images

Normally, we can represent any image as a superposition of sine/cosine waves. That is, any image can be seen as being made up of a series of sinusoidal gratings, each having a different amplitude, frequency, orientation, and phase. The 2D Fourier transform in Python enables you to deconstruct an image into these constituent parts, and you can also use these constituent parts to recreate the image, in full or in part. The very nature of how light travels and propagates is described through the Fourier transform.

**Fourier Transform Representation of Images**:
- The Fourier Transform is a mathematical tool that decomposes an image into its constituent frequencies. In a 2D image, these frequencies are sine and cosine waves of varying magnitudes and phases.
- **Spatial Domain to Frequency Domain**: In the spatial domain, an image is represented by the intensity of pixels. The Fourier Transform converts this into the frequency domain, representing the image as a sum of sinusoidal functions.
- **Frequency Components**: Low frequencies correspond to smooth, gradual changes in pixel intensity (large structures), while high frequencies correspond to abrupt changes (edges and fine details).

**Mathematical Expression**:
- For a 2D image \( f(x, y) \), the 2D Fourier Transform \( F(u, v) \) is given by:
  \[
  F(u, v) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) e^{-i 2\pi (ux + vy)} \, dx \, dy
  \]
- \( F(u, v) \) contains information about the amplitude and phase of each frequency component \( (u, v) \).

---

### 2. Gratings


**Sinusoidal Grating**:
   - A sinusoidal grating is a pattern where the intensity of the image varies sinusoidally along a specific direction.
   - The intensity \(I(x, y)\) at any point \((x, y)\) can be described by a sinusoidal function:
     \[
     I(x, y) = \sin\left(\frac{2\pi}{\lambda} (x \cos \theta + y \sin \theta)\right)
     \]
     where:
     - \(\lambda\) is the wavelength of the sine wave, determining the distance between successive peaks (or troughs) of the wave.
     - \(\theta\) is the angle at which the wave is oriented relative to the x-axis.
     - \(x\) and \(y\) are the coordinates on the grid.

**Spatial Frequency**:
   - The spatial frequency of the grating is inversely related to the wavelength \(\lambda\). It defines how many cycles (oscillations) of the wave occur per unit distance.
   - A higher spatial frequency means more oscillations per unit distance, resulting in a more densely packed grating.

**Purpose in Fourier Transform**:
   - A sinusoidal grating is often used in Fourier Transform studies because it represents a simple and well-defined frequency component in the spatial domain.
   - When the Fourier Transform is applied to a sinusoidal grating, it produces a distinct pattern in the frequency domain. This pattern helps illustrate how the transform decomposes an image into its constituent sine and cosine waves.


**Image Construction Using Gratings**:
- A composite image can be constructed by adding multiple sinusoidal gratings together, each with different wavelengths, orientations, and amplitudes.

- **Superposition Principle**: An image can be reconstructed by superimposing multiple sinusoidal gratings with different frequencies, amplitudes, orientations, and phases.
- **Inverse Fourier Transform**: The process of reconstructing an image from its frequency components involves the inverse Fourier Transform, which combines these sinusoidal components back into the spatial domain.

**Example**:
- In this [script](\basic_fourier), we generated a composite grating by adding several sinusoidal gratings:
  \[
  I(x, y) = \sum_{i} A_i \sin \left( \frac{2\pi}{\lambda_i} (x \cos \theta_i + y \sin \theta_i) \right) + \text{background}
  \]
- This composite grating can represent an image with specific patterns and textures.

---

### 3. Practical Limitations and Industry Use

**Advantages**:
- **Mathematical Foundation**: Provides a robust mathematical foundation for analyzing and processing images.
- **Frequency Analysis**: Useful for tasks like filtering, compression, and feature extraction based on frequency content.
- **Theoretical Insights**: Helps in understanding the structure and components of images.

**Challenges and Industry Adoption**:
- **Computational Complexity**: Fourier Transform operations, especially for large images, can be computationally intensive.

- **Practical Limitations**: Direct application in real-time image processing or large-scale image datasets is limited by performance constraints.
- **Alternative Methods**: Other image processing techniques (e.g., convolutional neural networks) are more suited for practical applications due to their efficiency and ability to learn hierarchical features from data.
> [!CAUTION]
> Using this technique to maually reconstrct an image via inverse FT after applying FT to a real image is too costly.It's very time consuming and requires around ~100k iterations for an image with good features.

---

### 4. Neural Networks and Fourier Transforms

**Usefulness in Neural Networks**:
- **Feature Extraction**: Fourier transforms can extract frequency-based features that are invariant to certain transformations, which can be useful for robust image recognition.
- **Frequency Domain Analysis**: Helps neural networks to focus on important frequency components, reducing the noise and improving performance.

**Integration with Neural Networks**:
- **Preprocessing Step**: Fourier transforms can be used as a preprocessing step to convert images to the frequency domain before feeding them into neural networks.
- **Hybrid Models**: Some architectures combine spatial and frequency domain information to leverage the strengths of both representations.

>[!NOTE]
> The model we obtain when we map the i/p features into fourier series and then use them as i/p to train is actually quite good.

---

### 5. Improved Model Accuracy with Fourier Transforms

**Training with Fourier Transforms**:
- **Enhanced Features**: Training a model on Fourier-transformed images can enhance its ability to recognize patterns based on frequency information.
- **Reduction of Noise**: Frequency domain representation can suppress irrelevant high-frequency noise, leading to better generalization.
- **Efficient Learning**: Models can learn more efficiently by focusing on significant frequency components, improving accuracy and convergence rates.

**Example**:
- Converting an image to its frequency domain representation and using this as input to a neural network can improve the network’s ability to detect edges and textures, leading to more accurate classification.

---

### 6. Challenges with Large Datasets: Curse of Dimensionality

**Curse of Dimensionality**:
- **Definition**: The curse of dimensionality refers to the exponential increase in volume associated with adding extra dimensions to Euclidean space. This means that as the number of dimensions grows, the amount of data needed to generalize accurately grows exponentially.
- **Impact on Fourier Transforms**: When using Fourier transforms, the frequency domain representation can become high-dimensional, especially for large images, making it challenging to train models effectively.

**Limitations for Large Datasets**:
- **Computational Resources**: Handling large datasets and high-dimensional Fourier-transformed data requires significant computational resources.
- **Overfitting**: With high-dimensional data, the risk of overfitting increases as the model may learn noise and irrelevant patterns.
- **Data Sparsity**: In high-dimensional spaces, data points become sparse, making it difficult for models to find meaningful patterns without large amounts of data.

**Summary**:
- Fourier feature mapping can be used to overcome the spectral bias of coordinate-based MLPs towards low frequencies by allowing them to learn much higher frequencies.
- Random Fourier feature mapping with an appropriately chosen scale can dramatically improve the performance of coordinate-based MLPs across many low-dimensional tasks in computer vision and graphics.
- While using Fourier transforms for small datasets can improve model accuracy, scaling this approach to large datasets is impractical due to the increased computational burden and the challenges posed by the curse of dimensionality.


>[!NOTE]
> While using images individually, to obtain complex patterns we need an higher order series and this series grows exponentially, which can sometimes result in obtaining an useless model. 

---

### How to train a model using FT?

What are the individual units that make up an image? One answer is pixels, each having a certain value. Another surprising one is sine functions with different parameters. Basically any two-dimensional (2D) image can be reconstructed using only sine functions and nothing else.

This paper [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/pdf/2006.10739) says " MLPs have difficulty learning high frequency functions, a phenomenon referred to in the literature as “spectral bias”. Neural Tangent Kernel theory suggests that this is because standard coordinate-based MLPs correspond to kernels with a rapid frequency falloff, which effectively prevents them from being able to represent the high-frequency content present in natural images and scenes. ", and they solve this problem by mapping input coordinates to a `Fourier Series`. They show that this techniques improves the performance of coordinate-based MLPs. 


**Ex**:
 Say we have a point $(x, y)$, and if we want to convert this into a Fourier series in 2D then we use the below eqation to convert this point into frequencies in $(n, m)$ dimensions, where $(n, m)$ are the `order` of the series.
   - [`cos(n * x) * cos(m * y)`, `cos(n * x) * sin(m * y)`, `sin(n * x) * cos(m * y)`, `sin(n * x) * sin(m * y)`]
 These features are the real and imaginary parts of the Fourier transform basis functions evaluated at the point `(x, y)`.


### Dataset: Mandelbrot Set

**Definition**:
- The Mandelbrot set is defined by the set of complex numbers \( c \) for which the complex numbers of the sequence \( z_n \) remain bounded in absolute value. The sequence \( z_n \) is defined by:
  \[
  z_0 = 0
  \]
  \[
  z_{n+1} = z_n^2 + c
  \]
- The modulus of a complex number is its distance to 0. In Python, this is obtained using `abs(z)` where `z` is a complex number. The sequence is considered unbounded if the modulus of one of its terms is greater than 2.

**Visualization**:
- The visual representation of the Mandelbrot set can be created by determining, for each point of a part of the complex plane, whether \( z_n \) is bounded within a range of 2 from zero. The number of iterations to reach a modulus greater than 2 can be used to determine the color to use.

**Training Neural Networks**:
- Training a neural network to recreate an image using just MLPs requires a minimum of 2.8M parameters, i.e., a neural network with at least 30 hidden layers each having around 300 neurons. This approach is feasible only for simple images and datasets, which is why we often shift to CNNs for more complex tasks.
- But here we try to explore how without using CNNs we can train a model using FT.


---

### Acknowledgments

This project was inspired by the work of various contributors from GitHub and other internet resources. I would like to thank the following individuals and sources for their valuable contributions and insights:

- **[Max Robinson]**: [mandelbrotnn](https://github.com/MaxRobinsonTheGreat/mandelbrotnn) - Modified some of the code from this repository. I was inspired by his youtube [video](https://youtu.be/TkwXa7Cvfr8?si=YgqcCX-lzjtAxJE2).

- **[Book](https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/)**:  - How to Create Any Image Using Only Sine Functions | 2D Fourier Transform in Python

---
