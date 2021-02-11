**Week 1**

- Image representation

  - Image is an array of numbers
  - number indicates intensity: [0,255] for 8 bit representation
  - can be quantized from 0 to 255
  - can be normalized from 0 to 1 

- HSV

  - Hue: color in the angle in the HSV cone
  - Saturation: amount of 'grey' in the color, 0 is grey, 1 is pure primary color; small saturation means faded color
  - Value: represents intensity

- Image filtering

  - convolution kernel slides over the entire image to produce the output

  - padding handles boundary conditions

  - stride: how many pixels to shift the filter kernel in each step

    ![image-20210126221949001](C:\Users\ianli\AppData\Roaming\Typora\typora-user-images\image-20210126221949001.png)

    ![image-20210126223539047](C:\Users\ianli\AppData\Roaming\Typora\typora-user-images\image-20210126223539047.png)

  - cohort exercise: 30 31 33

    ​                          31 33 33

    ​                          33 33 32

- Additive gaussian noise (to model how noise is generated)

  ```
  I (distorted) = I (original) + n 
  
  where noise n is a sample from a normal distribution ~p(n; U, A)
  U: mean
  A: std dev
  ```

- Median filtering

  - non-linear filtering: use the median as the filter output; more robust against outliers

- Edge detection

  - edge: significant changes in intensity values 
  - prewitt operator (as the filter kernel/mask)
    - detect horizontal, vertical edges
    - compute the difference of the neighboring pixels to indicate likelihood of edges 
  - sobel operator (as the filter kernel/mask)
    - larger weight near the center pixels of the filter kernel 

- Image histogram

  - applying a transformation T to distribute the intensities evenly over the range -> increase contrast
  - a mapping of pixel value (no. of pixels in the histogram remains the same after transformation)
  - for a pixel with intensity k, transform it using (L= 256)
  - ![image-20210131004029792](C:\Users\ianli\AppData\Roaming\Typora\typora-user-images\image-20210131004029792.png)

- Image frequency

  - image frequency can be obtained quantitatively using 2D Fourier Transform, which decomposes the image to since and cosine components

  - identify vertical, horizontal spatial frequencies 

  - low-pass filtering retains low spatial frequency components, removes high spatial frequency components (noise, texture)

    - larger filter kernel - gives a 'smoothed' image output
- Gaussian function: use of different weights in filter kernels
    

- ![image-20210201163654690](C:\Users\ianli\AppData\Roaming\Typora\typora-user-images\image-20210201163654690.png)

