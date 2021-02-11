**Wk 2**

- **Histogram Equalization**

  - for an input of [[1 3 1 3]

    ​						[2 3 10 11]

    ​						[11 10 2 3]

    ​                        [1 2 3 3]]

    output: [[0 176 0 176]

    ​			 [58 176 215 255]

    ​			 [255 215 58 176]

    ​			 [0 58 176 176]]

  - ![image-20210202104357529](C:\Users\ianli\AppData\Roaming\Typora\typora-user-images\image-20210202104357529.png)

  - ![image-20210202105043509](C:\Users\ianli\AppData\Roaming\Typora\typora-user-images\image-20210202105043509.png)

  - **Image Classification**

    - given an input image, the algorithm produces one label from a fixed set of classes

    - Top-n accuracy: algorithm outputs k confidence for each of the k classes 

    - Linear Classifier

      ```
      s = f(x; W, b) = Wx + b
      
      Input x: D x 1
      Weight W: K x D
      Bias b: K x 1
      Score s: K x 1
      where K is the number of classes and D is the number of pixels
      ```

      - Loss function: measure how consistent are the ground-truth labels and the score function outputs, for some W

      - Softmax classifier with cross-entropy loss

      ​    ![image-20210203144408056](C:\Users\ianli\AppData\Roaming\Typora\typora-user-images\image-20210203144408056.png)

      - Entire loss for N training samples (xi, yi) :

        - L = 1/N sum(Li)

      - ![image-20210203013904989](C:\Users\ianli\AppData\Roaming\Typora\typora-user-images\image-20210203013904989.png)

      - why exp before normalization?

        - exp() > 0 so it allows for a much higher confidence if the activation is large

      - Cross-entropy loss interpretation

        - model estimated prob Q: softmax(f)
        - data true prob P: [0,0,...1,...0] at the yi-th position
        - measured by the KL divergence (Dkl = 0 when P, Q are the same)
          - Dkl (P||Q) = -sum( P(i) log (Q(i) / P(i)) )

      - Entire loss for N training samples (xi, yi) :

        - L = 1/N sum(Li)

      - ![image-20210203013904989](C:\Users\ianli\AppData\Roaming\Typora\typora-user-images\image-20210203013904989.png)

      - why exp before normalization?

        - exp() > 0 so it allows for a much higher confidence if the activation is large

      - Cross-entropy loss interpretation

        - model estimated prob Q: softmax(f)
        - data true prob P: [0,0,...1,...0] at the yi-th position
        - measured by the KL divergence (Dkl = 0 when P, Q are the same)
          - Dkl (P||Q) = -sum( P(i) log (Q(i) / P(i)) )

        - **Deep Learning**
        - stacking linear classifiers to improve representational power 
          - adding non-linearity between layers
            - by introducing activation function ie sigmoid: 1/1 + e^-x
              - tanh (saturation, vanishing gradient, slow and difficult to train with gradient descent)
              - ReLU (reduce issue of vanishing gradient)
            - additional normalization
          - neural network: collections of neurons
            - neuron is a computational unit ie. sigma . sum(Wi . Xi + b) , where sigma is the activation function 
            - connected in an acyclic graph
            - output of a neuron can be input of another 
            - hidden layers: values are not observed in the training set
            - output layer: no activation