# Journal

## 2023/12/26

Training a standard softmax regression on FashionMNIST.
- Initializing weights of the single linear layer using a standard gaussian leads to poor perf (~40% accuracy even when increasing number of epochs).
- When initialization code is removed entirely to rely on torch's default initialization, perf increased significantly (to ~65% with way fewer epochs). Based on the default, which is way lower in magnitude, it's likely because of the scale of initial weights. To test this, I put back the weight initialization code, but this time instead of using `std=1.0`, I used `std=0.1`, and perf increased to ~58% using the same settings otherwise as the default initialization.

This is the explanation given by ChatGPT
> Scale of the Weights: In a Gaussian distribution with a mean of 0 and a standard deviation of 1, a significant portion of the weights will be relatively large in magnitude (since values can range significantly above and below 0). When inputs are multiplied by these large weights, the resulting activations can become very large, leading to exploding gradients. Conversely, if the weights are too small, it can lead to vanishing gradients.