# Journal

## 2024/01/07
Working on the Kaggle housing dataset.

Using a MLP with 1 hidden layer and a ReLU activation. The model's outputs are improving (w.r.t. the loss), but it seems entirely at the wrong scale. It's outputting values in the scale of 1e1 when obviously housing prices are in the scale of 1e5 to 1e6. Going to try setting the bias of the output layer to the mean of the sale price of the dataset.

> init well. Initialize the final layer weights correctly. E.g. if you are regressing some values that have a mean of 50 then initialize the final bias to 50. If you have an imbalanced dataset of a ratio 1:10 of positives:negatives, set the bias on your logits such that your network predicts probability of 0.1 at initialization. Setting these correctly will speed up convergence and eliminate “hockey stick” loss curves where in the first few iteration your network is basically just learning the bias.
from: https://karpathy.github.io/2019/04/25/recipe/

- Going to also try scaling the target during pre-processing and after inference
- Convergence speed is also heavily affected by the learning rate...was using 1e-3 before, 1e-2 helped speed it up a lot (duh)

## 2023/12/26

Training a standard softmax regression on FashionMNIST.
- Initializing weights of the single linear layer using a standard gaussian leads to poor perf (~40% accuracy even when increasing number of epochs).
- When initialization code is removed entirely to rely on torch's default initialization, perf increased significantly (to ~65% with way fewer epochs). Based on the default, which is way lower in magnitude, it's likely because of the scale of initial weights. To test this, I put back the weight initialization code, but this time instead of using `std=1.0`, I used `std=0.1`, and perf increased to ~58% using the same settings otherwise as the default initialization.

This is the explanation given by ChatGPT
> Scale of the Weights: In a Gaussian distribution with a mean of 0 and a standard deviation of 1, a significant portion of the weights will be relatively large in magnitude (since values can range significantly above and below 0). When inputs are multiplied by these large weights, the resulting activations can become very large, leading to exploding gradients. Conversely, if the weights are too small, it can lead to vanishing gradients.