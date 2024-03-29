# SVM-from-scratch

```python

Author -> Stefanos Ginargyros

```

Implemented in Julia, however it is easy to transfer in any language you like if you follow along.
This repository extends the usability and functionality of julias [Convex library](https://jump.dev/Convex.jl/v0.13.2/examples/general_examples/svm/).


## 1. History behind Support Vector Machines 

Support Vector Machines are one of the most robust classification methods (can be regression too) we have today. 
However SVM's aint new, they are here from the 90's. They are used in the context of Suppervised Learning, since you need to have labeled data to train them.
It is even prooven that an SVM algorithm can solve every possible learning problem, if you have infinite computational resources or time. In reality as the number of trainable parameters increases, the complexity of the algorithm is escalating hyper-quickly! There are various smart tricks we do alongside SVM's, such as the Kernel trick (my mind
was blown when I realised what it does, read [here](https://medium.com/@zxr.nju/what-is-the-kernel-trick-why-is-it-important-98a98db0961d)) which makes them still relevant in modern times. In fact It is one of the commonly used classification algorithms in industrial and research applications.



## 2. The Task

As in all classification tasks, here we need to classify our data into some predifined classes (Negative and Positive). For simplicity shake this repository implements a simple 
2D classification scenario (where the data features are x, y). In order to achieve that we build a Linear SVM (a single line classifier), trying to classify as good as possible
two linearly seperable distributions. I think it's always beneficial to implement classifiers in these kind of datasets, in order to conceptualize their behaviour easier. The distributions I made are normal distributions arround (10,10) for Negatives and (30,30) for Positives. 

 ```julia
using Distributions: MvNormal
# positive data points
pos_data = rand(MvNormal([30,30], 5.0), N);


# negative data points
neg_data = rand(MvNormal([10, 10], 5.0), M)
```

<img src="https://github.com/stefgina/svm-from-scratch/blob/main/imgs/distributions.png" width=400 height=250>


## 3. Math behind SVM

The essence of our SVM is to estimate a line that seperates the two distributions
the best possible way. "Best possible way" mathematically means three things:

* Maximize the margin between the decision boundary (the seperating line) and both of the distribution outliers.

   Well the margin is 1/norm(w), so if we flip it we just have to Minimize norm(w), or 1/2*norm(w)^2

   ```julia
   Min(1/2*sumsquares(w))
   ```

* Negative Points shouldn't be classified as Positive.

   A Negative point missclassification is every Negative point above our decision boundary. Can be written as 

   ```julia
   max(1+b-w'*pos_data, 0)
   ```


* Positive Points shouldn't be classified as Negative.

   A Positive point missclassification is every Positive point bellow our decision boundary. Can be written as 

   ```julia
   max(1-b+w'*neg_data, 0)
   ```

Finally the whole objective function is constructed, and we can minimize it using the convex solver in SCS package.

```julia

obj = 1/2*sumsquares(w) + C*sum(max(1+b-w'*pos_data, 0)) + C*sum(max(1-b+w'*neg_data, 0))
problem = minimize(obj)
solve!(problem, solver)

```


## 4. Results

When the training ends (meaning the Convex Solver optimization) on our distribution dataset, the classifier has found the best seperating line (meaning the best w and b).
The decision boundary is shown bellow in green. I also extended the code to ask an input from the user and classify it accordingly (the yellow point bellow).


 
<img src="https://github.com/stefgina/svm-from-scratch/blob/main/imgs/classified.png" width=400 height=250>

















