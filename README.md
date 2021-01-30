# SVM-from-scratch

This repository extends the functionality of julias Convex library (https://jump.dev/Convex.jl/v0.13.2/examples/general_examples/svm/).


* Juicy History behind Support Vector Machines 

Support Vector Machines are one of the most robust classification methods (can be regression too) we have today. 
However SVM's aint new, they are here from the 90's. They are used in the context of Suppervised Learning, since you need to have labeled data to train them.
It is even prooven that an SVM algorithm can solve every possible learning problem, if you have infinite computational resources or time. In reality as the number of trainable parameters increases, the complexity of the algorithm is escalating hyper-quickly! There are various smart tricks we do alongside SVM's, such as the Kernel trick (my mind
was blown when I realised what it does, read here: https://medium.com/@zxr.nju/what-is-the-kernel-trick-why-is-it-important-98a98db0961d) which makes them still relevant in modern times. In fact It is one of the commonly used classification algorithms in industrial and research applications.



* Our Task

As in all classification tasks, here we need to classify our data into some predifined classes (Negative and Positive). For simplicity shake this repository implements a simple 
2D classification scenario (where the data features are x, y). In order to achieve that we build a Linear SVM (a single line classifier), trying to classify as good as possible
two linearly seperable distributions. I think it's always good to implement classifiers in these kind of datasets, in order to understand the behaviour of your classifier
easier. The distributions I made are normal distributions arround (10,10) for Negatives and (30,30) for Positives. 

```julia
using Distributions: MvNormal
# positive data points
pos_data = rand(MvNormal([30,30], 5.0), N);


# negative data points
neg_data = rand(MvNormal([10, 10], 5.0), M)
```

<img src="https://github.com/stefgina/svm-from-scratch/blob/main/imgs/distributions.png" width=400 height=250>












