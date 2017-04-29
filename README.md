# 17spr_wayman_mathew

2017 Spring CSE 847 class project repository for Eric Wayman and Adi Mathew

## Proposal ##

We have decided to focus on the fundamentals and theory behind the work done by Ba, et. al. in their paper [**Using Fast Weights to attend to the recent past**](https://papers.nips.cc/paper/6057-using-fast-weights-to-attend-to-the-recent-past.pdf), which was published at [NIPS 2016](https://papers.nips.cc/paper/6057-using-fast-weights-to-attend-to-the-recent-past).

Our current goal is to study the methods and replicate at least one comparitive test against other methods.

## Intermediate ##

Since the proposal, our efforts have diverged into both understanding the foundational principals behind the vanishing gradient problem as well as understanding implementations of standard Recurrent Neural Networks (RNN) for sequential data.
Upon discovering we were in fact not required to use MATLAB for the implementation, we switched to translating our existing code into Python.
Our implementation aims to use only [Numpy](http://www.numpy.org) and **not** existing Machine Learning/Deep Learning libraries.

## Final ##

Continuing on work from the previous report, we explored a Dynamical systems approach to RNNs, associative memory and derived the foundations underlying back propagation through time. Our implementation eventually used Tensorflow due to time constraints.
