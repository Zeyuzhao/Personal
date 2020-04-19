---
title: Clearing Kidney Exchanges via Graph Neural Network Guided Tree Search
date: 2020-03-30
category: project
tags:
  - deep-learning
  - pytorch
  - kidney-exchange
externalLink: false
image: "/assets/images/splash.jpg"
headerImage: true
projects: true
hidden: false
description: A geometric deep learning approach to solving kidney exchanges
author: zachzhao
---


## Summary
Over the summer, I worked on applying deep learning techniques to help solve kidney exchanges.

## Equations

We use the following propagation rule:

$$H^{(l+1)}_i = ReLU(\theta^{(l)}_{1}H^{(l)}_i + \phi(\{\theta^{(l)}_{2}H^{(l)}_a | a \in N(i)\})$$

We use the following cross-entropy loss:
$$L(\textbf{1}, G) = \sum_{i = 1}^{n}[\textbf{1}_{i}log(f_{\theta}(G)) + (1 - \textbf{1}_{i})log(1 - f_{\theta}(G))]$$

## Graphics



{% include mathjax.html %}
