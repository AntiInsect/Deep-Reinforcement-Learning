# week-four-report

Following the work of last week, I finished lecture 4, 5, 6 of David’s RL Class and review about the MC method in order to better understand its using in RL (code in appendix)
Also, I found the “Bias and Variance Tradeoff” very interesting and dig into it a bit.
Due to coming exam, I had no chance to read and gather a lot of resource like before but I did find some useful Python resource to help enhance my skills.

## Bias and Variance Tradeoff

* They are the mainly considered ::prediction errors::
* build accurate models but also to avoid the mistake of over-fitting and under-fitting

### Bias is the difference between the average prediction of our model and the correct value which we are trying to predict

* Model with high bias pays very little attention to the training data and 
* oversimplifies the model.
* It always leads to high error on training and test data.

### Variance is the variability of model prediction for a given data point or a value which tells us spread of our data

* Model with high variance pays a lot of attention to training data and does not generalize on the data which it hasn’t seen before.
* As a result, such models perform very well on training data but has high error rates on test data.

* figures that inspires intuition about the effect of Bias and Variance

![](week-four-report/70416304-930B-4A84-9F10-D2F0CE531688.png)
![](week-four-report/3FC32738-5486-4955-8408-FE990A8945C1.png)

* If our model is ::too simple and has very few parameters:: then it may *have high bias and low variance*. On the other hand if our model has ::large number of parameters:: then it’s going to have *high variance and low bias*. So we need to find the right/good balance without overfitting and under-fitting the data.
