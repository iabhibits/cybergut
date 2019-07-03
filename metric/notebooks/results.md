# Following are the results that compare three metrics on different version of synthetic dataset

In all below experiments 100 source images were generated independently for each experiment. Each image contained 25 circles of random radius~Uni(10,50) such that minimum distance between two circles was 10 pixels.

## Experiment 1
In this experiment first for sanity check both source and target were kept same. Without any surprise all scores were 1.0.

Then each target was perturbed in a minor way. See below for an example.

![Figure 1: Source(left) and target(right)](expr1_data/compare/0.png)

Below are the results for this experiment.

| Metric 	 | mean score|
|----------- |----------:|
| IoU		 	|	0.8262 |
| Adjusted Rand Index | 0.8619|
| New Metric | 0.9419 |


### Conclusion:
 One of the reasons IoU is not ideal for cell segmentation is that ground truth is not perfect. IoU is unable to capture this phenomena. Here although each cell has been detected it wasn't captured in IoU which is just 0.82. 
 
 New metric is able to capture this suitabilities, although not perfectly.
 
## Experiment 2
In this experiment some of the cells (N~Uni(2, 6)) were removed from target image to create source. Second column shows results on unperturbed targets. Third column shows results when targets were perturbed like experiment 1.

![](expr2_data/compare/3.png)

| Metric | mean score on unperturbed version | mean score on perturbed version |
|---|---:|---:|
|IoU | 0.87 | 0.74 |
|Adjusted Rand Index | 0.90 | 0.79 |
|New Metric | 1.0 | 0.94 |

### Conclusion:

When groundtruth misses some cells, which is common when there are many cells in a view, IoU and ARI perform poorly. On the other hand since New Metric ignores regions which were not present in groundtruth, it behaves similarly to experiment 1 which can be seen form last columns of table 1 and table 2.

Note that ignoring cells not found in groundtruth can be dangerous. My justification in doing so is that we will be using some other loss function while training which doesn't ignore these cases and penalizes network. I also assume that human annotator doesn't miss that many cells in groudtruth, otherwise training the network for segmentation is ill-defined problem.  


## Experiment 3
![](expr3_data/compare/0.png)

In this experiment two cells with minimum distance between their perimeter were merged into one blob, by increasing their radius. Other cells remains untouched. This kind of emulates the network when cells are adjacent to each other.

| Metric | avg. Score |
|---|---:|---:|
|IoU | 0.98 |
|Adjusted Rand Index | 0.97 | 
|New Metric | 0.94 |

### Conclusion
Although real network's output will have many such cells merged together, here we are only considering two cells merging together. Ideally a metric should penalize such cases. But IoU doesn't do that as it has no idea of cells. It only looks at foreground and background. ARI should help in this case. But in the case when merged cells don't occupy much area it doesn't help much. New Metric however, as can be seen from a table, penalizes such cases. 

## Experiment 4
This experiment shows effects of merging of more cells. Here we vary N from 1 to 15. N is number of times merging occures. So N == 1 refers to experiment 3.
![](expr4_data/Figure_1.png)
## Experiment 5
Here for each image N was chosen randomly from Uni(1, 10)
![](expr5_data/compare/0.png)

| Metric | avg. Score |
|---|---:|---:|
|IoU | 0.88 |
|Adjusted Rand Index | 0.89 | 
|New Metric | 0.73 |

### Conclusion
This experiment kind of relates to real world situation (except the missing cells, but new metric ignores them anyway). We can see that among all metrics New Metric penalizes the most. 