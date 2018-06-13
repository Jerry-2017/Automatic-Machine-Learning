This Project is a new algorithm for automatic machine learning.


## Idea
 
The Neural Network Architecture is represented by Incidence Matrix (IM). Incidence Matrix is represented by KMM tensor where K is number of supported operators and M is number of nodes in the graph. 
In the neural network, several size of convolution filter will be applied to the Matrix, followed by Subsampling. MLP is placed between CNN and output. 
At Start, there is some input and indirect defined output. Everytime, an edge will be connected based on the prediction. The model identified by IM will be trained for several times and record its value. Then Temporal Difference Updating rule is applied to update the reward. 

## Expectation : 

Incremental Training could reduce training cost.
CNN could capture the fundamental base structure of successful neural architecture.
Matrix could somehow play the rule of identifier. 

Matrix Unified Algorithm : 
Each node is identified by 1. degree 2. degree of neighbour 3. depth from each input 
Sort the identifier from the lowest to highest. 