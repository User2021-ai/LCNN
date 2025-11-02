# Learning to rank influential nodes in complex networks via convolutional neural networks
 
## LCNN
<p align="justify"> 
The repository contains an implementation of LCNN framework that uses convolutional neural networks and node-local representations to identify influential nodes in complex networks.
</p> 
The associated paper to this repository can be found here:
<a href="https://rdcu.be/dAfVw" > Learning to rank influential nodes in complex networks via convolutional neural networks </a> 

## Abstract
<p align="justify">
Identifying influential nodes is crucial for enhancing information diffusion in complex networks. Several approaches have
been proposed to find these influential nodes based on the network structure that significantly impacts the node influence.
Recently, several deep learning algorithms have also been introduced to identify influential nodes based on network exploration
and node feature selection. However, this has led to challenges in enhancing efficiency and minimizing computation time. To
address these challenges, we propose a novel framework called LCNN that uses convolutional neural networks and node-local
representations to identify influential nodes in complex networks. We argue that we can measure node influence capacity
using multi-scale metrics and a node’s adjacent matrix of one-hop neighbors to improve extracted information while reducing
running time. According to the susceptible-infectious-recovered (SIR) model, the experiment results demonstrate that our
proposed LCNN outperforms the state-of-the-art methods on both real-world and synthetic networks. Additionally, it exhibits
a moderate time consumption, which makes it suitable for large-scale networks.
</p>

## Keywords
 Influential nodes · Complex network · Information diffusion · Convolutional neural networks
 
 
![An overview of the LCNN framework. (a) An example of a toy
network with 6 nodes and 7 edges, where node weighting is determined
by multi-scales of degree and H-index, and the adjacency matrix with
size L = 3 is extracted based on only one-hop neighborhood. (b) Con-
structing two structural channel sets of a node based on the extracted
weights. (c) Model prediction of node influence capacity via convolu-
tional neural networks and node-local representation](https://github.com/User2021-ai/LCNN/blob/main/LCNN%20framework%20.svg)

An overview of the LCNN framework. (a) An example of a toy network with 6 nodes and 7 edges, where node weighting is determined by multi-scales of degree and H-index, and the adjacency matrix with size L = 3 is extracted based on only one-hop neighborhood. (b) Constructing two structural channel sets of a node based on the extracted weights. (c) Model prediction of node influence capacity via convolutional neural networks and node-local representation


# How to Cite
Please cite the following paper:<br>
<a href="https://rdcu.be/dAfVw" > Learning to rank influential nodes in complex networks via convolutional neural networks </a> 
 


 ```ruby
Ahmad, W., Wang, B. & Chen, S. Learning to rank influential nodes in complex networks via convolutional neural networks. Appl Intell (2024). https://doi.org/10.1007/s10489-024-05336-x

```
BibTeX
```ruby
@article{ahmad2024learning,
  title={Learning to rank influential nodes in complex networks via convolutional neural networks},
  author={Ahmad, Waseem and Wang, Bang and Chen, Si},
  journal={Applied Intelligence},
  pages={1--19},
  year={2024},
  publisher={Springer}
}
``` 
 
 
 
 
 



