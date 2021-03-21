# Generative-Image-Inpainting-with-Contextual-Attention
The PyTorch reimplementation of the paper *Generative Image Inpainting with Contextual Attention* by Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S. Huang, **(2018)** [arXiv](https://arxiv.org/abs/1801.07892).

This implementation heavily relies on Contextual Attention Layer implementaion from https://github.com/daa233/generative-inpainting-pytorch.


### Ideas for the experiments
1. Progressive growing mask
2. Multi-scale inference and network to merge the results
3. Sequential inference from outside inside

Also need to train on the Flickr dataset, to get more real-life results