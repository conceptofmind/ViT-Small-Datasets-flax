## Vision Transformer for Small Datasets

<img src="./images/vit_for_small_datasets.png" width="400px"></img>

"Recently, the Vision Transformer (ViT), which applied the transformer structure to the image classification task, has outperformed convolutional neural networks. However, the high performance of the ViT results from pre-training using a large-size dataset such as JFT-300M, and its dependence on a large dataset is interpreted as due to low locality inductive bias. This paper proposes Shifted Patch Tokenization (SPT) and Locality Self-Attention (LSA), which effectively solve the lack of locality inductive bias and enable it to learn from scratch even on small-size datasets. Moreover, SPT and LSA are generic and effective add-on modules that are easily applicable to various ViTs. Experimental results show that when both SPT and LSA were applied to the ViTs, the performance improved by an average of 2.96% in Tiny-ImageNet, which is a representative small-size dataset. Especially, Swin Transformer achieved an overwhelming performance improvement of 4.08% thanks to the proposed SPT and LSA." - Seung Hoon Lee, Seunghyun Lee, Byung Cheol Song

### Research Paper:
- https://arxiv.org/abs/2112.13492

### Official repository:
- https://github.com/aanna0701/SPT_LSA_ViT

### Usage:
```python
import numpy as np

key = jax.random.PRNGKey(0)

img = jax.random.normal(key, (4, 256, 256, 3))

v = ViT(
    image_size = 256,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

init_rngs = {'params': jax.random.PRNGKey(1), 
            'dropout': jax.random.PRNGKey(2), 
            'emb_dropout': jax.random.PRNGKey(3)}

params = v.init(init_rngs, img)
output = v.apply(params, img, rngs=init_rngs)
print(output.shape)

n_params_flax = sum(
    jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
)
print(f"Number of parameters in Flax model: {n_params_flax}")

spt = SPT(
    dim = 1024,
    patch_size = 16
)

spt_params = spt.init(init_rngs, img)
spt_output = spt.apply(spt_params, img, rngs=init_rngs)
print(spt_output.shape)
```

### Citation:
```bibtex
@misc{https://doi.org/10.48550/arxiv.2112.13492,
  doi = {10.48550/ARXIV.2112.13492},
  
  url = {https://arxiv.org/abs/2112.13492},
  
  author = {Lee, Seung Hoon and Lee, Seunghyun and Song, Byung Cheol},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Vision Transformer for Small-Size Datasets},
  
  publisher = {arXiv},
  
  year = {2021},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```