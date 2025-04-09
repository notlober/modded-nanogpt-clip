# Modded-NanoGPT-CLIP

This repository hosts a fork of *NanoGPT speedrun* for research purposes.

Currently, we are using learned clip on mlp activations, which improves the convergence by %20-25 relative to iteration count, for example, with learned clip, we are able to get down to 3.28 loss by around 1475 iterations, while original gets the same loss with 1770 iterations, our gets 3.21 loss on 1770.

Current code just uses relu6 and gets a loss around 3.24 at 1770 iterations and 3.28 by 1600, If you want to also experiment best version just swap the mlp code with this, the reason i didnt include best because its a little slower in time:

```
class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_clip_min = CastedLinear(dim, hdim)
        self.c_clip_max = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = torch.min(torch.max(self.c_fc(x), self.c_clip_max(x)), self.c_clip_min(x)).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x
```

`step:1495/1770 val_loss:3.2778 train_time:1583613ms step_avg:1059.27ms
`
`step:1770/1770 val_loss:3.2129 train_time:1887538ms step_avg:1066.41ms
`

You can run current code with a single h100:

```
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade
python data/cached_fineweb10B.py 8 # downloads only the first 800M training tokens to save time
./run.sh
```
