
Data preprocessing:

```bash
OMP_NUM_THREADS=2 th -i provider.lua
```

```lua
provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)
```
Takes about 30 seconds and saves 1400 Mb file.

Training Baseline model:

```bash
th traincolorize.lua --model colorize -s logs/colorize
```

Training adversarial model:

```bash
th traingan.lua --model gan -s logs/gan
```

Test baseline model on image:

```bash
th forward.lua -l logs/colorize -t -i 100
```

Test adversarial model on image:

```bash
th forwardgan.lua -l logs/gan -t -i 100
```
