## Introduction

The library comes in two parts:

1. distillation.labeller
2. distillation.utils
3. distillation.dataloaders

The basic idea is that you have a trained model that can be loaded in PyTorch.
If you do, you call `labeller.label_data(data_loader, checkpoint_filename)`,
and the
