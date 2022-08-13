# TPG Autoencoder workflows
> Luigi workflows for training and evaluating fairly complex Trigger Primitive Generator Autoencoder models

This repository contains relatively generic workflows written in Python via the great `Luigi` library. These so-called pipelines should help to streamline the development process of model architectures and trainings of a complex model. The workflows are designed in a flexible manner to allow for:

1. Training multi-stage models, either freely or for specified stages at a time.
2. Either train from scratch or preload weights to any submodel.
3. Allow training on multiple objectives with specified "objective weight".
4. Allow data preprocessing before model training in various ways.
5. Generate unique training/model IDs for storing weights and, if desired, continue training at a later point.
6. Generate live training statistics per epoch via Tensorboard to monitor user-defined performance plots.

Furthermore, this repository also contains predefined models as a starting point for exploration. These models are written in an Object-oriented way which allows easy composition to train on more than one task.
