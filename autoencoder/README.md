## AdaptAE: Running the Autoencoder model

The autoencoder model is used as the baseline model for the AdaptAE framework. 

### Usage
```bash
python train-autoencoder.py [-h] --dataset {mnist,fashion-mnist,cifar10,cifar100,
                            super-tiny-imagenet,tiny-imagenet} 
                            [--device {cpu,mps,cuda}] [--generate-imgs] 
                            [--save-results] [--num-images NUM_IMAGES]
                            [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE] 
                            [--result-strategy {batch-size,num-epochs,all-hyper,latent,all}] 
                            --task {reconstruction,anomaly-detection}

Train an autoencoder model

options:
  -h, --help            show this help message and exit

  --dataset {mnist,fashion-mnist,cifar10,cifar100,super-tiny-imagenet,tiny-imagenet}
                        The dataset to use 
                        (either 'mnist', 'fashion-mnist', 'cifar10', 
                        'cifar100', 'super-tiny-imagenet' or 'tiny-imagenet')

  --device {cpu,mps,cuda}
                        The device to use (either 'cpu', 'mps' or 'cuda'). 
                        Defaults to 'cuda' if not provided

  --generate-imgs       Whether to generate images of the reconstructions

  --save-results        Whether to save the results to a CSV file

  --num-images NUM_IMAGES
                        The number of images to generate. 
                        Defaults to 5 if not provided

  --num-epochs NUM_EPOCHS
                        The number of epochs to train for. 
                        Defaults to 30 if not provided

  --batch-size BATCH_SIZE
                        The batch size to use. 
                        Defaults to 64 if not provided

  --result-strategy {batch-size,num-epochs,all-hyper,latent,all}
                        If saving results, the independent variable to vary when saving results

  --task {reconstruction,anomaly-detection}
                        The task to perform (either 'reconstruction' or 'anomaly-detection')

```

### Example
`python train-autoencoder.py --dataset mnist --num-epochs 50 --device cpu --task reconstruction`
