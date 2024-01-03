## AdaptAE: Running the AdaptAE model

The AdaptAE model is the proposed framework that is tested against the baseline models (autoencoder and ELM-AE)

### Usage
```bash
python train-adapt-ae.py [-h] [--mode {sample,batch}] 
                         --dataset {mnist,fashion-mnist,cifar10,
                         cifar100,super-tiny-imagenet,tiny-imagenet} 
                         [--batch-size BATCH_SIZE] [--device {cpu,mps,cuda}]
                         [--seq-prop SEQ_PROP] [--generate-imgs] 
                         [--save-results] [--phased] 
                         [--result-strategy {batch-size,seq-prop,
                         all-hyper,latent,all}] [--num-images NUM_IMAGES] 
                         --task {reconstruction,anomaly-detection}

Training a AdaptAE model

options:
  -h, --help            show this help message and exit

  --mode {sample,batch}
                        The mode of sequential training (either 'sample' or 'batch')

  --dataset {mnist,fashion-mnist,cifar10,cifar100,super-tiny-imagenet,tiny-imagenet}
                        The dataset to use (either 'mnist', 'fashion-mnist', 
                        'cifar10', 'cifar100', 'super-tiny-imagenet' or 'tiny-imagenet')

  --batch-size BATCH_SIZE
                        The batch size to use. 
                        Defaults to 10 if not provided

  --device {cpu,mps,cuda}
                        The device to use (either 'cpu', 'mps' or 'cuda'). 
                        Defaults to 'cuda' if not provided

  --seq-prop SEQ_PROP   The sequential training data proportion. 
                        Must be between 0.01 and 0.99 inclusive. 
                        Defaults to 0.97 if not provided

  --generate-imgs       Whether to generate images of the reconstructions

  --save-results        Whether to save the results to a CSV file

  --phased              Whether to monitor and save phased or total performance results

  --result-strategy {batch-size,seq-prop,all-hyper,latent,all}
                        If saving results, the independent variable to vary when saving results

  --num-images NUM_IMAGES
                        The number of images to generate. Defaults to 5 if not provided

  --task {reconstruction,anomaly-detection}
                        The task to perform (either 'reconstruction' or 'anomaly-detection')
```

### Example
`python train-adapt-ae.py --mode sample --dataset mnist --task reconstruction`
