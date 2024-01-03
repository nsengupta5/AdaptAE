## AdaptAE: Running the ELM-AE model

The ELM-AE model is used as the state of the art baseline model for the AdaptAE framework. 

### Usage
```bash
python train-elm-ae.py [-h] --dataset {mnist,fashion-mnist,cifar10,cifar100,
                       super-tiny-imagenet,tiny-imagenet} 
                       [--device {cpu,mps,cuda}] [--num-images NUM_IMAGES] 
                       [--generate-imgs] [--save-results]
                       [--result-strategy {latent,all}] 
                       --task {reconstruction,anomaly-detection}
Train an ELM-AE model

options:
  -h, --help            show this help message and exit

  --dataset {mnist,fashion-mnist,cifar10,cifar100,super-tiny-imagenet,tiny-imagenet}
                        The dataset to use 
                        (either 'mnist', 'fashion-mnist', 'cifar10', 'cifar100', 
                        'super-tiny-imagenet' or 'tiny-imagenet')

  --device {cpu,mps,cuda}
                        The device to use (either 'cpu', 'mps' or 'cuda'). 
                        Defaults to 'cuda' if not provided

  --num-images NUM_IMAGES
                        The number of images to generate. Defaults to 5 if not provided

  --generate-imgs       Whether to generate images of the reconstructions

  --save-results        Whether to save the results to a CSV file

  --result-strategy {latent,all}
                        If saving results, the independent variable to vary when saving results

  --task {reconstruction,anomaly-detection}
                        The task to perform (either 'reconstruction' or 'anomaly-detection')
```

### Example
`python train-elm-ae.py --dataset cifar10 --task anomaly-detection`
