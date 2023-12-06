import logging
import matplotlib.pyplot as plt
import csv

"""
Visualize the original and reconstructed images
:param originals: The original images
:param reconstructions: The reconstructed images
:param dataset: The dataset used
:param n: The number of images to visualize
"""
def visualize_comparisons(originals, reconstructions, dataset, num_imgs, results_file):
    logging.info(f"Generating {num_imgs} images...")
    plt.figure(figsize=(20, 4))
    for i in range(num_imgs): # Display original images
        ax = plt.subplot(2, num_imgs, i + 1)
        if dataset in ["mnist", "fashion-mnist"]:
            plt.imshow(originals[i].reshape(28, 28))
        elif dataset in ["cifar10", "cifar100", "super-tiny-imagenet"]:
            plt.imshow(originals[i].reshape(3, 32, 32).transpose(1, 2, 0))
        else:
            plt.imshow(originals[i].reshape(3, 64, 64).transpose(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed images
        ax = plt.subplot(2, num_imgs, i + 1 + num_imgs)
        if dataset in ["mnist", "fashion-mnist"]:
            plt.imshow(reconstructions[i].reshape(28, 28))
        elif dataset in ["cifar10", "cifar100", "super-tiny-imagenet"]:
            plt.imshow(reconstructions[i].reshape(3, 32, 32).transpose(1, 2, 0))
        else:
            plt.imshow(reconstructions[i].reshape(3, 64, 64).transpose(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Save the images
    logging.info(f"Saving images to output file...")
    plt.savefig(results_file)

"""
Save the results to a CSV file
:param dataset: The dataset used
:param phased: Boolean indicating whether the model was monitored in a phased manner
:param result_strategy: The result strategy used
"""
def save_result_data(model, dataset, phased, result_strategy, result_data):
    target_dir = "phased" if phased else "total"
    with open (f'{model}/data/{target_dir}/{result_strategy}_{dataset}_performance.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_data)

"""
Print the header of a stage
:param header: The header to print
"""
def print_header(header):
    result_str = "\n" + header + "\n" + "=" * len(header)
    print(result_str)

"""
Exit the program with an error message of the correct usage
:param msg: The error message to display
:param parser: The parser to use to print the correct usage
"""
def exit_with_error(msg, parser):
    logging.error(msg)
    parser.print_help()
    exit(1)
