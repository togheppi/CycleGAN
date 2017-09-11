# CycleGAN
PyTorch implementation of [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)
* Dataset can be downloaded from [here](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/).
* Loss values are plotted using [Tensorboard in PyTorch](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard).

## horse2zebra dataset
* Image size: 256x256
* Number of training images: 1,334 for horse images, 1,067 for zebra images
* Number of test images: 120 for horse images, 140 for zebra images

### Results
* Adam optimizer is used. Learning rate = 0.0002, batch size = 1, # of epochs = 200:
* 6 resnet blocks used for Generator.

GAN losses<br> ( ![FF8900](https://placehold.it/10/FF8900/000000?text=+) : Discriminator A / ![AE0000](https://placehold.it/10/AE0000/000000?text=+) : Discriminator B <br> ![CD52A7](https://placehold.it/10/CD52A7/000000?text=+) : Generator A / ![2156C9](https://placehold.it/10/2156C9/000000?text=+) : Generator B <br> ![58BEE4](https://placehold.it/10/58BEE4/000000?text=+) : Cycle loss A / ![319F92](https://placehold.it/10/319F92/000000?text=+) : Cycle loss B ) | Generated images<br>(Input / Generated / Reconstructed)
:---:|:---:
<img src = 'horse2zebra_results/horse2zebra_CycleGAN_losses_epochs_200.png'> | <img src = 'horse2zebra_results/horse2zebra_CycleGAN_epochs_200.gif'>

* Generated images using test data

    |Horse to Zebra<br>1st column: Input / 2nd column: Generated / 3rd column: Reconstructed|
    |:---:|
    |![](horse2zebra_test_results/AtoB/Test_result_2.png)|
    |![](horse2zebra_test_results/AtoB/Test_result_11.png)|
    |![](horse2zebra_test_results/AtoB/Test_result_50.png)|
    |![](horse2zebra_test_results/AtoB/Test_result_112.png)|
    |![](horse2zebra_test_results/AtoB/Test_result_115.png)|
    |Zebra to Horse<br>1st column: Input / 2nd column: Generated / 3rd column: Reconstructed|
    |![](horse2zebra_test_results/BtoA/Test_result_14.png)|
    |![](horse2zebra_test_results/BtoA/Test_result_31.png)|
    |![](horse2zebra_test_results/BtoA/Test_result_68.png)|
    |![](horse2zebra_test_results/BtoA/Test_result_111.png)|
    |![](horse2zebra_test_results/BtoA/Test_result_140.png)|

## apple2orange dataset
* Image size: 256x256
* Number of training images: 1,019 for apple images, 995 for orange images
* Number of test images: 266 for apple images, 248 for orange images

### Results
* Adam optimizer is used. Learning rate = 0.0002, batch size = 1, # of epochs = 200:
* 9 resnet blocks used for Generator.

GAN losses<br> ( ![FF8900](https://placehold.it/10/FF8900/000000?text=+) : Discriminator A / ![AE0000](https://placehold.it/10/AE0000/000000?text=+) : Discriminator B <br> ![CD52A7](https://placehold.it/10/CD52A7/000000?text=+) : Generator A / ![2156C9](https://placehold.it/10/2156C9/000000?text=+) : Generator B <br> ![58BEE4](https://placehold.it/10/58BEE4/000000?text=+) : Cycle loss A / ![319F92](https://placehold.it/10/319F92/000000?text=+) : Cycle loss B ) | Generated images<br>(Input / Generated / Reconstructed)
:---:|:---:
<img src = 'apple2orange_results/apple2orange_CycleGAN_losses_epochs_200.png'> | <img src = 'apple2orange_results/apple2orange_CycleGAN_epochs_200.gif'>

* Generated images using test data

    |Apple to Orange<br>1st column: Input / 2nd column: Generated / 3rd column: Reconstructed|
    |:---:|
    |![](apple2orange_test_results/AtoB/Test_result_16.png)|
    |![](apple2orange_test_results/AtoB/Test_result_99.png)|
    |![](apple2orange_test_results/AtoB/Test_result_110.png)|
    |![](apple2orange_test_results/AtoB/Test_result_176.png)|
    |![](apple2orange_test_results/AtoB/Test_result_250.png)|
    |Orange to Apple<br>1st column: Input / 2nd column: Generated / 3rd column: Reconstructed|
    |![](apple2orange_test_results/BtoA/Test_result_109.png)|
    |![](apple2orange_test_results/BtoA/Test_result_128.png)|
    |![](apple2orange_test_results/BtoA/Test_result_142.png)|
    |![](apple2orange_test_results/BtoA/Test_result_183.png)|
    |![](apple2orange_test_results/BtoA/Test_result_221.png)|

### References
1. https://github.com/mrzhu-cool/CycleGAN-pytorch
2. https://github.com/junyanz/pytorch-CycleGAN-and-CycleGAN
3. https://github.com/znxlwm/pytorch-CycleGAN
4. https://affinelayer.com/CycleGAN/
