Data Augmentation Network

Examples of command line to train the model on borgy:
```shell
borgy submit --gpu=1 --cpu=4 --mem=20 -v /mnt:/mnt -i images.borgy.elementai.lan/saypraseuth.mounsaveng/v1 -w /mnt/projects/datgan/code --name limited_dataset_model_svhn --restartable -- /bin/bash -c "python3 main.py --cuda --pinnedmemory --dataset svhn --dataroot ../dataset/svhn --nc 3 --trainsetsize 1000 --valratio 0 --unlbldratio 0 --niter 1001 --batchSize 32 --lrDC 0.0005 --lrDD 0.0005 --lrG 0.0005 --lrC 0.0005 --fDC 1 --fGCl 0.0005 --fDD 1  --outDirPrefix ../resultsnew/ --outDirSuffix _run1  --classModel badGAN"

borgy submit --gpu=1 --cpu=4 --mem=20 -v /mnt:/mnt -i images.borgy.elementai.lan/saypraseuth.mounsaveng/v1 -w /mnt/projects/datgan/code --name limited_dataset_model_cifar10 --restartable -- /bin/bash -c "python3 main.py --cuda --pinnedmemory --dataset cifar10 --dataroot ../dataset/cifar-10 --nc 3 --trainsetsize 4000 --valratio 0 --unlbldratio 0 --niter 1001 --batchSize 32 --lrDC 0.0005 --lrDD 0.0005 --lrG 0.0005 --lrC 0.006 --fDC 0.1 --fGCl 0.001 --fDD 0.05  --outDirPrefix ../resultsnew/ --outDirSuffix _run1  --classModel badGAN"

borgy submit --gpu=1 --cpu=4 --mem=20 -v /mnt:/mnt -i images.borgy.elementai.lan/saypraseuth.mounsaveng/v1 -w /mnt/projects/datgan/code --name limited_dataset_model_mnist --restartable -- /bin/bash -c "python3 main.py --cuda --pinnedmemory --dataset mnist --dataroot ../dataset/mnist --nc 1 --trainsetsize 550 --valratio 0 --unlbldratio 0 --niter 1001 --batchSize 16 --lrDC 0.0005 --lrDD 0.0005 --lrG 0.0005 --lrC 0.006 --fDC 0.1 --fGCl 0.005 --fDD 0.05 --outDirPrefix ../resultsnew/ --outDirSuffix _run5  --classModel badGAN"

borgy submit --gpu=1 --cpu=4 --mem=20 -v /mnt:/mnt -i images.borgy.elementai.lan/saypraseuth.mounsaveng/v1 -w /mnt/projects/datgan/code --name limited_dataset_model_fmnist --restartable -- /bin/bash -c "python3 main.py --cuda --pinnedmemory --dataset fashionmnist --dataroot ../dataset/fashionmnist --nc 1 --trainsetsize 550 --valratio 0 --unlbldratio 0 --niter 1001 --batchSize 16 --lrDC 0.0005 --lrDD 0.0005 --lrG 0.0005 --lrC 0.0001 --fDC 0.1 --fGCl 0.005 --fDD 0.05 --outDirPrefix ../resultsnew/ --outDirSuffix _run5  --classModel badGAN"
```
