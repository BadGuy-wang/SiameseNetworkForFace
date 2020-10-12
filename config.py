class args(object):
    # 定义一些超参
    batch_size = 32  # 训练时batch_size
    epochs = 50  # 训练的epoch
    training_dir = 'dataset/faces/training'
    testing_dir = 'dataset/faces/testing'
    ContrastiveLoss_margin = 2.0
    lr = 0.0005
