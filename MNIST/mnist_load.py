import torch

from mnist_function import CNN, predict_local_image


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 实例化一个空壳模型
    model = CNN().to(DEVICE)
    
    # 2. 加载刚才训练保存的权重，填满这个空壳
    model.load_state_dict(
        torch.load(
            'X:/coding/github/Alpha/MNIST/weights/mnist_cnn_epoch1_1.pth',
            map_location=DEVICE,
            weights_only=True,
        )
    )
    
    print("权重加载成功，开始识别...")
    # 3. 预测
    predict_local_image("mnist_myPhoto/0.png", model, DEVICE)