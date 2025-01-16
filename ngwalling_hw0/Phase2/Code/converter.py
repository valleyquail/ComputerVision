import torch
from Network.Network import CIFAR10Model_Basic_Linear, CIFAR10Model_BN_CNN, CIFAR10Model_ResNet, CIFAR10Model_ResNeXt, \
    CIFAR10Model_DenseNet


def convert(ModelName, CheckpointPath, ExportPath):
    match ModelName:
        case 'Basic_Linear':
            model = CIFAR10Model_Basic_Linear(InputSize=3 * 32 * 32, OutputSize=10)
        case 'BN_CNN':
            model = CIFAR10Model_BN_CNN(InputSize=3 * 32 * 32, OutputSize=10)
        case 'ResNet':
            model = CIFAR10Model_ResNet(InChannels=3, OutputSize=10)
        case 'ResNeXt':
            model = CIFAR10Model_ResNeXt(32, OutputSize=10)
        case 'DenseNet':
            GrowthFactor = 24
            model = CIFAR10Model_DenseNet(InChannels=3, OutputSize=10, GrowthFactor=GrowthFactor)
        case _:
            print('Model not found')
    CheckpointPath +=  ModelName + '/' + 'best_' + ModelName + '_model.ckpt'

    checkpoint = torch.load(CheckpointPath, map_location=torch.device('cpu'))  # Load to CPU
    model.load_state_dict(checkpoint['model_state_dict'])  # Load state_dict

    # Set the model to evaluation mode
    model.eval()

    # Export the model to a .pt file
    export_path = f"{ModelName}.pt"
    torch.save(model, ExportPath)

    print(f"Model exported successfully to {ExportPath}")

if __name__ == '__main__':
    convert('Basic_Linear', './../Checkpoints/', 'Models/Basic_Linear.pt')
    convert('BN_CNN', './../Checkpoints/', 'Models/BN_CNN.pt')
    convert('ResNet', './../Checkpoints/', 'Models/ResNet.pt')
    convert('ResNeXt', './../Checkpoints/', 'Models/ResNeXt.pt')
    convert('DenseNet', './../Checkpoints/', 'Models/DenseNet.pt')
