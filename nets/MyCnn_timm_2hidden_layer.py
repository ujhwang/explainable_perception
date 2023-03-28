import torchvision.models as models
import torch.nn as nn
import torch

class MyCnn(nn.Module):

    def __init__(self, model, finetune=False):
        super(MyCnn, self).__init__()
        self.cnn = model
        if not finetune:
            for param in self.cnn.parameters():  # freeze cnn params
                param.requires_grad = False
        x = torch.randn([3, 224, 224]).unsqueeze(0)
        output_size = self.cnn.forward_features(x).size()
        self.dims = output_size[1] * 2
        self.cnn_size = output_size
        self.rank_fc_1 = nn.Linear(self.cnn_size[1] * self.cnn_size[2] * self.cnn_size[3], 4096)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.hidden1 = nn.Linear(4096, 2048)
        self.hidden2 = nn.Linear(2048, 1024)
        self.rank_fc_out = nn.Linear(1024, 1)

    def forward(self, left_batch, right_batch=None):
        if right_batch is None:
            return self.single_forward(left_batch)['output'].unsqueeze(0).unsqueeze(0)

        else:
            return {
                'left': self.single_forward(left_batch),
                'right': self.single_forward(right_batch)
            }

    def single_forward(self, batch):
        batch_size = batch.size()[0]
        x = self.cnn.forward_features(batch)

        x = x.reshape(batch_size, self.cnn_size[1] * self.cnn_size[2] * self.cnn_size[3])
        x = self.rank_fc_1(x)
        x = self.relu(x)
        x = self.drop(x)
        
        # hidden 1
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.drop(x)
        # hidden 2
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.rank_fc_out(x)

        return {
            'output': x
        }


if __name__ == '__main__':
    net = MyCnn(models.resnet50)
    x = torch.randn([3, 224, 224]).unsqueeze(0)
    y = torch.randn([3, 224, 224]).unsqueeze(0)
    fwd = net(x, y)
    print(fwd)
