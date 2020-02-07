import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss

# class CrossEntropyOneHot(object):
#     def __call__(self, sample):
#         _, labels = sample['Y'].max(dim=0)
#         # landmarks = landmarks.transpose((2, 0, 1))
#         return {'X_image': sample['X_image'],
#                 'Y': labels}


class CNN(nn.Module):
    def __init__(self, l2_reg):
        super(CNN, self).__init__()
        self.l2_reg = l2_reg
        self.conv1 = nn.Conv2d(
                in_channels=1,    # input height
                out_channels=32,  # n_filters
                kernel_size=5,    # filter size
                stride=1,         # filter movement/step
                padding=2,        # if want same width and length of
                                  # this image after Conv2d,
                                  # padding=(kernel_size-1)/2 if stride=1
            )

        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)

        self.layer1 = nn.Sequential(        # input shape (1, 28, 28)
            self.conv1,                     # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.layer2 = nn.Sequential(        # input shape (16, 14, 14)
            self.conv2,                     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.layer3 = nn.Sequential(        # input shape (16, 14, 14)
            self.conv3,                     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )

        self.fc1 = nn.Linear(128*8*8, 512)  # fully connected layer, output 10 classes
        self.fc2 = nn.Linear(512, 2)        # fully connected layer, output 10 classes

    # def penalty(self):
    #     return (self.l2_reg * (self.conv1.weight.norm(2)
    #                            + self.conv2.weight.norm(2)
    #                            + self.conv3.weight.norm(2)
    #                            + self.fc1.weight.norm(2)
    #                            + self.fc2.weight.norm(2))
    #             )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.fc1(x)
        output = self.fc2(x)
        # return output, x    # return x for visualization
        return F.softmax(output, dim=1)


class Autoencoder(nn.Module):
    def __init__(self, num_block):
        super(Autoencoder, self).__init__()
        self.num_block = num_block
        self.num_channel = 3
        self.skip = []

        filters = 44
        kernel_size = 3
        depth = 6

        for i in range(num_block):
            setattr(self, 'encoder{}'.format(i), nn.Sequential(
                nn.Conv2d(self.num_channel, filters * 2**i, kernel_size=kernel_size, padding=1),
                nn.ReLU(),
                nn.Conv2d(filters * 2**i, filters * 2**i, kernel_size=kernel_size, padding=1),
                nn.ReLU()))
            self.num_channel = filters * 2**i

        for i in range(depth):
            setattr(self, 'bottleneck{}'.format(i), nn.Sequential(
                nn.Conv2d(self.num_channel, self.num_channel, kernel_size=kernel_size, padding=1),  # , dilation=2**i),
                nn.ReLU(),
                ))

        for i in reversed(range(num_block)):
            setattr(self, 'decoder1{}'.format(i), nn.Sequential(
                nn.Conv2d(self.num_channel, filters * 2**i, kernel_size=kernel_size, padding=1),
                nn.ReLU(),
                ))
            setattr(self, 'decoder2{}'.format(i), nn.Sequential(
                nn.Conv2d(filters * 2**(i+1), filters * 2**i, kernel_size=kernel_size, padding=1),
                nn.ReLU()))
            self.num_channel = filters * 2**i
        self.out_layer = nn.Sequential(nn.Conv2d(self.num_channel, 1, kernel_size=1), nn.Sigmoid())

    def encoder(self, x, filters=44, num_block=3, kernel_size=3):
        self.num_channel = 3
        self.skip = []
        for i in range(num_block):
            self.num_channel = filters * 2**i
            x = eval("self.encoder{}(x)".format(i))
            self.skip.append(x)  # vérifier que la taille est bonne ?
            x = nn.MaxPool2d(2)(x)

        return x

    def bottleneck(self, x, filters=44, depth=6, kernel_size=3):
        for i in range(depth):
            x = eval("self.bottleneck{}(x)".format(i))
        return x

    def decoder(self, x, filters=44, num_block=3, kernel_size=3):
        for i in reversed(range(num_block)):
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
            x = eval("self.decoder1{}(x)".format(i))
            print(i, x.shape, self.skip[i].shape)
            x = torch.cat((self.skip[i], x), axis=1)  # vérifier que la taille est bonne ?
            print(i, x.shape)
            self.num_channel = filters * 2**i
            x = eval("self.decoder2{}(x)".format(i))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.out_layer(x)
        return x


def train(model, loader, f_loss, optimizer, device):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

    Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()

    N = 0
    tot_loss, correct = 0.0, 0.0
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # print("***",inputs.shape)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        print(outputs.shape)
        assert(False)

        loss = f_loss(outputs, targets)
        # print("Loss: ", loss)
        N += inputs.shape[0]
        tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

        # print("Output: ", outputs)
        predicted_targets = outputs

        correct += (predicted_targets == targets).sum().item()

        optimizer.zero_grad()
        # model.zero_grad()
        loss.backward()
        # model.penalty().backward()
        optimizer.step()
    return tot_loss/N, correct/N

def custom_loss(intput, target):
    m = nn.sigmoid
    loss = nn.BCELoss()
    output = loss(m(input), target)
    output.backward()
    return output

def test(model, loader, f_loss, device, final_test=False):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation

    Returns :

        A tuple with the mean loss and mean accuracy

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_loss, correct = 0.0, 0.0
        for i, (inputs, targets) in enumerate(loader):
            # pbar.update(1)
            # pbar.set_description("Testing step {}".format(i))
            # We got a minibatch from the loader within inputs and targets
            # With a mini batch size of 128, we have the following shapes
            #    inputs is of shape (128, 1, 28, 28)
            #    targets is of shape (128)

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)

            # We accumulate the exact number of processed samples
            N += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs
            correct += (predicted_targets == targets).sum().item()

            # if final_test:
            #     print("targets:\n", targets[0])
            #     print("predicted targets:\n", outputs[0])

    return tot_loss/N, correct/N


class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model to ", self.filepath)
            torch.save(self.model.state_dict(), self.filepath)
            # torch.save(self.model, self.filepath)
            self.min_loss = loss


def progress(loss, acc):
    print(' Training   : Loss : {:2.4f}, Acc : {:2.4f}\r'.format(loss, acc))
    sys.stdout.flush()
