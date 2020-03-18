import torch
import torch.nn as nn
import torch.nn.modules.loss
import sys


class AutoEncoder(nn.Module):

    def __init__(self, num_block, depth, droput = False, batch_norm = False, conv_transpose = False):
        super().__init__()
        self.num_block = num_block
        self.num_channel = 3
        self.skip = []

        self.filters = 44
        self.kernel_size = 3
        self.depth = depth
        self.dropout_rate = 0.2
        
        list_activate = [nn.ReLU()]
        if batch_norm:
            list_activate.insert(0,nn.BatchNorm2d(self.filters * 2**i))
        if droput:
            list_activate.append(nn.Dropout(p=self.dropout_rate))
        relu = nn.Sequential(*list_activate)
            

        for i in range(num_block):
            setattr(self, 'encoder{}'.format(i), nn.Sequential(
                nn.Conv2d(self.num_channel,
                          self.filters * 2**i,
                          kernel_size=self.kernel_size,
                          padding=1),
                # nn.BatchNorm2d(self.filters * 2**i),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                nn.Conv2d(self.filters * 2**i,
                          self.filters * 2**i,
                          kernel_size=self.kernel_size,
                          padding=1),
                # nn.BatchNorm2d(self.filters * 2**i),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate)
            ))
            self.num_channel = self.filters * 2**i

        for i in range(self.depth):
            setattr(self, 'bottleneck{}'.format(i), nn.Sequential(
                nn.Conv2d(self.num_channel,
                          self.num_channel,
                          kernel_size=self.kernel_size,
                          padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(self.num_channel)
            ))

        for i in reversed(range(num_block)):
            setattr(self, 'decoder1{}'.format(i), nn.Sequential(
                nn.Conv2d(self.num_channel,
                          self.filters * 2**i,
                          kernel_size=self.kernel_size,
                          padding=1),
                # nn.BatchNorm2d(self.filters * 2**i),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate)
            ))
            setattr(self, 'decoder2{}'.format(i), nn.Sequential(
                nn.Conv2d(self.filters * 2**(i+1),
                          self.filters * 2**i,
                          kernel_size=self.kernel_size,
                          padding=1),
                # nn.BatchNorm2d(self.filters * 2**i),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate)
            ))
            self.num_channel = self.filters * 2**i
        self.out_layer = nn.Sequential(
            nn.Conv2d(self.num_channel,
                      1,
                      kernel_size=1),
            nn.Sigmoid())
        # self.out_layer = nn.Sequential(
        #     nn.Conv2d(self.num_channel,
        #               1,
        #               kernel_size=1)
        #     )

    def encoder(self, x):
        self.num_channel = 3
        self.skip = []
        for i in range(self.num_block):
            self.num_channel = self.filters * 2**i
            x = eval("self.encoder{}(x)".format(i))
            self.skip.append(x)  # vérifier que la taille est bonne ?
            x = nn.MaxPool2d(2)(x)
        # print(sys.getsizeof(self.skip))

        return x

    def bottleneck(self, x):
        for i in range(self.depth):
            x = eval("self.bottleneck{}(x)".format(i))
        return x

    def decoder(self, x):
        for i in reversed(range(self.num_block)):
            x = nn.Upsample(scale_factor=2, mode='nearest')(x)
            # x = nn.ConvTranspose2d(in_channels=self.num_channel, out_channels=self.num_channel, stride=2, kernel_size=3, padding=1, output_padding=1)(x)
            x = eval("self.decoder1{}(x)".format(i))
            # vérifier que la taille est bonne ?
            x = torch.cat((self.skip[i], x), axis=1)
            self.num_channel = self.filters * 2**i
            x = eval("self.decoder2{}(x)".format(i))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.out_layer(x)
        return x


def train(model, loader, f_loss, optimizer, device, log_manager=None):
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

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # if i == 0:
        #     # if final_test:
        #     print("sending image")
        #     log_manager.tensorboard_send_image(
        #         i, inputs[0], targets[0], outputs[0], txt= "trainning")
        # print("Loss: ", loss)
        N += inputs.shape[0]
        tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

        # print("Output: ", outputs)
        # predicted_targets = outputs
        # correct += (predicted_targets == targets).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return tot_loss/N, correct/N


def test(model, loader, f_loss, device, log_manager=None, final_test=False, txt = "testing"):
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

            # send image to tensor board
            # if i == 0 and final_test:
            if final_test:
                print("sending image")
                log_manager.tensorboard_send_image(
                    i, inputs[0], targets[0], outputs[0], txt = txt)

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
            correct += (outputs == targets).sum().item()

            # if final_test:
            #     print("targets:\n", targets[0])
            #     print("predicted targets:\n", outputs[0])
    return tot_loss/N, correct/N
