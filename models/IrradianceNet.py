from collections import OrderedDict
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn as nn
import logging
import torch



def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name,
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))

class CLSTM_cell(pl.LightningModule):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features, seq_len=8):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features)  # best for regression
        )

        self.seq_len = seq_len

    def forward(self, inputs=None, hidden_state=None):
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(self.seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W

            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)

def convlstm_encoder_params(in_chan=7, image_size=128):
    size_l1 = image_size
    size_l2 = image_size - (image_size // 4)
    size_l3 = image_size - (image_size // 2)
    size_l4 = size_l1 - size_l2

    convlstm_encoder_params = [
        [
            OrderedDict({'conv1_leaky_1': [in_chan, size_l4, 3, 1, 1]}),  # [1, 32, 3, 1, 1]
            OrderedDict({'conv2_leaky_1': [size_l3, size_l3, 3, 2, 1]}),
            OrderedDict({'conv3_leaky_1': [size_l2, size_l2, 3, 2, 1]}),
        ],
        [
            CLSTM_cell(shape=(size_l1,size_l1), input_channels=size_l4, filter_size=5, num_features=size_l3, seq_len=4),
            CLSTM_cell(shape=(size_l3,size_l3), input_channels=size_l3, filter_size=5, num_features=size_l2, seq_len=4),
            CLSTM_cell(shape=(size_l4,size_l4), input_channels=size_l2, filter_size=5, num_features=size_l1, seq_len=4)
        ]
    ]
    return convlstm_encoder_params

def convlstm_decoder_params(seq_len, image_size=128):
    size_l1 = image_size
    size_l2 = image_size - (image_size // 4)
    size_l3 = image_size - (image_size // 2)
    size_l4 = size_l1 - size_l2

    convlstm_decoder_params = [
        [
            OrderedDict({'deconv1_leaky_1': [size_l1, size_l1, 4, 2, 1]}),
            OrderedDict({'deconv2_leaky_1': [size_l2, size_l2, 4, 2, 1]}),
            OrderedDict({
                'conv3_leaky_1': [size_l3, size_l4, 3, 1, 1],
                'conv4_leaky_1': [size_l4, 1, 1, 1, 0]
            }),
        ],
        [
            CLSTM_cell(shape=(size_l4,size_l4), input_channels=size_l1, filter_size=5, num_features=size_l1, seq_len=seq_len),
            CLSTM_cell(shape=(size_l3,size_l3), input_channels=size_l1, filter_size=5, num_features=size_l2, seq_len=seq_len),
            CLSTM_cell(shape=(size_l1,size_l1), input_channels=size_l2, filter_size=5, num_features=size_l3, seq_len=seq_len),
        ]
    ]
    return convlstm_decoder_params

class Encoder(pl.LightningModule):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)



class Decoder(pl.LightningModule):
    def __init__(self, subnets, rnns, seq_len):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)
        self.seq_len = seq_len

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state) # , seq_len=8
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        return inputs



class IrradianceNet(pl.LightningModule):

    def __init__(self, seq_len, in_chan=1, image_size =128):
        super(IrradianceNet, self).__init__()

        self.future_seq = seq_len
        encoder_params = convlstm_encoder_params(in_chan, image_size)
        decoder_params = convlstm_decoder_params(seq_len, image_size)

        self.encoder = Encoder(encoder_params[0], encoder_params[1])
        self.decoder = Decoder(decoder_params[0], decoder_params[1], seq_len=seq_len)

        self.loss = torch.nn.L1Loss()

    def forward(self, input, future_seq=10):
        #input = input.permute(0, 1, 4, 2, 3)
        state = self.encoder(input)
        output = self.decoder(state)
        return output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss      = self.loss(outputs.squeeze(), labels.squeeze())    
        # Logging info
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        #if batch_idx <=10 :
        #    bsize = inputs.shape[0]
        
        #    fig, axes = plt.subplots(nrows=(bsize//2)*2, ncols=self.future_seq, figsize=(self.future_seq*4, (bsize//2)*2*4))
            
        #    for b in range(bsize//2):
        #        for t in range(self.future_seq):
        #            lbl = labels.cpu().detach().numpy()[b,0,t,...]
        #            pre = outputs.cpu().detach().numpy()[b,0,t,...]

        #            lbl = (lbl + 1)/2
        #            pre = (pre + 1)/2
                
        #            axes[2*b, t].imshow(lbl)
        #            axes[2*b + 1, t].imshow(pre)
        #            axes[2*b, t].axis(False)
        #            axes[2*b + 1, t].axis(False)
            
        #    plt.tight_layout()
        #    self.logger.experiment.add_figure(f'Train-Prediction-{batch_idx}', plt.gcf(), global_step=self.current_epoch)
        #    plt.close()
    
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss      = self.loss(outputs.squeeze(), labels.squeeze())    
        # Logging info
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)

        if batch_idx <=3 :
            bsize = inputs.shape[0]
        
            fig, axes = plt.subplots(nrows=(bsize//2)*2, ncols=self.future_seq, figsize=(self.future_seq*4, (bsize//2)*2*4))
            
            for b in range(bsize//2):
                for t in range(self.future_seq):
                    lbl = labels.cpu().detach().numpy()[b,0,t,...]
                    pre = outputs.cpu().detach().numpy()[b,t,0,...]

                    #lbl = (lbl + 1)/2
                    #pre = (pre + 1)/2
                
                    axes[2*b, t].imshow(lbl)
                    axes[2*b + 1, t].imshow(pre)
                    axes[2*b, t].axis(False)
                    axes[2*b + 1, t].axis(False)
            
            plt.tight_layout()
            self.logger.experiment.add_figure(f'Valid-Prediction-{batch_idx}', plt.gcf(), global_step=self.current_epoch)
            plt.close()
    
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)