from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch.nn as nn
import torch



import torch.nn as nn
import torch


class ConvLSTMCell(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    
class EncoderDecoderConvLSTM(pl.LightningModule):
    def __init__(self, nf, in_chan, future_seq):
        super(EncoderDecoderConvLSTM, self).__init__()
        self.future_seq = future_seq
        self.loss = torch.nn.L1Loss()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Tanh()(outputs) # This is fundamental, with Sigmoid wont work

        return outputs

    def forward(self, x, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t   = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, self.future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss      = self.loss(outputs, labels)    
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
        loss      = self.loss(outputs, labels)    
        # Logging info
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)

        if batch_idx <=3 :
            bsize = inputs.shape[0]
        
            fig, axes = plt.subplots(nrows=(bsize//2)*2, ncols=self.future_seq, figsize=(self.future_seq*4, (bsize//2)*2*4))
            
            for b in range(bsize//2):
                for t in range(self.future_seq):
                    lbl = labels.cpu().detach().numpy()[b,0,t,...]
                    pre = outputs.cpu().detach().numpy()[b,0,t,...]

                    lbl = (lbl + 1)/2
                    pre = (pre + 1)/2
                
                    axes[2*b, t].imshow(lbl)
                    axes[2*b + 1, t].imshow(pre)
                    axes[2*b, t].axis(False)
                    axes[2*b + 1, t].axis(False)
            
            plt.tight_layout()
            self.logger.experiment.add_figure(f'Valid-Prediction-{batch_idx}', plt.gcf(), global_step=self.current_epoch)
            plt.close()
    
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss      = self.loss(outputs, labels)    
        # Logging info
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)

    
        bsize = inputs.shape[0]
    
        fig, axes = plt.subplots(nrows=(bsize//2)*2, ncols=self.future_seq, figsize=(self.future_seq*4, (bsize//2)*2*4))
        
        for b in range(bsize//2):
            for t in range(self.future_seq):
                lbl = labels.cpu().detach().numpy()[b,0,t,...]
                pre = outputs.cpu().detach().numpy()[b,0,t,...]

                lbl = (lbl + 1)/2
                pre = (pre + 1)/2
            
                axes[2*b, t].imshow(lbl)
                axes[2*b + 1, t].imshow(pre)
                axes[2*b, t].axis(False)
                axes[2*b + 1, t].axis(False)
        
        plt.tight_layout()
        self.logger.experiment.add_figure(f'Test-Prediction-{batch_idx}', plt.gcf(), global_step=self.current_epoch)
        plt.close()
    
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


'''
class ConvLSTMCell(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              padding_mode='reflect',
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    
class EncoderDecoderConvLSTM(pl.LightningModule):
    def __init__(self, nf, in_chan, future_seq):
        super(EncoderDecoderConvLSTM, self).__init__()
        self.future_seq = future_seq
        self.loss       = ssim_mse_tv_loss

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Tanh()(outputs) # Sigmoid?

        return outputs

    def forward(self, x):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t   = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, self.future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss      = self.loss(outputs, labels)    
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
        loss      = self.loss(outputs, labels)    
        # Logging info
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)

        if batch_idx <=3 :
            bsize = inputs.shape[0]
        
            fig, axes = plt.subplots(nrows=(bsize//2)*2, ncols=self.future_seq, figsize=(self.future_seq*4, (bsize//2)*2*4))
            
            for b in range(bsize//2):
                for t in range(self.future_seq):
                    lbl = labels.cpu().detach().numpy()[b,0,t,...]
                    pre = outputs.cpu().detach().numpy()[b,0,t,...]

                    lbl = (lbl + 1)/2
                    pre = (pre + 1)/2
                
                    axes[2*b, t].imshow(lbl)
                    axes[2*b + 1, t].imshow(pre)
                    axes[2*b, t].axis(False)
                    axes[2*b + 1, t].axis(False)
            
            plt.tight_layout()
            self.logger.experiment.add_figure(f'Valid-Prediction-{batch_idx}', plt.gcf(), global_step=self.current_epoch)
            plt.close()
    
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss      = self.loss(outputs, labels)    
        # Logging info
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)

    
        bsize = inputs.shape[0]
    
        fig, axes = plt.subplots(nrows=(bsize//2)*2, ncols=self.future_seq, figsize=(self.future_seq*4, (bsize//2)*2*4))
        
        for b in range(bsize//2):
            for t in range(self.future_seq):
                lbl = labels.cpu().detach().numpy()[b,0,t,...]
                pre = outputs.cpu().detach().numpy()[b,0,t,...]

                lbl = (lbl + 1)/2
                pre = (pre + 1)/2
            
                axes[2*b, t].imshow(lbl)
                axes[2*b + 1, t].imshow(pre)
                axes[2*b, t].axis(False)
                axes[2*b + 1, t].axis(False)
        
        plt.tight_layout()
        self.logger.experiment.add_figure(f'Test-Prediction-{batch_idx}', plt.gcf(), global_step=self.current_epoch)
        plt.close()
    
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    
def ssim_mse_tv_loss(y_true, y_pred):
    device = y_true.device
    #ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    #ssim_loss = 1 - torch.mean(ssim(y_true, y_pred))
    mse_loss  = torch.nn.MSELoss()(y_pred, y_true)
    #mae_loss  = torch.nn.L1Loss()(y_pred, y_true)
    #tv_loss   = torch.mean(torch.sum(torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])) +
    #                    torch.sum(torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :])))
    
    return mse_loss.to(device) #+ ssim_loss.to(device) + 0.00001 * tv_loss.to(device)

'''