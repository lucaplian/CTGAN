"""TVAE module."""

import numpy as np
import pandas as pd
import torch
import math
from opacus import PrivacyEngine

from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers._utils import _format_score, _set_device, validate_and_set_device
from ctgan.synthesizers.base import BaseSynthesizer, random_state

class EncoderNDecoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim, decompress_dims):
        super(EncoderNDecoder, self).__init__()
        dim_encoder = data_dim
        dim_decoder = embedding_dim
        seq_encoder = []
        seq_decoder = []
        for item in list(compress_dims):
            
            seq_encoder += [Linear(dim_encoder, item), ReLU()]
            dim_encoder = item

        self.seq_encoder = Sequential(*seq_encoder)
        self.fc1 = Linear(dim_encoder, embedding_dim)
        self.fc2 = Linear(dim_encoder, embedding_dim)

        dim_decoder = embedding_dim
        for item in list(decompress_dims):
            seq_decoder += [Linear(dim_decoder, item), ReLU()]
            dim_decoder = item

        seq_decoder.append(Linear(dim_decoder, data_dim))
        self.seq_decoder = Sequential(*seq_decoder)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)
    
    def forward(self, combined_input):
        """Encode the passed `input_`."""
        data_dim = self.seq_encoder[0].in_features
        input_ = combined_input[:, :data_dim]
        eps = combined_input[:, data_dim:]
        feature = self.seq_encoder(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        emb = eps * std + mu 
        rec_raw = self.seq_decoder(emb)
        rec = torch.clamp(rec_raw, min=-5.0, max=5.0)       
        return mu, std, logvar, self.seq_decoder(emb), self.sigma

class Encoder(Module):
    """Encoder for the TVAE.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq**2 / 2 / (std**2+1e-9)).sum())
                loss.append(torch.log(std+1e-9) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(
                    cross_entropy(
                        recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'
                    )
                )
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class TVAE(BaseSynthesizer):
    """TVAE."""

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        enable_gpu=True,
        verbose=False,
        cuda=None,
        delta = 1e-5,
        epsilon = math.inf, 
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        self.verbose = verbose
        self._device = validate_and_set_device(enable_gpu, cuda)
        self._enable_gpu = cuda if cuda is not None else enable_gpu
        self.epsilon = epsilon
        self.delta = delta

    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        self.encoder_n_decoder = EncoderNDecoder(data_dim, self.compress_dims, self.embedding_dim, self.decompress_dims).to(self._device)
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(self.encoder_n_decoder.parameters()),weight_decay=self.l2scale
        )

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = 'Loss: {loss}'
            iterator.set_description(iterator_description.format(loss=_format_score(0)))

        #DELTA = 1 / len(loader)
        max_grad_norm = 50.0
        '''if self.epsilon is math.inf or self.epsilon == math.info:
            max_grad_norm = 100.0
        '''
        privacy_engine = PrivacyEngine()
        self.encoder_n_decoder, optimizerAE, loader = privacy_engine.make_private_with_epsilon(
            module=self.encoder_n_decoder,
            optimizer=optimizerAE,
            data_loader=loader,
            target_delta=self.delta,
            target_epsilon=self.epsilon,
            epochs=self.epochs,
            loss_reduction = "mean",
            max_grad_norm=max_grad_norm,
        )


        for i in iterator:
            loss_values = []
            batch = []
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
               
                real = data[0].to(self._device)
                eps = torch.randn(real.shape[0], self.embedding_dim, device=self._device)
                
                real_combined = torch.cat((real, eps), -1)
                
                mu, std, logvar, rec, sigmas = self.encoder_n_decoder(real_combined)
                loss_1, loss_2 = _loss_function(
                    rec,
                    real,
                    sigmas,
                    mu,
                    logvar,
                    self.transformer.output_info_list,
                    self.loss_factor,
                )
                loss = loss_1 + loss_2
                loss.backward()
                raw_module = getattr(self.encoder_n_decoder, "_module", self.encoder_n_decoder)
                if raw_module.sigma.grad is not None:
                    raw_module.sigma.grad *= 0
                optimizerAE.step()
                raw_module.sigma.data.clamp_(0.01, 1.0)
                batch.append(id_)
                loss_values.append(loss.detach().cpu().item())
            '''print("raw_module.seq_encoder(real)=",raw_module.seq_encoder(real))
            print("real=", real)
            print("eps=", eps)
            print("logvar=", logvar)
            print("logvar=", rec)
            print("sigmas=", sigmas)'''
            epoch_loss_df = pd.DataFrame({
                'Epoch': [i] * len(batch),
                'Batch': batch,
                'Loss': loss_values,
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self.verbose:
                iterator.set_description(
                    iterator_description.format(loss=_format_score(loss.detach().cpu().item()))
                )

    @random_state
    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        raw_module = getattr(self.encoder_n_decoder, "_module", self.encoder_n_decoder)
        raw_module.eval()

        steps = samples // self.batch_size + 1
        data = []
        data_before = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            
            fake, sigmas = raw_module.seq_decoder(noise), raw_module.sigma

            data.append(fake.detach().cpu().numpy())
            data_before.append(fake.detach().cpu().numpy())
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())
        print("sigmas=",  sigmas)
        print("data_beforee=", data_before[-1])
        print("data=", data[-1])
        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU')."""
        enable_gpu = getattr(self, '_enable_gpu', True)
        self._device = _set_device(enable_gpu, device)
        self.decoder.to(self._device)
