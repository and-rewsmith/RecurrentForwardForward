import math

from torch import cosine_similarity, nn, Tensor
import torch
from torch.optim import SGD
from typing import Callable, List, Self
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot  # type: ignore
from torch.nn import functional as F
import torch.nn.utils.parametrize as parametrize
import wandb

# data
NUM_SAMPLES = 1000
VOCAB_SIZE = 15
SEQUENCE_LEN = 40

# training
EPOCHS = 40
BATCH_SIZE = 50

TTT_BASE_INNER_LEARNING_RATE = 1e-4
TTT_INNER_LEARNING_RATE_LEARNING_RATE = 1e-4
TTT_OUTER_LEARNING_RATE = 1e-5

LOW_PASS_FILTER_DIM = 500
INPUT_DIM = 784
DROPOUT = 0.0
LEARNING_RATE_PARAMS_OUT_DIM_SCALE = 4


class VectorDataset(Dataset):
    def __init__(self, num_samples=1000, vector_dim=INPUT_DIM):
        self.num_samples = num_samples
        self.vector_dim = vector_dim
        self.data = self._generate_data()

    def _generate_data(self):
        return torch.randn(self.num_samples, self.vector_dim)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        vector = self.data[idx]
        return vector, vector  # Return the same vector as both input and target


class TTTInner(nn.Module):
    def __init__(self: Self, filter_dim: int,
                 get_theta_k: Callable[[], torch.nn.Parameter],
                 get_theta_q: Callable[[], torch.nn.Parameter],
                 get_theta_v: Callable[[], torch.nn.Parameter],
                 get_inner_learning_rate: Callable[[Tensor], Tensor]) -> None:
        super(TTTInner, self).__init__()

        self.filter_dim = filter_dim
        self.w = nn.Linear(filter_dim, filter_dim)
        torch.nn.init.kaiming_uniform_(self.w.weight)

        self.get_theta_k = get_theta_k
        self.get_theta_q = get_theta_q
        self.get_theta_v = get_theta_v
        self.get_inner_learning_rate = get_inner_learning_rate

    def online_inference(self: Self, src: torch.Tensor) -> torch.Tensor:
        # compute views
        train_view = src @ self.get_theta_k()  # type: ignore[operator]
        label_view = src @ self.get_theta_v()  # type: ignore[operator]
        test_view = src @ self.get_theta_q()  # type: ignore[operator]

        # reconstruction loss
        reconstruction_target = label_view - train_view  # type: ignore[operator]
        w_train_view = self.w(train_view)
        loss = nn.MSELoss()(w_train_view, reconstruction_target)
        self.inner_loss = loss
        # wandb.log({"inner_loss": loss})

        # compute gradients for `w` and manually update
        gradients = grad(loss, list(self.w.parameters()), create_graph=True)
        assert gradients[0].shape == self.w.weight.shape

        # wandb.log({"w_grad": gradients[0].norm()})
        # wandb.log({"w_bias_grad": gradients[1].norm()})

        # calculate the learned inner learning rate for each parameter and shape appropriately
        inner_learning_rate, inner_learning_rate_bias = self.get_inner_learning_rate(src)

        # TODO: consider adding layer norm here to stabilize batch effects of averaging

        # wandb.log({"inner_learning_rate": inner_learning_rate.norm()})
        # wandb.log({"inner_learning_rate_bias": inner_learning_rate_bias.norm()})
        # wandb.log( #     {"inner_learning_rate_specific_index": inner_learning_rate[0][0]})

        updated_weight = self.w.weight - inner_learning_rate * gradients[0]
        updated_bias = self.w.bias - inner_learning_rate_bias * gradients[1]

        # calculate output using updated `w_bar`
        z = torch.nn.functional.linear(test_view, updated_weight, updated_bias) + test_view

        # this was intended to stop grads from flowing back to weights (minor optimization)
        # doesn't seem to work though
        # leaving it
        self.w.weight.requires_grad_(False)
        self.w.bias.requires_grad_(False)

        # update `w` with `w_bar` which resets computation graph
        with torch.no_grad():
            self.w.weight = nn.Parameter(updated_weight, requires_grad=True)
            self.w.bias = nn.Parameter(updated_bias, requires_grad=True)

        return z


class TTTHead(nn.Module):
    def __init__(self: Self, input_dim: int, output_dim: int, filter_dim: int,
                 ttt_base_inner_learning_rate: float) -> None:
        super(TTTHead, self).__init__()

        self.ttt_base_inner_learning_rate = ttt_base_inner_learning_rate

        self.theta_k = nn.Parameter(torch.randn(input_dim, filter_dim))
        self.theta_v = nn.Parameter(torch.randn(input_dim, filter_dim))
        self.theta_q = nn.Parameter(torch.randn(input_dim, filter_dim))
        self.theta_o = nn.Parameter(torch.randn(filter_dim, output_dim))

        learning_rate_params_out_dim = filter_dim * 2
        self.inner_learning_rate_params = nn.Linear(input_dim, learning_rate_params_out_dim)

        self.inner = TTTInner(filter_dim=filter_dim,
                              get_theta_k=self.get_theta_k,
                              get_theta_q=self.get_theta_q, get_theta_v=self.get_theta_v,
                              get_inner_learning_rate=self.get_inner_learning_rate)

        torch.nn.init.kaiming_uniform_(self.theta_k)
        torch.nn.init.kaiming_uniform_(self.theta_v)
        torch.nn.init.kaiming_uniform_(self.theta_q)
        torch.nn.init.kaiming_uniform_(self.theta_o)
        torch.nn.init.kaiming_uniform_(self.inner_learning_rate_params.weight)

        self.input_dim = input_dim
        self.low_pass_filter_dim = filter_dim

    def train_head(self: Self, input: torch.Tensor) -> torch.Tensor:
        outputs = self.inner.online_inference(input)

        outputs: Tensor = outputs @ self.theta_o  # type: ignore
        return outputs

    def get_theta_k(self: Self) -> torch.nn.Parameter:
        return self.theta_k

    def get_theta_q(self: Self) -> torch.nn.Parameter:
        return self.theta_q

    def get_theta_v(self: Self) -> torch.nn.Parameter:
        return self.theta_v

    def get_inner_learning_rate(self: Self, input: torch.Tensor) -> Tensor:
        sigmoid_op = self.inner_learning_rate_params(input)
        # wandb.log({"pre_sigmoid": pre_sigmoid.mean()})
        sigmoid_op = self.ttt_base_inner_learning_rate * F.sigmoid(sigmoid_op)
        sigmoid_op = sigmoid_op.mean(dim=0)

        # repeat along dimension 0, which corresponds to the "to" dimension of nn.Linear weight
        # implication: all presynamptic neurons have the same learning rate
        learning_rate_matrix_params = sigmoid_op[0:self.low_pass_filter_dim].repeat(self.low_pass_filter_dim, 1)
        learning_rate_bias = sigmoid_op[self.low_pass_filter_dim:]

        return learning_rate_matrix_params, learning_rate_bias


class TTTBlock(nn.Module):

    # TODO: add residual connection
    # TODO: add layer norm at beginning and end of block
    # TODO: recursive structure
    def __init__(self: Self, filter_dim: int, embedding_dim: int, output_dim: int,
                 ttt_base_inner_learning_rate: float) -> None:
        super(TTTBlock, self).__init__()

        self.ttt_head = TTTHead(filter_dim=filter_dim, input_dim=embedding_dim, output_dim=output_dim,
                                ttt_base_inner_learning_rate=ttt_base_inner_learning_rate)

    def train_block(self, input: torch.Tensor) -> torch.Tensor:
        outputs = self.ttt_head.train_head(input)
        return outputs


class TTTModel(nn.Module):

    def __init__(
            self: Self, filter_dim: int, embedding_dim: int, output_dim: int,
            ttt_base_inner_learning_rate: float,
            num_layers: int) -> None:
        super(TTTModel, self).__init__()

        self.ttt_blocks = nn.ModuleList([
            TTTBlock(
                filter_dim=filter_dim, embedding_dim=embedding_dim, output_dim=output_dim,
                ttt_base_inner_learning_rate=ttt_base_inner_learning_rate)
            for _ in range(num_layers)])

    def forward(self: Self, src: torch.Tensor) -> torch.Tensor:
        output = src
        for block in self.ttt_blocks:
            output = block.train_block(output)

        return output

    def get_params_inner_learning_rates(self: Self) -> List[torch.nn.Parameter]:
        params = []
        for block in self.ttt_blocks:
            params.extend(block.ttt_head.inner_learning_rate_params.parameters())
        return params

    def get_params_ttt_heads(self: Self) -> List[torch.nn.Parameter]:
        params = []
        for block in self.ttt_blocks:
            params.extend([block.ttt_head.theta_k, block.ttt_head.theta_q,
                           block.ttt_head.theta_v, block.ttt_head.theta_o])
        return params


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    # torch.manual_seed(1234)

    # wandb.init(
    #     project="ttt-FF",
    #     config={
    #         "architecture": "Recurrent-FF",
    #         "dataset": "SequentialNumbers",
    #     }
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    model = TTTModel(
        num_layers=1, filter_dim=LOW_PASS_FILTER_DIM, embedding_dim=INPUT_DIM, output_dim=INPUT_DIM,
        ttt_base_inner_learning_rate=TTT_BASE_INNER_LEARNING_RATE)
    model = model.to(device)

    dataset = VectorDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # train
    model.train()

    optim = SGD(model.get_params_ttt_heads(), lr=TTT_OUTER_LEARNING_RATE)
    optim_inner_lr = SGD(model.get_params_inner_learning_rates(), lr=TTT_INNER_LEARNING_RATE_LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(0, EPOCHS):
        for i, (data, labels) in enumerate(dataloader):
            print(f"Epoch: {epoch}, Batch: {i} / {len(dataloader)}")

            optim.zero_grad()
            optim_inner_lr.zero_grad()

            data = data.to(device)
            labels = labels.to(device)

            output = model.forward(data)
            loss = criterion(output, labels)
            print(loss)

            # cosine similarity of output and labels first batch
            output_cossim = output[0]
            labels_cossim = labels[0]

            # use a library
            cos_sim = F.cosine_similarity(output_cossim, labels_cossim, dim=0)
            # wandb.log({"cos_sim": cos_sim.item()})
            # wandb.log({"outer_loss": loss.item()})

            loss.backward()

            optim.step()
            optim_inner_lr.step()
