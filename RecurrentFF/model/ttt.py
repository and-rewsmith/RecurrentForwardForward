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

EPOCHS = 60
BATCH_SIZE = 50

TTT_BASE_INNER_LEARNING_RATE = 1e-1
TTT_INNER_LEARNING_RATE_LEARNING_RATE = 1e-3
TTT_OUTER_LEARNING_RATE = 1e-5

LOW_PASS_FILTER_DIM = 308
INPUT_DIM = 6
DROPOUT = 0.0

CLIP_VALUE = 10.0

class VectorDataset(Dataset):
    def __init__(self, num_samples=1000, vector_dim=5, degree=2):
        self.num_samples = num_samples
        self.vector_dim = vector_dim
        self.degree = degree  # Degree of the polynomial
        self.data = self._generate_data()

    def _generate_data(self):
        # Generate random data with normal distribution
        return torch.randn(self.num_samples, self.vector_dim)

    def _polynomial_transform(self, vector):
        # Apply polynomial transformation (e.g., squaring)
        return torch.pow(vector, self.degree)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        vector = self.data[idx]
        transformed_vector = self._polynomial_transform(vector)
        return vector, transformed_vector  # Return input and the polynomial-transformed output


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
        assert w_train_view.shape == reconstruction_target.shape
        loss = nn.MSELoss()(w_train_view, reconstruction_target)
        self.inner_loss = loss
        # wandb.log({"inner_loss": loss})

        # compute gradients for `w` and manually update
        gradients = grad(loss, list(self.w.parameters()), create_graph=True)
        assert gradients[0].shape == self.w.weight.shape

        clipped_gradients = []
        for g in gradients:
            # g.clamp(-CLIP_VALUE, CLIP_VALUE)
            clipped_gradients.append(g)

        # wandb.log({"w_grad": gradients[0].norm()})
        # wandb.log({"w_bias_grad": gradients[1].norm()})

        # calculate the learned inner learning rate for each parameter and shape appropriately
        inner_learning_rate, inner_learning_rate_bias = self.get_inner_learning_rate(src)
        assert inner_learning_rate.shape == gradients[0].shape
        assert inner_learning_rate_bias.shape == gradients[1].shape
        assert inner_learning_rate.shape == clipped_gradients[0].shape
        assert inner_learning_rate_bias.shape == clipped_gradients[1].shape

        # TODO: consider adding layer norm here to stabilize batch effects of averaging

        # wandb.log({"inner_learning_rate": inner_learning_rate.norm()})
        # wandb.log({"inner_learning_rate_bias": inner_learning_rate_bias.norm()})
        # wandb.log({"inner_learning_rate_specific_index": inner_learning_rate[0][0]})

        updated_weight = self.w.weight - inner_learning_rate * clipped_gradients[0]
        updated_bias = self.w.bias - inner_learning_rate_bias * clipped_gradients[1]

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
    def __init__(self: Self, embedding_dim: int, output_dim: int, filter_dim: int,
                 ttt_base_inner_learning_rate: float) -> None:
        super(TTTHead, self).__init__()

        self.ttt_base_inner_learning_rate = ttt_base_inner_learning_rate

        self.theta_k = nn.Parameter(torch.randn(embedding_dim, filter_dim))
        self.theta_v = nn.Parameter(torch.randn(embedding_dim, filter_dim))
        self.theta_q = nn.Parameter(torch.randn(embedding_dim, filter_dim))
        self.theta_o = nn.Parameter(torch.randn(filter_dim, output_dim))

        learning_rate_params_out_dim = filter_dim * 2
        self.inner_learning_rate_params = nn.Linear(embedding_dim, learning_rate_params_out_dim)

        self.inner = TTTInner(filter_dim=filter_dim,
                              get_theta_k=self.get_theta_k,
                              get_theta_q=self.get_theta_q, get_theta_v=self.get_theta_v,
                              get_inner_learning_rate=self.get_inner_learning_rate)

        torch.nn.init.kaiming_uniform_(self.theta_k)
        torch.nn.init.kaiming_uniform_(self.theta_v)
        torch.nn.init.kaiming_uniform_(self.theta_q)
        torch.nn.init.kaiming_uniform_(self.theta_o)
        torch.nn.init.kaiming_uniform_(self.inner_learning_rate_params.weight)

        self.input_dim = embedding_dim
        self.low_pass_filter_dim = filter_dim

    def train_head(self: Self, input: torch.Tensor) -> torch.Tensor:
        outputs = self.inner.online_inference(input)
        # TODO: maybe remove?
        # outputs = F.leaky_relu(outputs)
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
        sigmoid_op = self.ttt_base_inner_learning_rate * F.leaky_relu(sigmoid_op)
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
    def __init__(self: Self, filter_dim: int, embedding_dim: int, output_dim,
                 ttt_base_inner_learning_rate: float, num_heads: int) -> None:
        super(TTTBlock, self).__init__()
        
        self.num_heads = num_heads
        self.head_embedding_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim

        assert self.head_embedding_dim * num_heads == embedding_dim, "embedding_dim must be divisible by num_heads"

        self.ttt_heads = nn.ModuleList([
            TTTHead(filter_dim=filter_dim, embedding_dim=self.head_embedding_dim, output_dim=self.head_embedding_dim,
                    ttt_base_inner_learning_rate=ttt_base_inner_learning_rate)
            for _ in range(num_heads)
        ])

        # TODO: initialization choice needed?
        self.output_linear = nn.Linear(embedding_dim, output_dim)


    def train_block(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, _ = input.shape

        # Split the input for each head
        split_input = input.view(batch_size, self.num_heads, self.head_embedding_dim)

        # Process each split through its corresponding head
        outputs = []
        for i, head in enumerate(self.ttt_heads):
            head_output = head.train_head(split_input[:, i])
            outputs.append(head_output)

        # Concatenate the outputs from all heads
        concat_output = torch.cat(outputs, dim=-1)
        assert concat_output.shape == (batch_size, self.embedding_dim)
        # concat_output = F.leaky_relu(concat_output)

        # Apply the output linear layer
        final_output = self.output_linear(concat_output)

        return final_output


class TTTModel(nn.Module):

    def __init__(
            self: Self, filter_dim: int, embedding_dim: int, output_dim: int,
            ttt_base_inner_learning_rate: float,
            num_layers: int, num_heads: int) -> None:
        super(TTTModel, self).__init__()

        self.ttt_blocks = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers-1:
                block = TTTBlock(num_heads=num_heads, filter_dim=filter_dim, embedding_dim=embedding_dim, output_dim=embedding_dim,
                    ttt_base_inner_learning_rate=ttt_base_inner_learning_rate)
            else:
                block = TTTBlock(num_heads=num_heads, filter_dim=filter_dim, embedding_dim=embedding_dim, output_dim=output_dim,
                    ttt_base_inner_learning_rate=ttt_base_inner_learning_rate)
            self.ttt_blocks.append(block)

    def forward(self: Self, src: torch.Tensor) -> torch.Tensor:
        output = src
        for block in self.ttt_blocks:
            output = block.train_block(output)

        return output

    def get_params_inner_learning_rate(self: Self) -> List[torch.nn.Parameter]:
        params = []
        for block in self.ttt_blocks:
            for head in block.ttt_heads:
                params.extend(head.inner_learning_rate_params.parameters())
        return params

    def get_params_outer_loop_non_lr(self: Self) -> List[torch.nn.Parameter]:
        params = []
        for block in self.ttt_blocks:
            for head in block.ttt_heads:
                params.extend([head.theta_k, head.theta_q, head.theta_v, head.theta_o])
            params.extend(block.output_linear.parameters())
        return params


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    wandb.init(
        project="ttt-FF",
        config={
            "architecture": "Recurrent-FF",
            "dataset": "SequentialNumbers",
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    model = TTTModel(
        num_layers=3, num_heads=3, filter_dim=LOW_PASS_FILTER_DIM, embedding_dim=INPUT_DIM, output_dim=INPUT_DIM,
        ttt_base_inner_learning_rate=TTT_BASE_INNER_LEARNING_RATE)
    model = model.to(device)

    dataset = VectorDataset(vector_dim=INPUT_DIM)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # train
    model.train()

    optim = SGD(model.get_params_outer_loop_non_lr(), lr=TTT_OUTER_LEARNING_RATE)
    optim_inner_lr = SGD(model.get_params_inner_learning_rate(), lr=TTT_INNER_LEARNING_RATE_LEARNING_RATE)
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

            wandb.log({"outer_loss": loss.item()})

            loss.backward()

            # # Clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)

            optim.step()
            optim_inner_lr.step()
