import torch
import torch.nn as nn
from tqdm import tqdm


class D3PM:

    def __init__(
        self,
        x0_model: nn.Module,
        device,
        n_steps: int,
        num_classes: int = 8,
        forward_type: str = "absorbing",
        hybrid_loss_coef=0.001,
    ):
        self.device = device
        self.x0_model = x0_model.to(device)
        self.n_steps = n_steps
        self.hybrid_loss_coef = hybrid_loss_coef

        steps = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )

        # self.beta_t = [1 / (self.n_T - t + 1) for t in range(1, self.n_T + 1)]
        self.eps = 1e-6
        self.num_classes = num_classes
        q_onestep_mats = []

        for beta in self.beta_t:
            if forward_type == "uniform":
                # each class has equal probability of being the next class
                mat = torch.ones(num_classes, num_classes) * beta / num_classes
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
                q_onestep_mats.append(mat)
            elif forward_type == "absorbing":
                # class 0 is absorbing
                mat = torch.zeros(num_classes, num_classes)
                mat.diagonal().fill_(1 - beta)
                mat[:, 0] = beta
                mat[0, 0] = 1.0
                q_onestep_mats.append(mat)
            else:
                raise NotImplementedError(f'Forward type {forward_type} is not implemented.')
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

        q_one_step_transposed = q_one_step_mats.transpose(
            1, 2
        )  # this will be used for q_posterior_logits

        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_steps):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.q_one_step_transposed = q_one_step_transposed.to(device)
        self.q_mats = q_mats.to(device)

        assert self.q_mats.shape == (
            self.n_steps,
            num_classes,
            num_classes,
        ), f"q_mats shape is not correct, got {self.q_mats.shape}"

    @staticmethod
    def _at(q: torch.Tensor, t: torch.Tensor, x: torch.Tensor):
        """
        Query q_mats or q_one_step_transposed with t (time step) and x (class index).

        Args:
            q: q_mats or q_one_step_transposed, shape (n_steps, num_classes, num_classes)
            t: shape (bs, ) of time steps
            x: shape (bs, C, H, W) of class indices

        Returns:
            Tensor of shape (bs, C, H, W, num_classes)
        """
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        return q[t - 1, x, :]

    def _q_posterior_logits(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        # if t == 1, this means we return the L_0 loss, so directly try to x_0 logits.
        # otherwise, we return the L_{t-1} loss.
        # Also, we never have t == 0.

        # if x_0 is integer, we convert it to one-hot.
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classes) + self.eps
            )
        else:
            x_0_logits = x_0.clone()

        assert x_0_logits.shape == x_t.shape + (self.num_classes,), \
            f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}"

        # Here, we calculate equation (3) of the paper.
        # Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.

        # fact1 is "guess of x_{t-1}" from x_t
        # fact2 is "guess of x_{t-1}" from x_0

        fact1 = self._at(self.q_one_step_transposed, t, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        # bs, num_classes, num_classes
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))

        bc = torch.where(t_broadcast == 1, x_0_logits, out)

        return bc

    def _vb(self, dist1: torch.Tensor, dist2: torch.Tensor):
        """Sum of KL divergence."""
        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1)
            - torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1).mean()

    def _sample_forward(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        # forward process, x_0 is the clean input.
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def _model_predict(self, x_0: torch.Tensor, t: torch.Tensor):
        # this part exists because in general, manipulation of logits from model's logit
        # so they are in form of x_0's logit might be independent to model choice.
        # for example, you can convert 2 * N channel output of model output to logit via get_logits_from_logistic_pars
        # they introduce at appendix A.8.

        predicted_x0_logits = self.x0_model(x_0, t)
        # shape (bs, C, H, W, num_classes), representing log probabilities of each class (logits)

        return predicted_x0_logits

    def calc_loss(self, x_0: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Makes forward diffusion x_t from x_0, and tries to guess x_0 value from x_t using x0_model.
        x_0 is one-hot of dim (bs, ...), with int values of 0 to num_classes - 1
        """
        x_0 = x_0.to(self.device)
        t = torch.randint(1, self.n_steps, (x_0.shape[0],), device=self.device)
        x_t = self._sample_forward(
            x_0, t, torch.rand((*x_0.shape, self.num_classes), device=self.device)
        )
        # x_t is same shape as x
        assert x_t.shape == x_0.shape, f"x_t.shape: {x_t.shape}, x.shape: {x_0.shape}"

        predicted_x0_logits = self._model_predict(x_t, t)

        # variational bound loss
        true_q_posterior_logits = self._q_posterior_logits(x_0, x_t, t)
        pred_q_posterior_logits = self._q_posterior_logits(predicted_x0_logits, x_t, t)
        vb_loss = self._vb(true_q_posterior_logits, pred_q_posterior_logits)

        # cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x_0 = x_0.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x_0)

        return self.hybrid_loss_coef * vb_loss + ce_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def _sample_backward_step(self, x_t: torch.Tensor, t: int, noise: torch.Tensor):
        t = torch.tensor([t] * x_t.shape[0], device=self.device)
        predicted_x0_logits = self._model_predict(x_t, t)
        pred_q_posterior_logits = self._q_posterior_logits(predicted_x0_logits, x_t, t)

        noise = torch.clip(noise, self.eps, 1.0)

        not_first_step = (t != 1).float().reshape((x_t.shape[0], *[1] * (x_t.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample

    def infer(self, n_sample: int, size: tuple[int, int], channels: int):
        """Generates a batch of images."""
        img_shape = (n_sample, channels, *size)
        x = torch.randint(0, self.num_classes, img_shape).to(self.device)
        with torch.no_grad():
            for t in tqdm(range(self.n_steps, 0, -1), desc='Diffusion sampling'):
                x = self._sample_backward_step(x, t, torch.rand((*x.shape, self.num_classes), device=self.device))
        return x

    def infer_with_image_sequence(self, n_sample: int, size: tuple[int, int], channels: int, stride: int):
        """Generates a sequence of batches during the sampling process."""
        img_shape = (n_sample, channels, *size)
        x = torch.randint(0, self.num_classes, img_shape).to(self.device)

        steps = 0
        images = []
        with torch.no_grad():
            for t in tqdm(range(self.n_steps, 0, -1), desc='Diffusion sampling'):
                x = self._sample_backward_step(x, t, torch.rand((*x.shape, self.num_classes), device=self.device))
                steps += 1
                if steps % stride == 0:
                    images.append(x)
            # if last step is not divisible by stride, we add the last image.
            if steps % stride != 0:
                images.append(x)

        return images
