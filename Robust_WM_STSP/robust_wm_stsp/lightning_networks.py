import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl


class dDMTSNet(pl.LightningModule):
    """distractedDelayedMatchToSampleNetwork. Class defines RNN for solving a
    distracted DMTS task. Implemented in Pytorch Lightning to enable smooth
    running on multiple GPUs. """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        dt_ann,
        alpha,
        alpha_W,
        g,
        nl,
        lr,
    ):
        super().__init__()
        self.stsp = False
        self.dt_ann = dt_ann
        self.lr = lr
        self.save_hyperparameters()
        self.act_reg = 0
        self.param_reg = 0

        if rnn_type == "vRNN":
            # if model is vanilla RNN
            self.rnn = vRNNLayer(input_size, hidden_size, output_size, alpha, g, nl)
            self.fixed_syn = True

        if rnn_type == "ah":
            # if model is anti-Hebbian
            self.rnn = aHiHRNNLayer(
                input_size, hidden_size, output_size, alpha, alpha_W, nl
            )
            self.fixed_syn = False

        if rnn_type == "stsp":
            # if model is Mongillo/Masse STSP
            self.rnn = stspRNNLayer(
                input_size, hidden_size, output_size, alpha, dt_ann, g, nl
            )
            self.fixed_syn = False
            self.stsp = True

    def forward(self, x):
        # defines foward method using the chosen RNN type
        out_readout, out_hidden, w_hidden, _ = self.rnn(x)
        return out_readout, out_hidden, w_hidden, _

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        inp, out_des, y, test_on, dis_bool = batch
        out_readout, out_hidden, _, _ = self.rnn(inp)

        #accumulate losses. if penalizing activity, then add it to the loss
        if self.act_reg != 0:
            loss = self.act_reg*out_hidden.norm(p = 'fro')
            loss /= out_hidden.shape[0]*out_hidden.shape[1]*out_hidden.shape[2]
        else:
            loss = 0
        
        for i in test_on.unique():
            inds = torch.where(test_on == i)[0]
            test_end = int(i) + int(500 / self.dt_ann)
            response_end = test_end + int(500 / self.dt_ann)
            loss += F.mse_loss(
                out_readout[inds, test_end:response_end],
                out_des[inds, test_end:response_end],
            )
        return loss

    def validation_step(self, batch, batch_idx):
        # defines validation step
        inp, out_des, y, test_on, dis_bool = batch
        out_readout, _, _, _ = self.rnn(inp)

        accs = np.zeros(out_readout.shape[0])
        # test model performance
        for i in range(out_readout.shape[0]):
            curr_max = (
                out_readout[
                    i,
                    int(test_on[i])
                    + int(500 / self.dt_ann) : int(test_on[i])
                    + 2 * int(500 / self.dt_ann),
                    :-1,
                ]
                .argmax(dim=1)
                .cpu()
                .detach()
                .numpy()
            )
            accs[i] = (y[i].item() == curr_max).sum() / len(curr_max)

        self.log("val_acc", accs.mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # by default, we use an L2 weight decay on all parameters.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.param_reg)

        # lr_scheduler = {'scheduler':  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1),"monitor": 'val_acc'}
        return [optimizer]  # ,[lr_scheduler]


class vRNNLayer(pl.LightningModule):
    """Vanilla RNN layer in continuous time."""

    def __init__(self, input_size, hidden_size, output_size, alpha, g, nonlinearity):
        super(vRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.inv_sqrt_alpha = 1 / np.sqrt(alpha)
        self.cont_stab = False
        self.disc_stab = True
        self.g = g
        self.process_noise = 0.05

        # set nonlinearity of the vRNN
        self.nonlinearity = nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        if nonlinearity == "relu":
            self.phi = F.relu
        if nonlinearity == "none":
            print("Nl = none")
            self.phi = torch.nn.Identity()

        # initialize the input-to-hidden weights
        self.weight_ih = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, input_size))
        )

        # initialize the hidden-to-output weights
        self.weight_ho = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (output_size, hidden_size))
        )

        # initialize the hidden-to-hidden weights
        self.W = nn.Parameter(
            torch.normal(0, self.g / np.sqrt(hidden_size), (hidden_size, hidden_size))
        )

        # initialize the output bias weights
        self.bias_oh = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (1, output_size))
        )

        # initialize the hidden bias weights
        self.bias_hh = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(hidden_size), (1, hidden_size))
        )

        # define mask for weight matrix do to structural perturbation experiments
        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_()
            > self.struc_p_0,
        )

    def forward(self, input):

        # initialize state at the origin. randn is there just in case we want to play with this later.
        state = 0 * torch.randn(input.shape[0], self.hidden_size, device=self.device)

        # defines process noise using Euler-discretization of stochastic differential equation defining the RNN
        noise = (
            1.41
            * self.process_noise
            * torch.randn(
                input.shape[0], input.shape[1], self.hidden_size, device=self.device
            )
        )

        # for storing RNN outputs and hidden states
        outputs = []
        states = []

        # loop over input
        for i in range(input.shape[1]):

            # compute output
            hy = state @ self.weight_ho.T + self.bias_oh

            # save output and hidden states
            outputs += [hy]
            states += [state]

            # compute the RNN update
            fx = -state + self.phi(
                state @ (self.W * self.struc_perturb_mask)
                + input[:, i, :] @ self.weight_ih.T
                + self.bias_hh
                + self.inv_sqrt_alpha * noise[:, i, :]
            )

            # step hidden state foward using Euler discretization
            state = state + self.alpha * (fx)

        # organize states and outputs and return
        return (
            torch.stack(outputs).permute(1, 0, 2),
            torch.stack(states).permute(1, 0, 2),
            noise,
            None,
        )


class aHiHRNNLayer(pl.LightningModule):
    """
    Network for anti-Hebbian / Inhibitory-Hebbian plasticity
    """

    def __init__(
        self, input_size, hidden_size, output_size, alpha, alpha_W, nonlinearity
    ):
        super(aHiHRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.alpha = alpha
        self.alpha_W = alpha_W
        self.root_inv_alpha = 1 / np.sqrt(self.alpha)
        self.root_inv_hidden = 1 / np.sqrt(hidden_size)
        self.inv_hidden = 1 / hidden_size
        self.inv_hidden_power_4 = hidden_size ** (-0.25)

        self.root_inv_inp = 1 / np.sqrt(input_size)

        self.nonlinearity = nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        if nonlinearity == "relu":
            self.phi = F.relu
        if nonlinearity == "none":
            self.phi = torch.nn.Identity()

        self.S = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size).uniform_(-0.5, 0.5)
        )
        # self.register_buffer("K", torch.FloatTensor(hidden_size, hidden_size).uniform_(-.5, .5))
        self.gamma_val = 1/200
        self.register_buffer("gamma", self.gamma_val* torch.ones(1))
        self.register_buffer("beta", torch.ones(1))

        self.weight_ih = nn.Parameter(
            torch.FloatTensor(hidden_size, input_size).uniform_(
                -self.root_inv_inp, self.root_inv_inp
            )
        )

        self.weight_ho = nn.Parameter(
            torch.FloatTensor(output_size, hidden_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        self.bias_oh = nn.Parameter(
            torch.FloatTensor(1, output_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )
        self.bias_hh = nn.Parameter(
            torch.FloatTensor(1, hidden_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # self.half_I = 0.5*torch.eye(hidden_size,device = self.device)
        self.register_buffer("half_I", 0.5 * torch.eye(hidden_size))
        # self.half_I = self.half_I.type_as(self.half_I)

        # self.ones_mat = torch.ones((hidden_size,hidden_size),device = self.device)
        self.register_buffer("ones_mat", torch.ones((hidden_size, hidden_size)))
        # self.ones_mat = self.ones_mat.type_as(self.ones_mat)

        self.eps = 0.95
        self.weight_inds_to_save_1 = torch.tensor(
            np.random.choice(np.arange(self.hidden_size), 50)
        ).long()
        self.weight_inds_to_save_2 = torch.tensor(
            np.random.choice(np.arange(self.hidden_size), 50)
        ).long()

        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_()
            > self.struc_p_0,
        )
        self.process_noise = 0.05

    def forward(self, input):
        """Forward method for anti-Hebbian RNN, which is desribed by two coupled dynamical systems as desribed in Kozachkov et al (2020), PLoS Comp Bio:

        dxdt = -x + Wx + u(t)
        dWdt = -gamma W - K.*(xx')

        where K is positive semi-definite and has positive elements.

        """

        # initialize x,W at the origin
        x_state = torch.zeros(input.shape[0], self.hidden_size, device=self.device)

        W_state = torch.zeros(input.shape[0], self.hidden_size, self.hidden_size, device=self.device)

        # define neural process noise
        neural_noise = (
            1.41
            * self.process_noise
            * self.root_inv_alpha
            * torch.randn(
                input.shape[0], input.shape[1], self.hidden_size, device=self.device
            )
        )

        # for storing neural outputs, neural hidden states, and synaptic states
        
        outputs = []
        x_states = []
        W_states = []

        # loop over input dim
        for i in range(input.shape[1]):

            # compute and store neural output
            hy = x_state @ self.weight_ho.T + self.bias_oh
            outputs += [hy]

            # store neural hidden state
            x_states += [x_state]

            # store synaptic state
            W_states += [
                W_state[:, self.weight_inds_to_save_1, self.weight_inds_to_save_2]
            ]

            #assert (W_state.transpose(1, 2) == W_state).all(), print("W is not symmetric!")

            # compute outer product in synaptic learning rule. use einsum magic to do it across batches.
            hebb_term = torch.einsum(
                "bq, bk-> bqk", self.phi(x_state), self.phi(x_state)
            )

            # compute K as the sum of three terms:
            # (S**2)'(S**2) ensures K is positive semidefinite and has non-negative elements
            # self.ones_mat ensures K has strictly positive elements
            # half_I ensures that K is positive-definite (not needed but it seems to help)
            K = (
                (((self.S) ** 2).T @ ((self.S) ** 2))
                + 1e-2 * self.ones_mat
                + 1e-2 * self.half_I
            )
            K *= self.struc_perturb_mask

            # batch matrix multiply W and x for updating x
            prod = torch.bmm(
                W_state, x_state.view(input.shape[0], self.hidden_size, 1)
            ).view(input.shape[0], self.hidden_size)

            # compute x update
            fx = -(self.beta) * x_state + self.phi(
                prod
                + input[:, i, :] @ self.weight_ih.T
                + self.bias_hh
                + neural_noise[:, i, :]
            )

            # compute W update
            fW = -K * hebb_term - (self.gamma) * W_state

            # step x and W forward
            x_state = x_state + self.alpha * (fx)
            W_state = W_state + self.alpha * (fW)

        # organize results and return
        
        return (
            torch.stack(outputs).permute(1, 0, 2),
            torch.stack(x_states).permute(1, 0, 2),
            torch.stack(W_states).permute(1, 0, 2),
            W_state,
        )
        
        '''
        return (
            torch.stack(outputs).permute(1, 0, 2),[],[],W_state
        )
        '''
        


class stspRNNLayer(pl.LightningModule):
    """Implements the RNN of Mongillo/Masse, using pre-synaptic STSP"""

    def __init__(
        self, input_size, hidden_size, output_size, alpha, dt, g, nonlinearity
    ):
        super(stspRNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.inv_sqrt_alpha = 1 / np.sqrt(alpha)
        self.root_inv_hidden = 1 / np.sqrt(hidden_size)
        self.g = g
        self.dt = dt
        self.f_out = nn.Softplus()

        # define time-constants for the network, in units of ms
        self.tau_x_facil = 200
        self.tau_u_facil = 1500
        self.U_facil = 0.15

        self.tau_x_depress = 1500
        self.tau_u_depress = 200
        self.U_depress = 0.45

        # define nonlinearity for the neural dynamics
        self.nonlinearity = nonlinearity
        if nonlinearity == "tanh":
            self.phi = torch.tanh
        if nonlinearity == "relu":
            self.phi = F.relu
        if nonlinearity == "retanh":
            self.phi = torch.nn.ReLU(torch.nn.Tanh())
        if nonlinearity == "none":
            self.phi = torch.nn.Identity()

        # initialize input-to-hidden weights
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(hidden_size, input_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # initialize hidden to output weights
        self.weight_ho = nn.Parameter(
            torch.FloatTensor(output_size, hidden_size).uniform_(
                -self.root_inv_hidden, self.root_inv_hidden
            )
        )

        # initialize hidden-to-hidden weights
        W = torch.FloatTensor(hidden_size, hidden_size).uniform_(
            -self.root_inv_hidden, self.root_inv_hidden
        )
        W.log_normal_(0, self.g / np.sqrt(hidden_size))
        W = F.relu(W)
        W /= 10 * (torch.linalg.vector_norm(W, ord=2))
        self.W = nn.Parameter(W)

        # define seperate inhibitory and excitatory neural populations
        diag_elements_of_D = torch.ones(self.hidden_size)
        diag_elements_of_D[int(0.8 * self.hidden_size) :] = -1
        syn_inds = torch.arange(self.hidden_size)
        syn_inds_rand = torch.randperm(self.hidden_size)
        diag_elements_of_D = diag_elements_of_D[syn_inds_rand]
        D = diag_elements_of_D.diag_embed()

        self.register_buffer("D", D)

        self.register_buffer("facil_syn_inds", syn_inds[: int(self.hidden_size / 2)])
        self.register_buffer("depress_syn_inds", syn_inds[int(self.hidden_size / 2) :])

        # time constants
        tau_x = torch.ones(self.hidden_size)
        tau_x[self.facil_syn_inds] = self.tau_x_facil
        tau_x[self.depress_syn_inds] = self.tau_x_depress
        self.register_buffer("Tau_x", (1 / tau_x))

        tau_u = torch.ones(self.hidden_size)
        tau_u[self.facil_syn_inds] = self.tau_u_facil
        tau_u[self.depress_syn_inds] = self.tau_u_depress
        self.register_buffer("Tau_u", (1 / tau_x))

        U = torch.ones(self.hidden_size)
        U[self.facil_syn_inds] = self.U_facil
        U[self.depress_syn_inds] = self.U_depress
        self.register_buffer("U", U)

        # initialize output bias
        self.bias_oh = nn.Parameter(
            0 * torch.normal(0, 1 / np.sqrt(hidden_size), (1, output_size))
        )

        # initialize hidden bias
        self.bias_hh = nn.Parameter(
            0 * torch.normal(0, 1 / np.sqrt(hidden_size), (1, hidden_size))
        )

        # for structurally perturbing weight matrix
        self.struc_p_0 = 0
        self.register_buffer(
            "struc_perturb_mask",
            torch.FloatTensor(self.hidden_size, self.hidden_size).uniform_()
            > self.struc_p_0,
        )
        self.process_noise = 0.05

    def forward(self, input):

        # initialize neural state and synaptic states
        state = 0 * torch.randn(input.shape[0], self.hidden_size, device=self.device)
        u_state = 0 * torch.rand(input.shape[0], self.hidden_size, device=self.device)
        x_state = torch.ones(input.shape[0], self.hidden_size, device=self.device)

        # defines process noise
        noise = (
            1.41
            * self.process_noise
            * torch.randn(
                input.shape[0], input.shape[1], self.hidden_size, device=self.device
            )
        )

        # for storing neural outputs, hidden states, and synaptic states
        outputs = []
        states = []
        states_x = []
        states_u = []

        for i in range(input.shape[1]):

            # compute and save neural output
            hy = state @ self.weight_ho.T + self.bias_oh
            outputs += [hy]

            # save neural and synaptic hidden states
            states += [state]
            states_x += [x_state]
            states_u += [u_state]

            # compute update for synaptic variables
            fx = (1 - x_state) * self.Tau_x - u_state * x_state * state * (
                self.dt / 1000
            )
            fu = (self.U - u_state) * self.Tau_u + self.U * (1 - u_state) * state * (
                self.dt / 1000
            )

            # define modulated presynaptic input based on STSP rule
            I = (x_state * state * u_state) @ (
                (self.D @ F.relu(self.W)) * self.struc_perturb_mask
            )

            # compute neural update
            fstate = -state + self.phi(
                I
                + input[:, i, :] @ self.weight_ih.T
                + self.bias_hh
                + self.inv_sqrt_alpha * noise[:, i, :]
            )

            # step neural and synaptic states forward
            state = state + self.alpha * fstate
            x_state = torch.clamp(x_state + self.alpha * fx, min=0, max=1)
            u_state = torch.clamp(u_state + self.alpha * fu, min=0, max=1)

        # organize and return variables
        x_hidden = torch.stack(states_x).permute(1, 0, 2)
        u_hidden = torch.stack(states_u).permute(1, 0, 2)

        return (
            torch.stack(outputs).permute(1, 0, 2),
            torch.stack(states).permute(1, 0, 2),
            torch.cat((x_hidden, u_hidden), dim=2),
            noise,
        )

