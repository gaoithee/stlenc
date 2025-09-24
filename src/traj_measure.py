import torch
import copy


class Measure:
    def sample(self, samples=100000, varn=2, points=100):
        # Must be overridden
        pass


class BaseMeasure(Measure):
    def __init__(
        self, mu0=0.0, sigma0=1.0, mu1=0.0, sigma1=1.0, q=0.1, q0=0.5, device="cpu"
    ):
        """

        Parameters
        ----------
        mu0 : mean of normal distribution of initial state, optional
            The default is 0.0.
        sigma0 : standard deviation of normal distribution of initial state, optional
            The default is 1.0.
        mu1 : DOUBLE, optional
            mean of normal distribution of total variation. The default is 0.0.
        sigma1 : standard deviation of normal distribution of total variation, optional
            The default is 1.0.
        q : DOUBLE, optional
            probability of change of sign in derivative. The default is 0.1.
        q0 : DOUBLE, optional
            probability of initial sign of  derivative. The default is 0.5.
        device : 'cpu' or 'cuda', optional
            device on which to run the algorithm. The default is 'cpu'.

        Returns
        -------
        None.

        """
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.q = q
        self.q0 = q0
        self.device = device

    def sample(self, samples=100000, varn=2, points=100):
        """
        Samples a set of trajectories from the basic measure space, with parameters
        passed to the sampler

        Parameters
        ----------
        points : INT, optional
            number of points per trajectory, including initial one. The default is 1000.
        samples : INT, optional
            number of trajectories. The default is 100000.
        varn : INT, optional
            number of variables per trajectory. The default is 2.


        Returns
        -------
        signal : samples x varn x points double pytorch tensor
            The sampled signals.

        """
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU card or CUDA library not available!")

        # generate unif RN
        signal = torch.rand(samples, varn, points, device=self.device)
        # first point is special - set to zero for the moment, and set one point to 1
        signal[:, :, 0] = 0.0
        signal[:, :, -1] = 1.0
        # sorting each trajectory
        signal, _ = torch.sort(signal, 2)
        # computing increments and storing them in points 1 to end
        signal[:, :, 1:] = signal[:, :, 1:] - signal[:, :, :-1]
        # generate initial state, according to a normal distribution
        signal[:, :, 0] = self.mu0 + self.sigma0 * torch.randn(signal[:, :, 0].size())

        # sampling change signs from bernoulli in -1, 1
        derivs = (1 - self.q) * torch.ones(samples, varn, points, device=self.device)
        derivs = 2 * torch.bernoulli(derivs) - 1
        # sampling initial derivative
        derivs[:, :, 0] = self.q0
        derivs[:, :, 0] = 2 * torch.bernoulli(derivs[:, :, 0]) - 1
        # taking the cumulative product along axis 2
        derivs = torch.cumprod(derivs, 2)

        # sampling total variation
        totvar = torch.pow(
            self.mu1 + self.sigma1 * torch.randn(samples, varn, 1, device=self.device),
            2,
         )
        # multiplying total variation and derivatives and making initial point non-invasive
        derivs = derivs * totvar
        derivs[:, :, 0] = 1.0

        # computing trajectories by multiplying and then doing a cumulative sum
        signal = signal * derivs
        signal = torch.cumsum(signal, 2)
        return signal
