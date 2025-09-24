import torch
import numpy as np


class StlKernel:
    def __init__(
        self,
        measure,
        normalize=True,
        exp_kernel=True,
        sigma2=0.2,
        integrate_time=False,
        samples=100000,
        varn=2,
        points=100,
        boolean=False,
        signals=None,
    ):
        self.traj_measure = measure
        self.exp_kernel = exp_kernel
        self.normalize = normalize
        self.sigma2 = sigma2
        self.samples = samples
        self.varn = varn
        self.points = points
        self.integrate_time = integrate_time
        if signals is not None:
            self.signals = signals
        else:
            self.signals = measure.sample(points=points, samples=samples, varn=varn)
        self.boolean = boolean

    def compute(self, phi1, phi2):
        return self.compute_one_one(phi1, phi2)

    def compute_one_one(self, phi1, phi2):
        phis1: list = [phi1]
        phis2: list = [phi2]
        ker = self.compute_bag_bag(phis1, phis2)
        return ker[0, 0]

    def compute_bag(self, phis, return_robustness=True):
        if self.integrate_time:
            rhos, selfk, len0 = self._compute_robustness_time(phis)
            kernel_matrix = self._compute_kernel_time(
                rhos, rhos, selfk, selfk, len0, len0
            )
        else:
            rhos, selfk = self._compute_robustness_no_time(phis)
            kernel_matrix = self._compute_kernel_no_time(rhos, rhos, selfk, selfk)
            len0 = None
        if return_robustness:
            return kernel_matrix.cpu(), rhos, selfk, len0
        else:
            return kernel_matrix.cpu()

    def compute_one_bag(self, phi1, phis2, return_robustness=False):
        phis1: list = [phi1]
        return self.compute_bag_bag(phis1, phis2, return_robustness)

    def compute_bag_bag(self, phis1, phis2, return_robustness=False):
        if self.integrate_time:
            rhos1, selfk1, len1 = self._compute_robustness_time(phis1)
            rhos2, selfk2, len2 = self._compute_robustness_time(phis2)
            kernel_matrix = self._compute_kernel_time(
                rhos1, rhos2, selfk1, selfk2, len1, len2
            )
        else:
            rhos1, selfk1 = self._compute_robustness_no_time(phis1)
            rhos2, selfk2 = self._compute_robustness_no_time(phis2)
            len1, len2 = [None, None]
            kernel_matrix = self._compute_kernel_no_time(rhos1, rhos2, selfk1, selfk2)
        if return_robustness:
            return kernel_matrix.cpu(), rhos1, rhos2, selfk1, selfk2, len1, len2
        else:
            return kernel_matrix.cpu()

    def compute_one_from_robustness(self, phi, rhos, rho_self, lengths=None, return_robustness=False):
        phis: list = [phi]
        return self.compute_bag_from_robustness(phis, rhos, rho_self, lengths, return_robustness)

    def compute_bag_from_robustness(self, phis, rhos, rho_self, lengths=None, return_robustness=False):
        if self.integrate_time:
            rhos1, selfk1, len1 = self._compute_robustness_time(phis)
            kernel_matrix = self._compute_kernel_time(
                rhos1, rhos, selfk1, rho_self, len1, lengths
            )
        else:
            rhos1, selfk1 = self._compute_robustness_no_time(phis)
            len1 = None
            kernel_matrix = self._compute_kernel_no_time(rhos1, rhos, selfk1, rho_self)
        if return_robustness:
            return kernel_matrix.cpu(), rhos1, selfk1, len1
        else:
            return kernel_matrix.cpu()

    def _compute_robustness_time(self, phis):
        n = self.samples
        p = self.points
        k = len(phis)
        rhos = torch.zeros((k, n, p), device="cpu")
        lengths = torch.zeros(k)
        self_kernels = torch.zeros((k, 1))
        for i, phi in enumerate(phis):
            if self.boolean:
                rho = phi.boolean(self.signals, evaluate_at_all_times=True).float()
                rho[rho == 0.0] = -1.0
            else:
                rho = phi.quantitative(self.signals, evaluate_at_all_times=True)
            actual_p = rho.size()[2]
            rho = rho.reshape(n, actual_p).cpu()
            rhos[i, :, :actual_p] = rho
            lengths[i] = actual_p
            self_kernels[i] = torch.tensordot(
                rho.reshape(1, n, -1), rho.reshape(1, n, -1), dims=[[1, 2], [1, 2]]
            ) / (actual_p * n)
        return rhos, self_kernels, lengths

    def _compute_robustness_no_time(self, phis):
        n = self.samples
        k = len(phis)
        rhos = torch.zeros((k, n), device=self.traj_measure.device)
        self_kernels = torch.zeros((k, 1), device=self.traj_measure.device)
        for i, phi in enumerate(phis):
            if self.boolean:
                rho = phi.boolean(self.signals, evaluate_at_all_times=False).float()
                rho[rho == 0.0] = -1.0
            else:
                rho = phi.quantitative(self.signals, evaluate_at_all_times=False)
            self_kernels[i] = rho.dot(rho) / n
            rhos[i, :] = rho
        return rhos, self_kernels

    def _compute_kernel_time(self, rhos1, rhos2, selfk1, selfk2, len1, len2):
        kernel_matrix = torch.tensordot(rhos1, rhos2, [[1, 2], [1, 2]])
        length_normalizer = self._compute_trajectory_length_normalizer(len1, len2)
        kernel_matrix = kernel_matrix * length_normalizer / self.samples
        if self.normalize:
            kernel_matrix = self._normalize(kernel_matrix, selfk1, selfk2)
        if self.exp_kernel:
            kernel_matrix = self._exponentiate(kernel_matrix, selfk1, selfk2)
        return kernel_matrix

    def _compute_kernel_no_time(self, rhos1, rhos2, selfk1, selfk2):
        kernel_matrix = torch.tensordot(rhos1, rhos2, [[1], [1]])
        kernel_matrix = kernel_matrix / self.samples
        if self.normalize:
            kernel_matrix = self._normalize(kernel_matrix, selfk1, selfk2)
        if self.exp_kernel:
            kernel_matrix = self._exponentiate(kernel_matrix, selfk1, selfk2)
        return kernel_matrix

    @staticmethod
    def _normalize(kernel_matrix, selfk1, selfk2):
        normalize = torch.sqrt(torch.matmul(selfk1, torch.transpose(selfk2, 0, 1)))
        kernel_matrix = kernel_matrix / normalize
        return kernel_matrix

    def _exponentiate(self, kernel_matrix, selfk1, selfk2, sigma2=None):
        if sigma2 is None:
            sigma2 = self.sigma2
        if self.normalize:
            # selfk is (1.0^2 + 1.0^2)
            selfk = 2.0
        else:
            k1 = selfk1.size()[0]
            k2 = selfk2.size()[0]
            selfk = (selfk1 * selfk1).repeat(1, k2) + torch.transpose(
                selfk2 * selfk2, 0, 1
            ).repeat(k1, 1)
        return torch.exp(-(selfk - 2 * kernel_matrix) / (2 * sigma2))

    @staticmethod
    def _compute_trajectory_length_normalizer(len1, len2):
        k1 = len1.size()[0]
        k2 = len2.size()[0]
        y1 = len1.reshape(-1, 1)
        y1 = y1.repeat(1, k2)
        y2 = len2.repeat(k1, 1)
        return 1.0 / torch.min(y1, y2)


class GramMatrix:
    def __init__(self, kernel, formulae, store_robustness=True, sample=False, sampler=None, bag_size=None):
        self.kernel = kernel
        self.formulae_list = formulae
        # if kernel is computed from robustness at time zero only,
        # we store the robustness for each formula and each sample
        # to speed up computation later
        self.store_robustness = store_robustness
        self.dim = len(self.formulae_list) if not bag_size else int(bag_size)
        self.sample = sample  # whether to generate formulae in a controlled manner
        if self.sample:
            self.t = 0.99 if self.kernel.boolean else 0.85
        self.sampler = sampler  # stl formulae generator
        self._compute_gram_matrix()

    def _compute_gram_matrix(self):
        if self.sample:
            gram = torch.zeros(self.dim, self.dim)
            rhos = torch.zeros((self.dim, self.kernel.samples), device=self.kernel.traj_measure.device) if \
                not self.kernel.integrate_time else torch.zeros((self.dim, self.kernel.samples, self.kernel.points),
                                                                device=self.kernel.traj_measure.device)
            lengths = torch.zeros(self.dim) if self.kernel.integrate_time else np.zeros(self.dim)
            kernels = torch.zeros((self.dim, 1), device=self.kernel.traj_measure.device)
            phis = [self.sampler.sample(nvars=self.kernel.varn)]
            gram[0, :1], rhos[0], kernels[0, :], lengths[0] = self.kernel.compute_bag(phis, return_robustness=True)
            while len(phis) < self.dim:
                i = len(phis)
                phi = self.sampler.sample(nvars=self.kernel.varn)
                gram[i, :i], rhos[i], kernels[i, :], lengths[i] = self.kernel.compute_one_from_robustness(
                    phi, rhos[:i, :], kernels[:i, :], lengths[:i], return_robustness=True)
                if torch.sum(gram[i, :i + 1] >= self.t) < 3:
                    phis.append(phi)
                    gram[:i, i] = gram[i, :i]
                    gram[i, i] = kernels[i, :]

            self.formulae_list = phis
            self.gram = gram.cpu()
            self.robustness = rhos if self.store_robustness else None
            self.self_kernels = kernels if self.store_robustness else None
            self.robustness_lengths = lengths if self.store_robustness else None
        else:
            if self.store_robustness:
                k_matrix, rhos, selfk, len0 = self.kernel.compute_bag(
                    self.formulae_list, return_robustness=True
                )
                self.gram = k_matrix
                self.robustness = rhos
                self.self_kernels = selfk
                self.robustness_lengths = len0
            else:
                self.gram = self.kernel.compute_bag(
                    self.formulae_list, return_robustness=False
                )
                self.robustness = None
                self.self_kernels = None
                self.robustness_lengths = None

    def compute_kernel_vector(self, phi):
        if self.store_robustness:
            return self.kernel.compute_one_from_robustness(
                phi, self.robustness, self.self_kernels, self.robustness_lengths
            )
        else:
            return self.kernel.compute_one_bag(phi, self.formulae_list)

    def compute_bag_kernel_vector(self, phis, generate_phis=False, bag_size=None):
        if generate_phis:
            gram_test = torch.zeros(bag_size, self.dim)  # self.dim, bag_size
            rhos_test = torch.zeros((bag_size, self.kernel.samples), device=self.kernel.traj_measure.device) if \
                not self.kernel.integrate_time else torch.zeros((bag_size, self.kernel.samples, self.kernel.points),
                                                                device=self.kernel.traj_measure.device)
            lengths_test = torch.zeros(bag_size) if self.kernel.integrate_time else np.zeros(bag_size)
            kernels_test = torch.zeros((bag_size, 1), device=self.kernel.traj_measure.device)
            phi_test = []
            while len(phi_test) < bag_size:
                i = len(phi_test)
                phi = self.sampler.sample(nvars=self.kernel.varn)
                if self.store_robustness:
                    gram_test[i, :], rhos_test[i], kernels_test[i, :], lengths_test[i] = \
                        self.kernel.compute_one_from_robustness(phi, self.robustness, self.self_kernels,
                                                                self.robustness_lengths, return_robustness=True)
                else:
                    gram_test[i, :], rhos_test[i], _, kernels_test[i, :], _, lengths_test[i], _ = \
                        self.kernel.compute_one_bag(phi, self.formulae_list, return_robustness=True)
                if not ((rhos_test[i] > 0).all() or (rhos_test[i] < 0).all()):
                    phi_test.append(phi)
            return phi_test, gram_test.cpu()
        else:
            if self.store_robustness:
                return self.kernel.compute_bag_from_robustness(
                    phis, self.robustness, self.self_kernels, self.robustness_lengths
                )
            else:
                return self.kernel.compute_bag_bag(phis, self.formulae_list)

    def invert_regularized(self, alpha):
        regularizer = abs(pow(10, alpha)) * torch.eye(self.dim)
        return torch.inverse(self.gram + regularizer)
