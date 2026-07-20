# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import math
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DPDescrptDPA1
from deepmd.pt.model.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from .test_mlp import (
    get_tols,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestDescrptSeAtten(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_get_numb_attn_layer(self) -> None:
        """Cover both code paths: attn_layer == 0 and attn_layer > 0."""
        dd0 = DescrptDPA1(
            self.rcut, self.rcut_smth, self.sel_mix, self.nt, attn_layer=0
        ).to(env.DEVICE)
        self.assertEqual(dd0.get_numb_attn_layer(), 0)
        dd2 = DescrptDPA1(
            self.rcut, self.rcut_smth, self.sel_mix, self.nt, attn_layer=2
        ).to(env.DEVICE)
        self.assertEqual(dd2.get_numb_attn_layer(), 2)

    def test_consistency(
        self,
    ) -> None:
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for idt, sm, to, tm, prec, ect in itertools.product(
            [False, True],  # resnet_dt
            [False, True],  # smooth_type_embedding
            [False, True],  # type_one_side
            ["concat", "strip"],  # tebd_input_mode
            [
                "float64",
            ],  # precision
            [False, True],  # use_econf_tebd
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            err_msg = f"idt={idt} prec={prec}"

            # dpa1 new impl
            dd0 = DescrptDPA1(
                self.rcut,
                self.rcut_smth,
                self.sel_mix,
                self.nt,
                attn_layer=2,
                precision=prec,
                resnet_dt=idt,
                smooth_type_embedding=sm,
                type_one_side=to,
                tebd_input_mode=tm,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)
            dd0.se_atten.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.se_atten.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            )
            # serialization
            dd1 = DescrptDPA1.deserialize(dd0.serialize())
            rd1, _, _, _, _ = dd1(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd1.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )
            # dp impl. `mapping` is passed because that is the production
            # invocation (DPAtomicModel.forward_atomic always forwards it);
            # the dense `.call()` must give the same answer with or without
            # it -- mapping only enables ghost folding on graph routes, it
            # must never change the dense numerics.
            dd2 = DPDescrptDPA1.deserialize(dd0.serialize())
            rd2, _, _, _, _ = dd2.call(
                self.coord_ext,
                self.atype_ext,
                self.nlist,
                self.mapping,
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd2,
                rtol=rtol,
                atol=atol,
                err_msg=err_msg,
            )

    def test_jit(
        self,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for idt, prec, sm, to, tm, ect in itertools.product(
            [
                False,
            ],  # resnet_dt
            [
                "float64",
            ],  # precision
            [False, True],  # smooth_type_embedding
            [
                False,
            ],  # type_one_side
            ["concat", "strip"],  # tebd_input_mode
            [False, True],  # use_econf_tebd
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            err_msg = f"idt={idt} prec={prec}"
            # dpa1 new impl
            dd0 = DescrptDPA1(
                self.rcut,
                self.rcut_smth,
                self.sel,
                self.nt,
                precision=prec,
                resnet_dt=idt,
                smooth_type_embedding=sm,
                type_one_side=to,
                tebd_input_mode=tm,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            )
            dd0.se_atten.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.se_atten.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            # dd1 = DescrptDPA1.deserialize(dd0.serialize())
            model = torch.jit.script(dd0)
            # model = torch.jit.script(dd1)


class TestDPA1AngularMoments(unittest.TestCase):
    """Test the dense PyTorch angular moment basis of DPA1."""

    dtype = torch.float64

    @staticmethod
    def _build_descriptor(lmax: int) -> DescrptDPA1:
        return DescrptDPA1(
            rcut=3.0,
            rcut_smth=2.5,
            sel=4,
            ntypes=1,
            neuron=[4, 8, 8],
            axis_neuron=4,
            lmax=lmax,
            tebd_dim=2,
            tebd_input_mode="strip",
            set_davg_zero=True,
            attn_layer=0,
            precision="float64",
            concat_output_tebd=False,
            seed=11,
        ).to(env.DEVICE)

    @classmethod
    def _evaluate(
        cls,
        descriptor: DescrptDPA1,
        neighbors: torch.Tensor,
        *,
        requires_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coord = torch.cat(
            [
                torch.zeros(
                    (1, 3),
                    dtype=cls.dtype,
                    device=env.DEVICE,
                ),
                neighbors,
            ]
        ).reshape(1, -1)
        coord.requires_grad_(requires_grad)
        atype = torch.zeros((1, 5), dtype=torch.long, device=env.DEVICE)
        nlist = torch.tensor(
            [[[1, 2, 3, 4]]],
            dtype=torch.long,
            device=env.DEVICE,
        )
        result = descriptor(coord, atype, nlist)[0]
        return result, coord

    @classmethod
    def _square_directions(cls) -> torch.Tensor:
        return torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=cls.dtype,
            device=env.DEVICE,
        )

    @classmethod
    def _tetrahedral_directions(cls) -> torch.Tensor:
        return torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=cls.dtype,
            device=env.DEVICE,
        ) / math.sqrt(3.0)

    def test_lmax_two_resolves_quadrupole_collision(self) -> None:
        square = self._square_directions()
        tetrahedral = self._tetrahedral_directions()

        degree_one = self._build_descriptor(lmax=1).eval()
        square_l1, _ = self._evaluate(degree_one, square)
        tetrahedral_l1, _ = self._evaluate(degree_one, tetrahedral)
        torch.testing.assert_close(square_l1, tetrahedral_l1, atol=1e-12, rtol=1e-12)

        degree_two = self._build_descriptor(lmax=2).eval()
        square_l2, _ = self._evaluate(degree_two, square)
        tetrahedral_l2, _ = self._evaluate(degree_two, tetrahedral)
        self.assertGreater(
            torch.linalg.vector_norm(square_l2 - tetrahedral_l2).item(),
            1e-8,
        )

        restored = DescrptDPA1.deserialize(degree_two.serialize()).to(env.DEVICE).eval()
        restored_square, _ = self._evaluate(restored, square)
        torch.testing.assert_close(restored_square, square_l2)
        scripted_square, _ = self._evaluate(torch.jit.script(degree_two), square)
        torch.testing.assert_close(scripted_square, square_l2)

    def test_lmax_two_is_rotation_and_permutation_invariant(self) -> None:
        descriptor = self._build_descriptor(lmax=2).eval()
        descriptor.se_atten.mean[..., 0] = 0.25
        descriptor.se_atten.stddev[..., 0] = 1.75
        neighbors = torch.tensor(
            [
                [1.1, 0.2, -0.1],
                [-0.4, 0.9, 0.3],
                [0.2, -0.5, 1.2],
                [-0.7, -0.3, -0.8],
            ],
            dtype=self.dtype,
            device=env.DEVICE,
        )
        rotation = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=self.dtype,
            device=env.DEVICE,
        )

        reference, _ = self._evaluate(descriptor, neighbors)
        rotated, _ = self._evaluate(descriptor, neighbors @ rotation.T)
        permuted, _ = self._evaluate(descriptor, neighbors[[2, 0, 3, 1]])

        torch.testing.assert_close(rotated, reference, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(permuted, reference, atol=1e-10, rtol=1e-10)

    def test_lmax_two_coordinate_derivatives_are_finite(self) -> None:
        descriptor = self._build_descriptor(lmax=2)
        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [-0.3, 0.9, -0.2]]],
            dtype=self.dtype,
            device=env.DEVICE,
            requires_grad=True,
        )
        atype = torch.zeros((1, 3), dtype=torch.long, device=env.DEVICE)
        nlist = torch.tensor(
            [[[1, 2, -1, -1]]],
            dtype=torch.long,
            device=env.DEVICE,
        )
        result = descriptor(coord.reshape(1, -1), atype, nlist)[0]
        (first_derivative,) = torch.autograd.grad(
            result.sum(),
            coord,
            create_graph=True,
        )
        (second_derivative,) = torch.autograd.grad(
            first_derivative.square().sum(),
            coord,
        )

        self.assertTrue(torch.isfinite(first_derivative).all())
        self.assertTrue(torch.isfinite(second_derivative).all())
        self.assertGreater(first_derivative.abs().max().item(), 0.0)
        self.assertGreater(second_derivative.abs().max().item(), 0.0)
