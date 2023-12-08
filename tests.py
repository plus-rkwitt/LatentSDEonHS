import torch
import unittest

from core.pathdistribution import SOnPathDistribution, GLnPathDistribution
from core.power_spherical import PowerSpherical
from utils.misc import vec_to_matrix


class TestSOnPathDistribution(unittest.TestCase):
    def setUp(self):
        location = torch.randn(5, 4)
        location = location / location.norm(dim=1, keepdim=True)
        scale = torch.randn(5).pow(3).exp()
        pz0 = PowerSpherical(location, scale)
        def K(t): return torch.randn(5, len(t), 6)
        self.dist = SOnPathDistribution(
            pz0, K, torch.tensor([0.5]), torch.linspace(0, 1, 10)
        )

    def test_basis(self):
        group_dim, basis = self.dist.group_dim, self.dist.basis
        self.assertEqual(group_dim, basis.shape[0])
        self.assertEqual((basis + basis.permute(0, 2, 1)).norm(), 0)

        g = torch.linalg.matrix_exp(vec_to_matrix(self.dist.Kt, basis))
        I = torch.einsum("btij, btkj -> btik", g, g)
        Id = torch.eye(self.dist.dim, self.dist.dim)[None, None, :]
        max_error = (I - Id).abs().max()
        self.assertLess(max_error, 1e-5)

        det = torch.linalg.det(g)
        self.assertLess((det - 1).abs().max(), 1e-5)

    def test_sampling(self):
        sample = self.dist.sample((3,))
        norm = sample.norm(dim=-1)
        val = torch.ones_like(norm)
        max_error = (norm - val).abs().max()
        self.assertLess(max_error, 1e-4)


class TestGLnPathDistribution(unittest.TestCase):
    def setUp(self):
        location = torch.randn(5, 4)
        scale = torch.randn(5, 4).pow(3).exp()
        pz0 = torch.distributions.Normal(location, scale)
        def K(t): return torch.randn(5, len(t), 16)
        self.dist = GLnPathDistribution(
            pz0, K, torch.tensor([0.5]), torch.linspace(0, 1, 10)
        )

    def test_basis(self):
        group_dim, basis = self.dist.group_dim, self.dist.basis
        self.assertEqual(group_dim, basis.shape[0])
        g = torch.linalg.matrix_exp(vec_to_matrix(self.dist.Kt, basis))
        det = torch.linalg.det(g)
        self.assertGreater(det.abs().min(), 1e-6)

    def test_sampling(self):
        pass


from data.activity_provider import HumanActivityProvider, HumanActivityDataset
from data.mnist_provider import RotatingMNISTProvider, RotatingMNISTSDataset
from data.pendulum_provider import PendulumProvider, PendulumDataset
from data.physionet_provider import PhysioNetProvider, PhysioNetDataset


class TestPendulumDatasetRegression(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.modes = ["train", "test", "valid"]
        self.data = {
            mode: PendulumDataset(data_dir="data_dir", task="regression", mode=mode)
            for mode in self.modes
        }
        self.attributes = [
            "inp_obs",
            "inp_msk",
            "inp_tid",
            "evd_obs",
            "evd_msk",
            "evd_tid",
            'aux_obs',
            'aux_tid',
        ]

    def setUp(self):
       pass

    def test_attributes_exist(self):
        target = {attr: True for attr in self.attributes}
        for mode in self.modes:
            ds = self.data[mode]
            has_attributes = {attr: hasattr(ds, attr) for attr in self.attributes}            
            self.assertDictEqual(has_attributes, target)
            self.assertEqual(hasattr(ds, "num_timepoints"), True)

    def test_shapes(self):
        for mode in self.modes:
            ds = self.data[mode]
            target_lengths = {attr: len(ds) for attr in self.attributes}            
            lengths = {attr: len(getattr(ds, attr)) for attr in self.attributes}
            self.assertDictEqual(lengths, target_lengths)

            shapes = {
                attr: getattr(ds, attr).shape
                for attr in ["inp_obs", "inp_msk", "evd_obs", "evd_msk"]
            }
            target_shapes = {
                attr: ds.inp_obs.shape
                for attr in ["inp_obs", "inp_msk", "evd_obs", "evd_msk"]
            }
            self.assertDictEqual(shapes, target_shapes)
            self.assertEqual(ds.inp_tid.shape, ds.evd_tid.shape)
            self.assertEqual(ds.inp_tid.shape, ds.aux_tid.shape)

    def test_dtypes(self):
        target_dtypes ={
                "inp_obs": torch.float32,
                "inp_msk": torch.long,
                "inp_tid": torch.long,
                "evd_obs": torch.float32,
                "evd_msk": torch.long,
                "evd_tid": torch.long,
                "aux_obs": torch.float32,
                "aux_tid": torch.long
            }
        for mode in self.modes:
            ds = self.data[mode]
            dtypes = {attr: getattr(ds, attr).dtype for attr in self.attributes}
            self.assertDictEqual(dtypes, target_dtypes)

    def test_masks(self):
        for mode in self.modes:
            ds = self.data[mode]
            self.assertTrue(torch.all(ds.inp_obs*ds.inp_msk==ds.inp_obs))
            self.assertTrue(torch.all(ds.evd_obs*ds.evd_msk==ds.evd_obs))

            inp_tps_with_value = ds.inp_msk.sum(dim=(2,3,4)) == 0
            self.assertFalse(torch.any(ds.inp_tid[inp_tps_with_value]))
            evd_tps_with_value = ds.evd_msk.sum(dim=(2,3,4)) == 0
            self.assertFalse(torch.any(ds.evd_tid[evd_tps_with_value]))

    def test_aux(self):
        for mode in self.modes:
            ds = self.data[mode]
            aux_norm = ds.aux_obs.norm(dim=-1)
            self.assertLess((aux_norm-1).abs().max(), 1e-4)
            
class TestPendulumDatasetInterpolation(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.modes = ["train", "test", "valid"]
        self.data = {
            mode: PendulumDataset(data_dir="data_dir", task="interpolation", mode=mode)
            for mode in self.modes
        }
        self.attributes = [
            "inp_obs",
            "inp_msk",
            "inp_tid",
            "evd_obs",
            "evd_msk",
            "evd_tid",
            #'aux_obs','aux_msk','aux_tid',
        ]

    def setUp(self):
       pass

    def test_attributes_exist(self):
        target = {attr: True for attr in self.attributes}
        for mode in self.modes:
            ds = self.data[mode]
            has_attributes = {attr: hasattr(ds, attr) for attr in self.attributes}            
            self.assertDictEqual(has_attributes, target)
            self.assertEqual(hasattr(ds, "num_timepoints"), True)

    def test_shapes(self):
        for mode in self.modes:
            ds = self.data[mode]
            target_lengths = {attr: len(ds) for attr in self.attributes}            
            lengths = {attr: len(getattr(ds, attr)) for attr in self.attributes}
            self.assertDictEqual(lengths, target_lengths)

            shapes = {
                attr: getattr(ds, attr).shape
                for attr in ["inp_obs", "inp_msk", "evd_obs", "evd_msk"]
            }
            target_shapes = {
                attr: ds.inp_obs.shape
                for attr in ["inp_obs", "inp_msk", "evd_obs", "evd_msk"]
            }
            self.assertDictEqual(shapes, target_shapes)
            self.assertEqual(ds.inp_tid.shape, ds.evd_tid.shape)

    def test_dtypes(self):
        target_dtypes ={
                "inp_obs": torch.float32,
                "inp_msk": torch.long,
                "inp_tid": torch.long,
                "evd_obs": torch.float32,
                "evd_msk": torch.long,
                "evd_tid": torch.long
            }
        for mode in self.modes:
            ds = self.data[mode]
            dtypes = {attr: getattr(ds, attr).dtype for attr in self.attributes}
            self.assertDictEqual(dtypes, target_dtypes)

    def test_masks(self):
        for mode in self.modes:
            ds = self.data[mode]
            self.assertTrue(torch.all(ds.inp_obs*ds.inp_msk==ds.inp_obs))
            self.assertTrue(torch.all(ds.evd_obs*ds.evd_msk==ds.evd_obs))

            inp_tps_with_value = ds.inp_msk.sum(dim=(2,3,4)) == 0
            self.assertFalse(torch.any(ds.inp_tid[inp_tps_with_value]))
            evd_tps_with_value = ds.evd_msk.sum(dim=(2,3,4)) == 0
            self.assertFalse(torch.any(ds.evd_tid[evd_tps_with_value]))


class TestPhysionetDataset(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.modes = ["train", "test"]
        self.data = {
            mode: PhysioNetDataset(data_dir="data_dir", quantization=0.1, mode=mode)
            for mode in self.modes
        }
        self.attributes = [
            "evd_obs",
            "evd_msk",
            "evd_tid",
        ]

    def setUp(self):
       pass

    def test_attributes_exist(self):
        target = {attr: True for attr in self.attributes}
        for mode in self.modes:
            ds = self.data[mode]
            has_attributes = {attr: hasattr(ds, attr) for attr in self.attributes}            
            self.assertDictEqual(has_attributes, target)
            self.assertEqual(hasattr(ds, "num_timepoints"), True)

    def test_shapes(self):
        for mode in self.modes:
            ds = self.data[mode]
            target_lengths = {attr: len(ds) for attr in self.attributes}            
            lengths = {attr: len(getattr(ds, attr)) for attr in self.attributes}
            self.assertDictEqual(lengths, target_lengths)
            self.assertEqual(ds.evd_obs.shape, ds.evd_msk.shape)

    def test_dtypes(self):
        target_dtypes ={
                "evd_obs": torch.float32,
                "evd_msk": torch.long,
                "evd_tid": torch.long
            }
        for mode in self.modes:
            ds = self.data[mode]
            dtypes = {attr: getattr(ds, attr).dtype for attr in self.attributes}
            self.assertDictEqual(dtypes, target_dtypes)

    def test_masks(self):
        for mode in self.modes:
            ds = self.data[mode]
            self.assertTrue(torch.all(ds.evd_obs*ds.evd_msk==ds.evd_obs))

            evd_tps_with_value = ds.evd_msk.sum(dim=-1) == 0
            self.assertFalse(torch.any(ds.evd_tid[evd_tps_with_value]))

            
class TestRotatingMNISTSDataset(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.modes = ["train", "test", "valid"]
        self.data = {
            mode: RotatingMNISTSDataset(data_dir="data_dir", mode=mode)
            for mode in self.modes
        }
        self.attributes = [
            "inp_obs",
            "inp_msk",
            "inp_tid",
            "evd_obs",
            "evd_msk",
            "evd_tid"
        ]

    def setUp(self):
       pass

    def test_attributes_exist(self):
        target = {attr: True for attr in self.attributes}
        for mode in self.modes:
            ds = self.data[mode]
            has_attributes = {attr: hasattr(ds, attr) for attr in self.attributes}            
            self.assertDictEqual(has_attributes, target)
            self.assertEqual(hasattr(ds, "num_timepoints"), True)

    # def test_shapes(self):
    #     for mode in self.modes:
    #         ds = self.data[mode]
    #         target_lengths = {attr: len(ds) for attr in self.attributes}            
    #         lengths = {attr: len(getattr(ds, attr)) for attr in self.attributes}
    #         self.assertDictEqual(lengths, target_lengths)

    #         shapes = {
    #             attr: getattr(ds, attr).shape
    #             for attr in ["inp_obs", "inp_msk", "evd_obs", "evd_msk"]
    #         }
    #         target_shapes = {
    #             attr: ds.inp_obs.shape
    #             for attr in ["inp_obs", "inp_msk", "evd_obs", "evd_msk"]
    #         }
    #         self.assertDictEqual(shapes, target_shapes)
    #         self.assertEqual(ds.inp_tid.shape, ds.evd_tid.shape)

    def test_dtypes(self):
        target_dtypes ={
                "inp_obs": torch.float32,
                "inp_msk": torch.long,
                "inp_tid": torch.long,
                "evd_obs": torch.float32,
                "evd_msk": torch.long,
                "evd_tid": torch.long
            }
        for mode in self.modes:
            ds = self.data[mode]
            dtypes = {attr: getattr(ds, attr).dtype for attr in self.attributes}
            self.assertDictEqual(dtypes, target_dtypes)

    # def test_masks(self):
    #     for mode in self.modes:
    #         ds = self.data[mode]
    #         self.assertTrue(torch.all(ds.inp_obs*ds.inp_msk==ds.inp_obs))
    #         self.assertTrue(torch.all(ds.evd_obs*ds.evd_msk==ds.evd_obs))

    #         inp_tps_with_value = ds.inp_msk.sum(dim=(2,3,4)) == 0
    #         self.assertFalse(torch.any(ds.inp_tid[inp_tps_with_value]))
    #         evd_tps_with_value = ds.evd_msk.sum(dim=(2,3,4)) == 0
    #         self.assertFalse(torch.any(ds.evd_tid[evd_tps_with_value]))

    def test_removed_angle(self):
        ds = self.data['train']
        removed_angle = 3
        self.assertTrue(torch.all(ds.evd_tid!=removed_angle))

class TestHumanActivityDataset(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.modes = ["train", "test", "valid"]
        self.data = {
            mode: HumanActivityDataset(data_dir="data_dir", mode=mode)
            for mode in self.modes
        }
        self.attributes = [
            "inp_obs",
            "inp_msk",
            "inp_tid",
            "evd_obs",
            "evd_msk",
            "evd_tid",
            'aux_obs',
            'aux_tid',
        ]

    def setUp(self):
       pass

    def test_attributes_exist(self):
        target = {attr: True for attr in self.attributes}
        for mode in self.modes:
            ds = self.data[mode]
            has_attributes = {attr: hasattr(ds, attr) for attr in self.attributes}            
            self.assertDictEqual(has_attributes, target)
            self.assertEqual(hasattr(ds, "num_timepoints"), True)

    def test_shapes(self):
        for mode in self.modes:
            ds = self.data[mode]
            target_lengths = {attr: len(ds) for attr in self.attributes}            
            lengths = {attr: len(getattr(ds, attr)) for attr in self.attributes}
            self.assertDictEqual(lengths, target_lengths)

            shapes = {
                attr: getattr(ds, attr).shape
                for attr in ["inp_obs", "inp_msk", "evd_obs", "evd_msk"]
            }
            target_shapes = {
                attr: ds.inp_obs.shape
                for attr in ["inp_obs", "inp_msk", "evd_obs", "evd_msk"]
            }
            self.assertDictEqual(shapes, target_shapes)
            self.assertEqual(ds.inp_tid.shape, ds.evd_tid.shape)
            self.assertEqual(ds.inp_tid.shape, ds.aux_tid.shape)

    def test_dtypes(self):
        target_dtypes ={
                "inp_obs": torch.float32,
                "inp_msk": torch.long,
                "inp_tid": torch.long,
                "evd_obs": torch.float32,
                "evd_msk": torch.long,
                "evd_tid": torch.long,
                "aux_obs": torch.long,
                "aux_tid": torch.long
            }
        for mode in self.modes:
            ds = self.data[mode]
            dtypes = {attr: getattr(ds, attr).dtype for attr in self.attributes}
            self.assertDictEqual(dtypes, target_dtypes)

    def test_masks(self):
        for mode in self.modes:
            ds = self.data[mode]
            self.assertTrue(torch.all(ds.inp_obs*ds.inp_msk==ds.inp_obs))
            self.assertTrue(torch.all(ds.evd_obs*ds.evd_msk==ds.evd_obs))

            inp_tps_with_value = ds.inp_msk.sum(dim=2) == 0
            self.assertFalse(torch.any(ds.inp_tid[inp_tps_with_value]))
            evd_tps_with_value = ds.evd_msk.sum(dim=2) == 0
            self.assertFalse(torch.any(ds.evd_tid[evd_tps_with_value]))

    def test_aux(self):
        pass

if __name__ == "__main__":
    unittest.main()
