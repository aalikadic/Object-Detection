import unittest
import torch
import os
from copy import deepcopy

from model.resnet import BasicLayer, UpSampleLayer, ResNet18
from model.centernet_head import CenterNetHead
from model.model_utils import load_weights, create_model

assert torch.cuda.is_available(), "DCN (deformable conv) doesn't work on cpu"
device = torch.device('cuda')

root = os.path.dirname(os.path.abspath(__file__)).split('/')
root = os.path.join('/', *root[:-1])
#checkpoint_path = os.path.join(root, 'pre-trained-models', 'voc07_prediction_for_images_09-13-15-39.pt')
checkpoint_path = r'C:\Users\alika\Desktop\2.semestar\AIRO\Lab1\pre-trained-models\voc07_prediction_for_images_09-13-15-39.pt'
assert os.path.exists(checkpoint_path), \
    "Checkpoint does not exist in {}, declare full path to checkpoint in line 13".format(checkpoint_path)


class ForwardHook:
    """Forward hook to get size of intermediate tensors computed by a torch.nn.Module (i.e. a layer)"""
    def __init__(self, name, module):
        """ForwardHook constructor

        Args:
            name (str): module's name
            module (torch.nn.Module): module to register hook
        """
        self.name = name
        self.out_size = None
        self.handle = module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        """Function to be registered as forward hook

        Args:
            module (torch.nn.Module): module to which hook is registered
            input (torch.Tensor): module's input
            output (torch.Tensor): module's output
        """
        # take the size of output
        self.out_size = torch.tensor(output.size())

    def remove(self):
        self.handle.remove()


class BackwardHook:
    """Backward hook to check gradient magnitude of parameters (i.e. weights & biases)"""
    def __init__(self, name, param, is_cuda=False):
        """Constructor of BackwardHook

        Args:
            name (str): name of parameter
            param (torch.nn.Parameter): the parameter hook is registered to
            is_cuda (bool): whether parameter is on cuda or not
        """
        self.name = name
        self.hook_handle = param.register_hook(self.hook)
        self.grad_mag = -1.0
        self.is_cuda = is_cuda

    def hook(self, grad):
        """Function to be registered as backward hook

        Args:
            grad (torch.Tensor): gradient of a parameter W (i.e. dLoss/dW)
        """
        if not self.is_cuda:
            self.grad_mag = torch.norm(torch.flatten(grad, start_dim=0).detach())
        else:
            self.grad_mag = torch.norm(torch.flatten(grad, start_dim=0).detach().cpu())

    def remove(self):
        self.hook_handle.remove()


class TestResidualLayer(unittest.TestCase):
    def test_size_retain_case(self):
        inC, outC = 3, 3
        input = torch.randn(2, 3, 64, 64)
        expected_size = torch.tensor([2, 3, 64, 64])

        # declare model & register forward hook for BatchNorm layers
        res_layer_retain_size = BasicLayer(inC, outC, halve_size=False)
        fhooks = [ForwardHook(name, module) for name, module in res_layer_retain_size.named_modules()
                  if 'bn' in name]

        # forward pass
        _ = res_layer_retain_size(input)

        # check size
        for hook in fhooks:
            self.assertTrue(torch.all(hook.out_size == expected_size).item(),
                            msg="Wrong size at layer {}, expected {}, got {}".format(
                                hook.name, expected_size, hook.out_size
                            ))

    def test_size_halve_case(self):
        inC, outC = 3, 6
        input = torch.randn(2, 3, 64, 64)
        expected_size = torch.tensor([2, 6, 32, 32])

        # declare model & register forward hook for BatchNorm layers
        res_layer_halve_size = BasicLayer(inC, outC, halve_size=True)
        fhooks = [ForwardHook(name, module) for name, module in res_layer_halve_size.named_modules()
                  if 'bn' in name]

        # forward pass
        _ = res_layer_halve_size(input)

        # check size
        for hook in fhooks:
            self.assertTrue(torch.all(hook.out_size == expected_size).item(),
                            msg="Wrong size at layer {}, expected {}, got {}".format(
                                hook.name, expected_size, hook.out_size
                            ))

    def test_grad_magnitude_retain_case(self):
        inC, outC = 3, 3
        input = torch.randn(2, 3, 64, 64)
        target = torch.randn_like(input)
        loss_fnc = torch.nn.L1Loss()

        # declare model & register forward hook for BatchNorm layers
        res_layer_retain_size = BasicLayer(inC, outC, halve_size=False)
        bw_hooks = [BackwardHook(name, param) for name, param in res_layer_retain_size.named_parameters()]

        # forward pass
        out = res_layer_retain_size(input)
        # compute loss
        loss = loss_fnc(out, target)
        # backward pass
        loss.backward()

        # check gradient magnitude
        for hook in bw_hooks:
            self.assertGreater(hook.grad_mag.item(), 1e-5, msg="Zero grad at {}".format(hook.name))

    def test_grad_magnitude_halve_case(self):
        inC, outC = 3, 6
        input = torch.randn(2, inC, 64, 64)
        target = torch.randn(2, outC, 32, 32)
        loss_fnc = torch.nn.L1Loss()

        # declare model & register forward hook for BatchNorm layers
        res_layer_halve_size = BasicLayer(inC, outC, halve_size=True)
        bw_hooks = [BackwardHook(name, param) for name, param in res_layer_halve_size.named_parameters()]

        # forward pass
        out = res_layer_halve_size(input)
        # compute loss
        loss = loss_fnc(out, target)
        # backward pass
        loss.backward()

        # check gradient magnitude
        for hook in bw_hooks:
            self.assertGreater(hook.grad_mag.item(), 1e-5, msg="Zero grad at {}".format(hook.name))

    def test_size_upsample_layer(self):
        inC, outC = 6, 3
        input = torch.randn(2, inC, 32, 32).cuda()
        expected_size = {
            'bn1': torch.tensor([2, outC, 32, 32]),
            'bn2': torch.tensor([2, outC, 64, 64])
        }
        layer = UpSampleLayer(inC, outC).cuda()
        fw_hooks = [ForwardHook(name, module) for name, module in layer.named_modules() if 'bn' in name]
        out = layer(input)
        # test size
        for hook in fw_hooks:
            self.assertTrue((hook.out_size == expected_size[hook.name]).all().item(),
                            msg="Wrong size at {}".format(hook.name))

        # clean up
        for hook in fw_hooks:
            hook.remove()
        del input, out, layer

    def test_grad_mag_upsample_layer(self):
        inC, outC = 6, 3
        input = torch.randn(2, inC, 32, 32).cuda()
        target = torch.randn(2, outC, 64, 64).cuda()
        loss_fnc = torch.nn.L1Loss().cuda()

        layer = UpSampleLayer(inC, outC).cuda()
        bw_hooks = [BackwardHook(name, param) for name, param in layer.named_parameters()]

        out = layer(input)
        loss = loss_fnc(out, target)
        loss.backward()

        # for hook in bw_hooks:
        #     print("{} grad mag: {}".format(hook.name, hook.grad_mag.item()))

        # test gradient magnitude
        for hook in bw_hooks:
            if 'defconv.bias' == hook.name:
                # deconv.bias is expected to have zero (or very very small) grad
                # because of the presence of BatchNorm2d following it
                continue
            self.assertGreater(hook.grad_mag.item(), 1e-5, msg="Zero grad at {}".format(hook.name))

        # clean up
        for hook in bw_hooks:
            hook.remove()
        del input, target, out, loss, layer

    def test_size_resnet18_backbone(self):
        """Test size of intermediate tensor from the 1st layer to the last feature map (no head yet)"""
        N = 2  # batch size
        input = torch.randn(N, 3, 384, 384).cuda()
        expected_size = {
            'conv1': torch.tensor([N, 64, 192, 192]),
            'bn1': torch.tensor([N, 64, 192, 192]),
            'maxpool1': torch.tensor([N, 64, 96, 96]),
            'conv2_1': torch.tensor([N, 64, 96, 96]),
            'conv2_2': torch.tensor([N, 64, 96, 96]),
            'conv3_1': torch.tensor([N, 128, 48, 48]),
            'conv3_2': torch.tensor([N, 128, 48, 48]),
            'conv4_1': torch.tensor([N, 256, 24, 24]),
            'conv4_2': torch.tensor([N, 256, 24, 24]),
            'conv5_1': torch.tensor([N, 512, 12, 12]),
            'conv5_2': torch.tensor([N, 512, 12, 12]),
            'up1': torch.tensor([N, 256, 24, 24]),
            'up2': torch.tensor([N, 128, 48, 48]),
            'up3': torch.tensor([N, 64, 96, 96]),
        }

        resnet = ResNet18().cuda()
        fw_hooks = [ForwardHook(name, module) for name, module in resnet.named_modules()
                    if '.' not in name and name != '']
        out = resnet(input)

        # check size
        for hook in fw_hooks:
            self.assertTrue((hook.out_size == expected_size[hook.name]).all().item(),
                            msg="Wrong size at {}".format(hook.name))

        # clean up
        for hook in fw_hooks:
            hook.remove()
        del input, out, resnet

    def test_grad_mag_resnet18_backbone(self):
        """Test size of intermediate tensor from the 1st layer to the last feature map (no head yet)"""
        N = 2  # batch size
        input = torch.randn(2, 3, 384, 384).cuda()
        target = torch.randn(N, 64, 96, 96).cuda()
        loss_fnc = torch.nn.L1Loss().cuda()
        resnet = ResNet18().cuda()
        bw_hooks = [BackwardHook(name, param) for name, param in resnet.named_parameters()]

        # forward pass
        out = resnet(input)
        loss = loss_fnc(out, target)
        # backward pass
        loss.backward()

        # check gradient magnitude
        for hook in bw_hooks:
            if 'defconv.bias' in hook.name:
                # deconv.bias is expected to have zero (or very very small) grad
                # because of the presence of BatchNorm2d following it
                continue
            self.assertGreater(hook.grad_mag.item(), 1e-5, msg="Zero grad at {}".format(hook.name))

        # clean up
        for hook in bw_hooks:
            hook.remove()
        del input, target, out, loss, resnet

    def test_size_head(self):
        N, H, W = 2, 96, 96
        inC, midC, nClasses = 64, 256, 20
        input = torch.randn(N, inC, H, W)

        expected_size = {
            'heatmap': torch.tensor([N, nClasses, H, W]),
            'wh': torch.tensor([N, 2, H, W]),
            'offset': torch.tensor([N, 2, H, W])
        }

        head = CenterNetHead(inC, nClasses, midC)

        fw_hooks = [ForwardHook(name, module) for name, module in head.named_modules()
                    if '.' not in name and name != '']
        out = head(input)

        # check size
        for hook in fw_hooks:
            self.assertTrue((hook.out_size == expected_size[hook.name]).all().item(),
                            msg="Wrong size at {}".format(hook.name))

    def test_grad_mag_head(self):
        N, H, W = 2, 96, 96
        inC, midC, nClasses = 64, 256, 20
        input = torch.randn(N, inC, H, W)
        target = torch.randn(N, nClasses+4, H, W)
        loss_fnc = torch.nn.L1Loss()

        head = CenterNetHead(inC, nClasses, midC)
        bw_hooks = [BackwardHook(name, param) for name, param in head.named_parameters()]

        # forward pass
        out = head(input)
        loss = loss_fnc(out, target)
        # backward pass
        loss.backward()

        # check gradient magnitude
        for hook in bw_hooks:
            self.assertGreater(hook.grad_mag.item(), 1e-5, msg="Zero grad at {}".format(hook.name))



if __name__ == '__main__':
    unittest.main()
