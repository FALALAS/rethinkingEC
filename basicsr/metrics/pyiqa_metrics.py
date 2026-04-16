import torch
import numpy as np
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_pyiqa_psnr(img, img2, crop_border=0, input_order='HWC', test_y_channel=False, **kwargs):
    """使用 pyiqa 计算 PSNR 的 BasicSR 适配函数。"""
    global _pyiqa_metric_models
    try:
        import pyiqa
    except ImportError:
        raise ImportError('请先安装 pyiqa: pip install pyiqa')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 使用 test_y_channel 作为 key 的一部分，区分是否只计算 Y 通道
    model_key = f'psnr_y{test_y_channel}'

    # 1. 懒加载
    if model_key not in _pyiqa_metric_models:
        _pyiqa_metric_models[model_key] = pyiqa.create_metric('psnr', test_y_channel=test_y_channel, as_loss=False).to(device)

    # 2. 边缘裁剪
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, :]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]

    # 3. 数据归一化 [0, 255] -> [0, 1]
    img = img.astype(np.float32) / 255.
    img2 = img2.astype(np.float32) / 255.

    # 4. 维度与通道转换 (BGR -> RGB, HWC -> CHW)
    if input_order == 'HWC':
        img = img[:, :, [2, 1, 0]]
        img2 = img2[:, :, [2, 1, 0]]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
    else:
        img = torch.from_numpy(img[[2, 1, 0], :, :])
        img2 = torch.from_numpy(img2[[2, 1, 0], :, :])

    # 5. 增加 Batch 维度并放入 GPU
    img = img.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)

    # 6. 计算指标
    with torch.no_grad():
        score = _pyiqa_metric_models[model_key](img, img2)

    return score.item()


@METRIC_REGISTRY.register()
def calculate_pyiqa_ssim(img, img2, crop_border=0, input_order='HWC', **kwargs):
    """使用 pyiqa 计算 彩色SSIM (SSIMC) 的 BasicSR 适配函数。"""
    global _pyiqa_metric_models
    try:
        import pyiqa
    except ImportError:
        raise ImportError('请先安装 pyiqa: pip install pyiqa')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_key = 'ssimc'

    # 1. 懒加载
    if model_key not in _pyiqa_metric_models:
        _pyiqa_metric_models[model_key] = pyiqa.create_metric('ssimc', as_loss=False).to(device)

    # 2. 边缘裁剪
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, :]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]

    # 3. 数据归一化
    img = img.astype(np.float32) / 255.
    img2 = img2.astype(np.float32) / 255.

    # 4. 维度与通道转换
    if input_order == 'HWC':
        img = img[:, :, [2, 1, 0]]
        img2 = img2[:, :, [2, 1, 0]]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
    else:
        img = torch.from_numpy(img[[2, 1, 0], :, :])
        img2 = torch.from_numpy(img2[[2, 1, 0], :, :])

    # 5. 增加 Batch 维度并放入 GPU
    img = img.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)

    # 6. 计算指标
    with torch.no_grad():
        score = _pyiqa_metric_models[model_key](img, img2)

    return score.item()


@METRIC_REGISTRY.register()
def calculate_pyiqa_niqe(img, crop_border=0, input_order='HWC', **kwargs):
    """使用 pyiqa 计算 NIQE 的 BasicSR 适配函数 (无参考指标，不传 img2)。"""
    global _pyiqa_metric_models
    try:
        import pyiqa
    except ImportError:
        raise ImportError('请先安装 pyiqa: pip install pyiqa')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_key = 'niqe'

    # 1. 懒加载
    if model_key not in _pyiqa_metric_models:
        _pyiqa_metric_models[model_key] = pyiqa.create_metric('niqe', as_loss=False).to(device)

    # 2. 边缘裁剪
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, :]

    # 3. 数据归一化
    img = img.astype(np.float32) / 255.

    # 4. 维度与通道转换
    if input_order == 'HWC':
        img = img[:, :, [2, 1, 0]]
        img = torch.from_numpy(img).permute(2, 0, 1)
    else:
        img = torch.from_numpy(img[[2, 1, 0], :, :])

    # 5. 增加 Batch 维度并放入 GPU
    img = img.unsqueeze(0).to(device)

    # 6. 计算指标 (只有 img)
    with torch.no_grad():
        score = _pyiqa_metric_models[model_key](img)

    return score.item()
