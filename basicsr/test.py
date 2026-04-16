import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):#输入根目录
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)#字典

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)#创建文件夹或文件
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")#日志
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)#初始化日志系统
    logger.info(get_env_info())#会收集环境信息，写入日志
    logger.info(dict2str(opt))#把配置字典 opt 转换成可读的字符串（键值对格式），写出完整配置

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):#数据集 每个
        test_set = build_dataset(dataset_opt)#实例
        test_loader = build_dataloader(#迭代器
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)#模型

    for test_loader in test_loaders:#测试 每组 是迭代器 不是tensor
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        #print(test_loader.shape)
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])#运行


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))#命令行输入文件
    test_pipeline(root_path)
