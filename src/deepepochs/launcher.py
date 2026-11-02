"""从普通Python入口启动单机训练进程。"""

import os
import signal
import threading
from uuid import uuid4

import torch
import torch.distributed as dist


def launch(main, num_processes=None, args=()):
    """运行训练函数，并返回主进程的结果。

    main是模块顶层函数，通过args传入位置参数。模型、优化器、DataLoader和Trainer均在
    函数内部创建。启动调用放在if __name__ == '__main__'中，支持Python
    脚本和IDE运行。多进程返回值应为可序列化的CPU数据，例如指标字典。

    Args:
        main: 每个训练进程执行的函数。
        num_processes: 进程数；None使用可见CUDA设备数量，没有CUDA时为1。
            显式进程数也可用于CPU多进程，具体训练设备由main中的Trainer选择。
        args: 传给main的位置参数元组，例如args=(config,)。多进程参数应为
            普通可序列化数据；模型等训练对象在main内部创建。

    单进程直接调用main；torchrun进程和已初始化进程组也直接调用main，
    由现有启动方管理进程。多进程启动使用spawn，训练函数必须可被子进程导入。
    """
    if not callable(main):
        raise TypeError('main必须是可调用的训练函数。')
    if num_processes is not None and (
            isinstance(num_processes, bool) or not isinstance(num_processes, int) or num_processes < 1):
        raise ValueError('num_processes必须为正整数或None。')

    # 外部启动器已分配rank；进程组的初始化和所有权仍由Trainer或调用方管理。
    if 'RANK' in os.environ or (dist.is_available() and dist.is_initialized()):
        return main(*args)
    if num_processes is None:
        num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if num_processes == 1:
        return main(*args)

    # 仅在需要创建worker时加载启动器，单进程导入保持轻量。
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=num_processes,
        run_id=uuid4().hex,
        rdzv_backend='c10d',
        rdzv_endpoint='127.0.0.1:0',
        local_addr='127.0.0.1',
        start_method='spawn',
        max_restarts=0,
        monitor_interval=0.1,
    )
    # 每个worker独立使用CPU线程，避免多进程同时占满线程池；显式配置优先。
    # 环境只在启动期间影响新进程，结束或异常退出时恢复调用方环境。
    # Elastic在启动期间接管退出信号；返回调用方后恢复其原有处理方式。
    # signal只能在主线程设置，其他线程的行为交给原生启动器处理。
    signal_handlers = {}
    if threading.current_thread() is threading.main_thread():
        signals = (signal.SIGINT, signal.SIGTERM)
        if os.name != 'nt':
            signals += (signal.SIGHUP, signal.SIGQUIT)
        signal_handlers = {signum: signal.getsignal(signum) for signum in signals}

    added_omp_threads = 'OMP_NUM_THREADS' not in os.environ
    if added_omp_threads:
        os.environ['OMP_NUM_THREADS'] = '1'
    try:
        results = elastic_launch(config, main)(*args)
        return results[0]
    finally:
        if added_omp_threads:
            os.environ.pop('OMP_NUM_THREADS', None)
        for signum, handler in signal_handlers.items():
            signal.signal(signum, handler)
