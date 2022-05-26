import os
import sys
import time
import config

import ray
# ray.init(address='ray://10.0.0.1:10001')
ray.init(address='auto', _redis_password='<insert password as reported by ray>', _node_ip_address='10.0.0.1')


@ray.remote(num_gpus=0.5)
def check_remote():
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print(sys.executable)
    print(os.getcwd())
    time.sleep(1)


@ray.remote(num_gpus=0.5)
def train_model(model_name, fold):
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print(sys.executable, model_name, fold)
    time.sleep(1)
    os.system(f'python train_cls.py train --fold {fold} {model_name}')


def main():
    all_models = [
        m[0] for m in config.MODELS
    ]
    folds = [0, 1, 2, 3]

    requests = []
    for fold in folds:
        for model in all_models:
            # print(f'python train_cls.py train --fold {fold} {model}')
            requests.append(train_model.remote(model, fold))

    ray.get(requests)

    # to test ray setup
    # ray.get([
    #     check_remote.remote()
    #     for _ in range(16)
    # ])


if __name__ == '__main__':
    main()
