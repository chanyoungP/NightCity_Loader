import argparse


def obtain_search_args():
    parser = argparse.ArgumentParser(description="NightCity Dataloader")
    parser.add_argument('--dataset', type=str, default='nightcity',
                        choices=['video','cityscapes','nightcity'],
                        help='dataset name (default: cityscapes)')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=320,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=321,
                        help='crop image size')
    parser.add_argument('--resize', type=int, default=512,
                        help='resize image size')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    #dist sampling
    parser.add_argument('--dist', default=False, #window env different
                        type=bool, help='whether use distributed sampling ')

    args = parser.parse_args()
    return args
