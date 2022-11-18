from dataloaders import make_data_loader
from mypath import Path     # 파일 없음 구현 X
from args import obtain_search_args

args = obtain_search_args()


kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
train_loader,val_loader,test_loader, num_classes = make_data_loader(args, **kwargs)