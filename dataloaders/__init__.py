from dataloaders.datasets import cityscapes,nightcity
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import os



def make_data_loader(args, **kwargs):
    if args.dist:
        print("Using Distributed Sampler :::: trainA and trainB for bilevel optimization")
        if args.dataset == 'cityscapes':
            train_set1, train_set2 = cityscapes.twoTrainSeg(args)
            num_class = train_set1.NUM_CLASSES
            sampler1 = torch.utils.data.distributed.DistributedSampler(train_set1)
            sampler2 = torch.utils.data.distributed.DistributedSampler(train_set2)
            train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=False, sampler=sampler1,
                                       **kwargs)
            train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=False, sampler=sampler2,
                                       **kwargs)
            val_set = cityscapes.CityscapesSegmentation(args, split='val')
            test_set = cityscapes.CityscapesSegmentation(args, split='test')
            sampler3 = torch.utils.data.distributed.DistributedSampler(val_set)
            sampler4 = torch.utils.data.distributed.DistributedSampler(test_set)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, sampler=sampler3, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, sampler=sampler4, **kwargs)

            return train_loader1, train_loader2, val_loader, test_loader, num_class

        elif args.dataset =='nightcity':
            train_set1, train_set2 = nightcity.twoTrainSeg(args)
            num_class = train_set1.NUM_CLASSES
            sampler1 = torch.utils.data.distributed.DistributedSampler(train_set1)
            sampler2 = torch.utils.data.distributed.DistributedSampler(train_set2)
            train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=False, sampler=sampler1,
                                       **kwargs)
            train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=False, sampler=sampler2,
                                       **kwargs)
            val_set = nightcity.NightCitySegmentation(args, split='val')
            # test_set = nightcity.NightCitySegmentation(args, split='test')  # NightCity doesn't have test set
            sampler3 = torch.utils.data.distributed.DistributedSampler(val_set)
            # sampler4 = torch.utils.data.distributed.DistributedSampler(test_set)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, sampler=sampler3, **kwargs)
            # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, sampler=sampler4, **kwargs)

            return train_loader1, train_loader2, val_loader,num_class # NightCity dataset NO test set

        else:
            raise NotImplementedError
    else:
        if args.dataset == 'cityscapes':
            train_set = cityscapes.CityscapesSegmentation(args, split='retrain')
            val_set = cityscapes.CityscapesSegmentation(args, split='val')
            test_set = cityscapes.CityscapesSegmentation(args, split='test')

            num_class = train_set.NUM_CLASSES

            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

            return train_loader, val_loader, test_loader, num_class

        elif args.dataset == 'nightcity':
            train_set = nightcity.NightCitySegmentation(args, split='retrain')
            val_set = nightcity.NightCitySegmentation(args, split='val')
            # test_set = nightcity.NightCitySegmentation(args, split='test')  # NightCity doesn't have test set

            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

            return train_loader, val_loader, num_class