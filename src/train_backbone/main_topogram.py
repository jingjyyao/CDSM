from run_topogram import *
from pathlib import Path
from parameters import parse_args
import torch.multiprocessing as mp

SELECT_DATA="product"
NEIGHBOR_NUM=5
SELECT_NUM=5
AGG='mean'
MODE='test'

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    args = parse_args()
    cont=False
    if SELECT_DATA=="dblp":
        args.train_data_path = '/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/dblp_graph_data/sample_train.tsv'
        args.valid_data_path='/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/dblp_graph_data/valid.tsv'
        # args.test_data_path='/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/dblp_graph_data/test.tsv'
        # args.test_data_path='/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/dblp_twoorder_neighbors/test_50.tsv'
        args.test_data_path='/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/dblp_twoorder_neighbors/test_5.tsv'
        args.block_size = 32
        args.schedule_step = (13000) * args.epochs
        args.save_steps = 10 ** 4
        args.log_steps = 100
        args.warmup_step = 10 ** 3
        args.train_batch_size = 30
        args.valid_batch_size = 30
        args.test_batch_size = 2
    elif SELECT_DATA=="wiki":
        args.train_data_path = '/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/wikidata5m_without_overlap/local_train.tsv'
        args.valid_data_path='/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/wikidata5m_without_overlap/valid.tsv'
        args.test_data_path='/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/wikidata5m_without_overlap/test.tsv'
        args.block_size = 64
        args.schedule_step = (17000) * args.epochs
        args.save_steps = 10 ** 4
        args.log_steps = 100
        args.warmup_step = 10 ** 3
        args.train_batch_size = 20
        args.valid_batch_size = 50
        args.test_batch_size = 50
    elif SELECT_DATA=="product":
        args.train_data_path='/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/product_oneorder_neighbors/TrainData_5_shuf.tsv'
        args.valid_data_path='/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/product_oneorder_neighbors/ValidData_5_shuf.tsv'
        # args.test_data_path='/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/product_oneorder_neighbors/TestData_50_shuf.tsv'
        args.test_data_path='/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/product_oneorder_neighbors/TestData_5_shuf.tsv'
        args.block_size = 32
        args.schedule_step = (10**5) * args.epochs
        args.save_steps = 5*10 ** 4
        args.log_steps = 100
        args.warmup_step = 10 ** 4
        args.train_batch_size = 30
        args.valid_batch_size = 300
        args.test_batch_size = 300 

    # args.model_dir='./model'
    # args.enable_gpu=True

    args.warmup_lr=False
    args.savename='topogram'
    args.world_size=1
    args.neighbor_type=2
    args.neighbor_num=NEIGHBOR_NUM
    args.self_mask=False
    args.epochs=3
    args.mlm_loss=False
    args.return_last_station_emb=False
    args.mapping_graph=False
    args.model_type='topogram'
    args.model_name_or_path="/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/Turing/base-uncased.bin"
    args.config_name="/home/v-jingyao/projects/NeighborSelection/TopoGramFiles/Turing/unilm2-base-uncased-config.json"
    args.pretrain_lr=5e-6
    args.fp16=True
    args.neighbor_mask=False
    args.aggregation=AGG
    args.mode=MODE

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    print("running with configuration: ", args)

    if 'train' in args.mode:
        print('-----------train------------')
        if args.world_size > 1:
            mp.freeze_support()
            mgr = mp.Manager()
            end = mgr.Value('b', False)
            global_prefetch_step = mgr.list([0] * args.world_size)

            mp.spawn(train, args = (args,global_prefetch_step, end, cont),nprocs=args.world_size, join=True)
        else:
            mgr = mp.Manager()
            end = mgr.Value('b', False)
            prefetch_step = mgr.list([0] * args.world_size)
            train(0,args,prefetch_step,end,cont)


    if 'test' in args.mode:
        print('-------------test--------------')
        args.load_ckpt_name="/home/v-jingyao/projects/NeighborSelection-Final/checkpoint/topogram-{}-best.pt".format(SELECT_DATA)
        if args.world_size > 1:
            mp.freeze_support()
            mgr = mp.Manager()
            end = mgr.Value('b', False)
            global_prefetch_step = mgr.list([0] * args.world_size)

            mp.spawn(test, args = (args,global_prefetch_step, end, cont),nprocs=args.world_size, join=True)
        else:
            mgr = mp.Manager()
            end = mgr.Value('b', False)
            prefetch_step = mgr.list([0] * args.world_size)
            test(0,args,prefetch_step,end)

