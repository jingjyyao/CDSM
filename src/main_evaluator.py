from pathlib import Path
from run_evaluator import *
from parameters import parse_args
import torch.multiprocessing as mp

SELECT_DATA="dblp"
NEIGHBOR_NUM=5
AGG='mean'  # aggregation method for graphsage, max/mean/gat
EVALUATOR_TYPE='counterpart' # counterpart/graphsage/gat

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    args = parse_args()
    if SELECT_DATA=="dblp":
        args.train_data_path = '../../data/dblp_twoorder_neighbors/sample_train.tsv'
        args.valid_data_path='../../data/dblp_twoorder_neighbors/valid_50.tsv'
        # args.test_data_path='../../data/dblp_twoorder_data/test.tsv'
        args.test_data_path='../../data/dblp_twoorder_neighbors/test_popularity.tsv'
        args.load_ckpt_name = "../../checkpoint/counterpart-dblp-best.pt"
        args.block_size = 32
        args.schedule_step = (15000) * args.epochs
        args.save_steps = 10 ** 4
        args.log_steps = 20
        args.warmup_step = 10 ** 3
        args.train_batch_size = 30
        args.valid_batch_size = 30
        args.test_batch_size = 30
    elif SELECT_DATA=="wiki":
        args.train_data_path = '../../data/wikidata5m_oneorder_neighbors/sample_train.tsv'
        args.valid_data_path='../../data/wikidata5m_oneorder_neighbors/valid.tsv'
        # args.test_data_path='../../data/wikidata5m_without_overlap/test.tsv'
        args.test_data_path='../../data/wikidata5m_oneorder_neighbors/test.tsv'
        args.block_size = 64
        args.schedule_step = (30000) * args.epochs
        args.save_steps = 10 ** 4
        args.log_steps = 2000
        args.warmup_step = 10 ** 3
        args.train_batch_size = 30
        args.valid_batch_size = 30
        args.test_batch_size = 30
    elif SELECT_DATA=="product":
        args.train_data_path='../../data/product_oneorder_neighbors/sample_Train_shuf.tsv'
        args.valid_data_path='../../data/product_oneorder_neighbors/Valid_shuf.tsv'
        args.test_data_path='.../../data/product_oneorder_neighbors/Test_shuf.tsv'
        args.block_size = 32
        args.schedule_step = (30000) * args.epochs
        args.save_steps = 5*10 ** 4
        args.log_steps = 20
        args.warmup_step = 10 ** 3
        args.train_batch_size = 100
        args.valid_batch_size = 100
        args.test_batch_size = 100

    args.model_dir='../../checkpoint'
    args.enable_gpu=True

    args.warmup_lr=True
    args.savename='evaluator'
    args.world_size=1
    args.neighbor_type=2
    args.neighbor_num=NEIGHBOR_NUM
    args.self_mask=False
    args.epochs=3
    args.mlm_loss=False
    args.return_last_station_emb=False
    args.mapping_graph=False
    args.model_type='evaluator'
    args.model_name_or_path="../../data/Turing/base-uncased.bin"
    args.config_name="../../data/Turing/unilm2-base-uncased-config.json"
    args.pretrain_lr=5e-6
    args.fp16=True
    args.neighbor_mask=False
    args.aggregation=AGG
    args.mode='test'
    args.evaluator_type=EVALUATOR_TYPE

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    print("running with configuration: ", args)

    print('-------------evaluate {}--------------'.format(args.mode))
    args.evaluator_ckpt = "../../checkpoint/{}-{}-best.pt".format(EVALUATOR_TYPE, SELECT_DATA)
    if args.evaluator_type == 'graphsage':
        args.evaluator_ckpt = "../../checkpoint/graphsage-{}-{}-best.pt".format(AGG, SELECT_DATA)
    if args.world_size > 1:
        mp.freeze_support()
        mgr = mp.Manager()
        end = mgr.Value('b', False)
        global_prefetch_step = mgr.list([0] * args.world_size)

        mp.spawn(evaluate, args = (args,global_prefetch_step, end, args.mode), nprocs=args.world_size, join=True)
    else:
        mgr = mp.Manager()
        end = mgr.Value('b', False)
        prefetch_step = mgr.list([0] * args.world_size)
        evaluate_single_process(args, args.mode)