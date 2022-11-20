from pathlib import Path
from run_predictor import *
from parameters import parse_args
import torch.multiprocessing as mp

SELECT_DATA="dblp"

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12360'
    args = parse_args()
    if SELECT_DATA=="dblp":
        args.train_data_path = '/home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/dblp_graph_data/sample_train.tsv'
        args.valid_data_path='/home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/dblp_graph_data/valid.tsv'
        args.test_data_path='/home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/dblp_graph_data/test.tsv'
        # args.test_data_path='/home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/dblp_twoorder_neighbors/test_50.tsv'
        args.load_ckpt_name = "./model/best/dblp/gat-best.pt"
        args.block_size = 32
        args.schedule_step = (15000) * args.epochs
        args.save_steps = 10 ** 4
        args.log_steps = 100
        args.warmup_step = 10 ** 3
        args.train_batch_size = 64
        args.valid_batch_size = 60
        args.test_batch_size = 60
    elif SELECT_DATA=="wiki":
        args.train_data_path = '/home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/wikidata5m_without_overlap/train.tsv'
        args.valid_data_path='/home/jingyao/projects/CDSM/NeighborSelectionTopoGramFiles/wikidata5m_without_overlap/valid.tsv'
        args.test_data_path='/home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/wikidata5m_without_overlap/test.tsv'
        # args.test_data_path='/home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/wikidata5m_twoorder_neighbors/test_50.tsv'
        args.block_size = 64
        args.schedule_step = (30000) * args.epochs
        args.save_steps = 10 ** 4
        args.log_steps = 100
        args.warmup_step = 10 ** 3
        args.train_batch_size = 64
        args.valid_batch_size = 60
        args.test_batch_size = 60
    elif SELECT_DATA=="product":
        args.train_data_path='/home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/Graph_1206_small/sample_Train_shuf.tsv'
        args.valid_data_path='/home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/Graph_1206_small/Valid_shuf.tsv'
        args.test_data_path='./home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/Graph_1206_small/Test_shuf.tsv'
        args.block_size = 32
        args.schedule_step = (30000) * args.epochs
        args.save_steps = 5*10 ** 4
        args.log_steps = 100
        args.warmup_step = 10 ** 3
        args.train_batch_size = 64
        args.valid_batch_size = 300
        args.test_batch_size = 300

    args.warmup_lr=True
    args.savename='counterpart'
    args.world_size=1
    args.neighbor_num=50
    args.total_neighbor_num=50
    args.self_mask=False
    args.epochs=3
    args.mlm_loss=False
    args.return_last_station_emb=False
    args.mapping_graph=False
    args.model_type='predictor'
    args.model_name_or_path="/home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/Turing/roberta-base.bin"
    args.config_name="/home/jingyao/projects/CDSM/NeighborSelection/TopoGramFiles/Turing/roberta-base-config.json"
    args.pretrain_lr=1e-5
    args.fp16=True
    args.neighbor_mask=False
    args.mode='test'
    args.condition='key'
    args.basic='roberta'
    args.select='counterpart'
    args.aggregation='max'
    args.predictor_type='counterpart'
    args.selector_task='similarity'
    args.select_num=10
    args.count_threshold = None
    args.threshold = None
    args.stop_condition = 'fixed_num'  # ['query', 'matching_threshold', 'softmax_threshold', 'count_threshold', 'fixed_num']

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    print("running with configuration: ", args)

    print('-------------predicting--------------')
    if args.selector_task == 'similarity':
        args.selector_ckpt = os.path.join(args.model_dir, 'selector-'+SELECT_DATA+'-onestep-similarity-best.pt')
    elif args.selector_task == 'matching':
        args.selector_ckpt = os.path.join(args.model_dir, 'selector-'+SELECT_DATA+'-onestep-matching-best.pt')

    if args.predictor_type == 'counterpart':
        args.predictor_ckpt = os.path.join(args.model_dir, 'counterpart-dot-'+SELECT_DATA+'-twoorder-best.pt')
    elif args.predictor_type == 'gat':
        args.predictor_ckpt = os.path.join(args.model_dir, 'gat-'+SELECT_DATA+'-best.pt')
    elif args.predictor_type == 'graphsage':
        args.predictor_ckpt = os.path.join(args.model_dir, 'graphsage-'+args.aggregation+'-'+SELECT_DATA+'-best.pt')
    
    if args.world_size > 1:
        mp.freeze_support()
        mgr = mp.Manager()
        end = mgr.Value('b', False)
        global_prefetch_step = mgr.list([0] * args.world_size)
        mp.spawn(predict, args = (args,global_prefetch_step, end, args.mode),
                nprocs=args.world_size, join=True)
    else:
        mgr = mp.Manager()
        end = mgr.Value('b', False)
        prefetch_step = mgr.list([0] * args.world_size)
        predict_single_process(args, args.mode)