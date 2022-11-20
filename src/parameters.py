import argparse
import utils
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default="train_test",
                        choices=['train', 'test', 'train_test'])
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="/data/t-shxiao/test/cosmos-speedup-turing/rec_bert/data/train",
    )
    parser.add_argument("--train_batch_size", type=int, default=30)

    parser.add_argument(
        "--valid_data_path",
        type=str,
        default="/data/t-shxiao/test/cosmos-speedup-turing/rec_bert/data/valid"
    )
    parser.add_argument("--valid_batch_size", type=int, default=300)

    parser.add_argument(
        "--test_data_path",
        type=str,
        default="/data/t-shxiao/test/cosmos-speedup-turing/rec_bert/data/test"
    )
    parser.add_argument("--test_batch_size", type=int, default=300)


    parser.add_argument("--model_dir", type=str, default='../checkpoint')  # path to save
    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)
    # parser.add_argument("--num_workers", type=int, default=4)

    # parser.add_argument("--freeze_bert", type=utils.str2bool, default=False)
    parser.add_argument("--warmup_lr", type=utils.str2bool, default=True)
    parser.add_argument("--savename", type=str, default='model')
    parser.add_argument("--warmup_step", type=int, default=5000)
    parser.add_argument("--schedule_step", type=int, default=25000)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--neighbor_type", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=32)  # max length of inputted sequence
    parser.add_argument("--neighbor_num", type=int, default=5)  # num of neighbors used in training
    parser.add_argument("--total_neighbor_num", type=int, default=50)  # total num of neighbors in datasets
    parser.add_argument("--self_mask", type=utils.str2bool, default=False)

    # model configuration
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--condition", type=str,default="key",
                        choices=['key', 'key_neighbor', 'query_key_neighbor'])
    parser.add_argument("--select", type=str, default=None, choices=['random', 'counterpart'])
    parser.add_argument("--select_num", type=int, default=5)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--attention_type", type=str, default='dot_product', choices=['dot_product', 'additive'])
    parser.add_argument("--selector_ckpt", type=str)
    parser.add_argument("--predictor_ckpt", type=str)
    parser.add_argument("--negative_num", type=int, default=5)
    parser.add_argument("--positive_num", type=int, default=5)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--selector_task", type=str, default="similarity", choices=['similarity', 'matching'])

    # model training
    parser.add_argument("--epochs", type=int, default=1)
    # parser.add_argument("--num_words_title", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--log_steps", type=int, default=100)
    # parser.add_argument("--max_steps_per_epoch", type=int, default=1000000)
    parser.add_argument("--mlm_loss", type=utils.str2bool, default=True)
    parser.add_argument("--return_last_station_emb", type=utils.str2bool, default=False)
    parser.add_argument("--mapping_graph",type=utils.str2bool,default=False)

    # turing
    parser.add_argument("--model_type", default="retrive", type=str)
    parser.add_argument("--model_name_or_path", default="../Turing/unilm2-base-uncased.bin", type=str,
                        help="Path to pre-trained model or shortcut name. ")
    # parser.add_argument("--model_name_or_path", default="model/retrive-epoch-2.pt", type=str,
    #                     help="Path to pre-trained model or shortcut name. ")
    parser.add_argument("--config_name", default="../Turing/unilm2-base-uncased-config.json", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    # parser.add_argument("--tokenizer_name", default="../Turing/unilm2-base-uncased-vocab.txt", type=str,
    #                     help="Pretrained tokenizer name or path if not the same as model_name")

    # parser.add_argument(
    #     "--finetune_blocks",
    #     type=int,
    #     nargs='+',
    #     default=[],
    #     choices=list(range(12)))

    parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default = 'model/TopoGram_NeighborMask/topogram-epoch-1.pt',
        help="choose which ckpt to load and test"
    )

    parser.add_argument(
        "--load_selector_name",
        type=str,
        help="choose selector ckpt to load and test"
    )

    # lr schedule
    # parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--pretrain_lr", type=float, default=0.00001)

    # parser.add_argument("--bert_layer_hidden", type=int, default=None, choices=list(range(12)))

    # half float
    parser.add_argument("--fp16",type=utils.str2bool,default=False)
    parser.add_argument("--neighbor_mask",type=utils.str2bool,default=False)
    parser.add_argument("--aggregation",type=str,default='mean',choices=['mean','max','gat'])

    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
