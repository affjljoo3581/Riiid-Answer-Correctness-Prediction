import argparse

from eil_riiid.prediction.submission import predict_riiid_correctness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pretrain EiL model")

    group = parser.add_argument_group("Dataset and vocabulary")
    group.add_argument("--vocab_path", required=True)
    group.add_argument("--context_path", required=True)

    group = parser.add_argument_group("Model configurations")
    group.add_argument("--model_path", default="eil-riiid.pth")

    group.add_argument("--seq_len", default=128, type=int)
    group.add_argument("--max_lag_time", default=1000, type=float)
    group.add_argument("--max_pqet", default=1000, type=float)

    group.add_argument("--num_layers", default=6, type=int)
    group.add_argument("--num_heads", default=8, type=int)
    group.add_argument("--hidden_dims", default=512, type=int)
    group.add_argument("--bottleneck", default=4, type=int)

    group = parser.add_argument_group("Extensions")
    group.add_argument("--use_fp16", action="store_true")

    predict_riiid_correctness(parser.parse_args())
