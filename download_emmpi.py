"""
Full data instructions at https://huggingface.co/datasets/blutjens/em-mpi
"""
import argparse
from huggingface_hub import snapshot_download # Requires pip install huggingface_hub

def get_args():
    parser = argparse.ArgumentParser(description='Download Em-MPI dataset from huggingface')
    parser.add_argument('--data_dir', type=str, default='../em-mpi-data', help='Data variable that fit')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    snapshot_download(repo_id='blutjens/em-mpi',
        repo_type='dataset', 
        local_dir=args.data_dir # Local path to data 
    )