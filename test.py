import os
import torch
import json
from models.our.our_model import ourModel
from train import eval
import argparse
from utils.logger import get_logger
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader
from dataset import *

class Opt:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test MDPP Model")
    parser.add_argument('--labelcount', type=int, default=2,
                        help="Number of data categories (2, 3, or 5).")
    parser.add_argument('--track_option', type=str, required=True,
                        help="Track1 or Track2")
    parser.add_argument('--feature_max_len', type=int, required=True,
                        help="Max length of feature.")
    parser.add_argument('--data_rootpath', type=str, required=True,
                        help="Root path to the program dataset")
    parser.add_argument('--train_model', type=str, required=True,
                        help="Path to the training model")

    parser.add_argument('--test_json', type=str, required=False, 
                        help="File name of the testing JSON file")
    parser.add_argument('--personalized_features_file', type=str,
                        help="File name of the personalized features file")

    parser.add_argument('--audiofeature_method', type=str, default='wav2vec',
                        choices=['mfccs', 'opensmile', 'wav2vec'],
                        help="Method for extracting audio features.")
    parser.add_argument('--videofeature_method', type=str, default='openface',
                        choices=['openface', 'resnet', 'densenet'],
                        help="Method for extracting video features.")
    parser.add_argument('--splitwindow_time', type=str, default='1s',
                        help="Time window for splitted features. e.g. '1s' or '5s'")

    parser.add_argument('--batch_size', type=int, default=24,
                        help="Batch size for testing")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device to test the model on, e.g. 'cuda' or 'cpu'")

    args = parser.parse_args()

    args.test_json = os.path.join(args.data_rootpath, 'Testing', 'labels', 'Testing_files.json')
    args.personalized_features_file = os.path.join(args.data_rootpath, 'Testing', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')


    config = load_config('config.json')
    opt = Opt(config)

    # Modify individual dynamic parameters in opt according to task category
    opt.emo_output_dim = args.labelcount
    opt.feature_max_len = args.feature_max_len
    opt.lr = args.lr

    # Splice out feature folder paths according to incoming audio and video feature types
    audio_path = os.path.join(args.data_rootpath, 'Testing', f"{args.splitwindow_time}", 'Audio', f"{args.audiofeature_method}") + '/'
    video_path = os.path.join(args.data_rootpath, 'Testing', f"{args.splitwindow_time}", 'Visual', f"{args.videofeature_method}") + '/'

    # Obtain input_dim_a, input_dim_v
    for filename in os.listdir(audio_path):
        if filename.endswith('.npy'):
            opt.input_dim_a = np.load(audio_path + filename).shape[1]
            break

    for filename in os.listdir(video_path):
        if filename.endswith('.npy'):
            opt.input_dim_v = np.load(video_path + filename).shape[1]            
            break

    opt.name = f'{args.splitwindow_time}_{args.labelcount}labels_{args.audiofeature_method}+{args.videofeature_method}'
    logger_path = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    logger = get_logger(logger_path, 'result')

    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
    best_model_name = f"best_model_{cur_time}.pth"

    logger.info(f"splitwindow_time={args.splitwindow_time}, audiofeature_method={args.audiofeature_method}, "
                f"videofeature_method={args.videofeature_method}")
    logger.info(f"batch_size={args.batch_size}, , "
                f"labels={opt.emo_output_dim}, feature_max_len={opt.feature_max_len}")


    model = ourModel(opt)
    model.load_state_dict(torch.load(args.train_model, map_location=torch.device(args.device)))
    model.to(args.device)
    test_data = json.load(open(args.test_json, 'r'))
    test_loader = DataLoader(
        AudioVisualDataset(test_data, args.labelcount, args.personalized_features_file, opt.feature_max_len,
                           batch_size=args.batch_size,
                           audio_path=audio_path, video_path=video_path,isTest=True), batch_size=args.batch_size, shuffle=False)
    logger.info('The number of testing samples = %d' % len(test_loader.dataset))

    # testing
    _, pred, *_ = eval(model, test_loader, args.device)

    filenames = [item["audio_feature_path"] for item in test_data if "audio_feature_path" in item]
    IDs = [path[:path.find('.')] for path in filenames]

    if args.labelcount==2:
        label="bin"
    elif args.labelcount==3:
        label="tri"
    elif args.labelcount==5:
        label="pen"
    

    # output results to CSV
    pred_col_name = f"{args.splitwindow_time}_{label}"

    result_dir = f"./answer_{args.track_option}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    csv_file = f"{result_dir}/submission.csv"

    # Get the order of the IDs in the test data to ensure consistency
    test_ids = [np.int64(item["audio_feature_path"].split('_')[0])for item in test_data]
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["ID"])

    pred = np.array(pred)

    # 安全检查
    if len(pred) != len(test_data):
        logger.error(f"Prediction length {len(pred)} does not match test data length {len(test_data)}")
        raise ValueError("Mismatch between predictions and test data")

    # 构建带 ID 和预测结果的 dataframe
    result_df = pd.DataFrame({
        "ID": [item["id"] for item in test_data],
        pred_col_name: pred
    })

    # 按顺序输出，ID 可重复
    result_df.to_csv(csv_file, index=False)
    logger.info(f"Testing complete. Results saved to: {csv_file}. Shape={result_df.shape}")


    logger.info(f"Testing complete. Results saved to: {csv_file}.")
