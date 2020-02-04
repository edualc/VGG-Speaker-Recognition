from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

sys.path.append('../tool')
import toolkits
import utils as ut

import logging
logging.getLogger('tensorflow').disabled = True

import pdb
# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--data_path', default='/mnt/all1/voxceleb1/all', type=str)
# parser.add_argument('--data_path', default='/cluster/data/lehmacl1/datasets/raw/vox1', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
parser.add_argument('--overlap', default='', type=str)

global args
args = parser.parse_args()

def main():

    # gpu configuration
    toolkits.initialize_GPU(args)

    import model
    # ==================================
    #       Get Train/Val.
    # ==================================
    print('==> calculating test({}) data lists...'.format(args.test_type))

    if args.test_type == 'normal':
        verify_list = np.loadtxt('../meta/voxceleb1_veri_test.txt', str)
    elif args.test_type == 'hard':
        verify_list = np.loadtxt('../meta/voxceleb1_veri_test_hard.txt', str)
    elif args.test_type == 'extend':
        verify_list = np.loadtxt('../meta/voxceleb1_veri_test_extended.txt', str)
    else:
        raise IOError('==> unknown test type.')

    verify_lb = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(args.data_path, i[1]) for i in verify_list])
    list2 = np.array([os.path.join(args.data_path, i[2]) for i in verify_list])

    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list)

    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)

    # ==> load pre-trained model ???
    if args.resume:
        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
            result_path = set_result_path(args)
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    else:
        raise IOError('==> please type in the model to load')

    print('==> start testing.')

    def get_eval_lists():
        # return {
        #     'vox1-cleaned':          '../meta/voxceleb1_veri_test_fixed.txt'
        # }

        return {
            'vox1':                  '../meta/voxceleb1_veri_test.txt',
            'vox1-cleaned':          '../meta/voxceleb1_veri_test_fixed.txt',
            'vox1-E':                '../meta/voxceleb1_veri_test_extended.txt',
            'vox1-E-cleaned':        '../meta/voxceleb1_veri_test_extended_fixed.txt',
            'vox1-H':                '../meta/voxceleb1_veri_test_hard.txt',
            'vox1-H-cleaned':        '../meta/voxceleb1_veri_test_hard_fixed.txt'
        }

    def unique_utterances():
        utterances = set()
        eval_lists = get_eval_lists()

        for key in eval_lists.keys():
            for line in open(eval_lists[key], 'r'):
                label, file1, file2 = line.rstrip().split(' ')


                utterances.add(file1)
                utterances.add(file2)

        return utterances

    def extract_embeddings_for_eval_lists(sliding_window_shift=params['spec_len']//2, identifier=''):
        with h5py.File('../result/vgg_embeddings_' + identifier + '.h5', 'a') as f:
            already_extracted_labels = f['labels'][:]

        # import code; code.interact(local=dict(globals(), **locals()))
        unique_utterances_to_extract = np.array(list(unique_utterances()))
        num_unique = unique_utterances_to_extract.shape[0]

        unique_utterances_to_extract = np.setdiff1d(unique_utterances_to_extract, already_extracted_labels)

        print("Already Extracted: {}/{} ({} remaining)".format(already_extracted_labels.shape[0], num_unique, unique_utterances_to_extract.shape[0]))


        with h5py.File('../result/vgg_embeddings_' + identifier + '.h5', 'a') as f:
            for idx, utterance in enumerate(tqdm(unique_utterances_to_extract, ascii=True, desc='preparing spectrogram windows for predictions with sliding window shift ' + identifier)):
                spectrogram_labels = list()

                specs = ut.load_data(args.data_path + '/' + utterance, win_length=params['win_length'], sr=params['sampling_rate'],
                                     hop_length=params['hop_length'], n_fft=params['nfft'],
                                     spec_len=params['spec_len'], mode='eval')

                if specs.shape[1] < params['spec_len'] + 4 * sliding_window_shift:
                    num_repeats = ((params['spec_len'] + 4 * sliding_window_shift) // specs.shape[1]) + 1
                    specs = np.tile(spect, (1, num_repeats))

                offset = 0
                sample_spects = list()

                while offset < specs.shape[1] - params['spec_len']:
                    sample_spects.append(specs[:, offset:offset + params['spec_len']])
                    spectrogram_labels.append(utterance)
                    offset += sliding_window_shift

                specs = np.expand_dims(np.asarray(sample_spects), -1)
                embeddings = network_eval.predict(specs)

                spectrogram_labels = np.string_(spectrogram_labels)

            
                if 'labels' not in f.keys():
                    f.create_dataset('labels', data=spectrogram_labels, maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
                else:
                    f['labels'].resize((f['labels'].shape[0] + spectrogram_labels.shape[0]), axis=0)
                    f['labels'][-spectrogram_labels.shape[0]:] = spectrogram_labels

                if 'embeddings' not in f.keys():
                    f.create_dataset('embeddings', data=embeddings, maxshape=(None, embeddings.shape[1]))
                else:
                    f['embeddings'].resize((f['embeddings'].shape[0] + embeddings.shape[0]), axis=0)
                    f['embeddings'][-embeddings.shape[0]:] = embeddings

    if args.overlap == '50_percent_shift':
        extract_embeddings_for_eval_lists(params['spec_len'] // 2, '50_percent_shift')
    
    if args.overlap == '75_percent_shift':
        extract_embeddings_for_eval_lists(3 * (params['spec_len'] // 4), '75_percent_shift')
    
    if args.overlap == '25_percent_shift':
        extract_embeddings_for_eval_lists(params['spec_len'] // 4, '25_percent_shift')

    sys.exit()

    # # The feature extraction process has to be done sample-by-sample,
    # # because each sample is of different lengths.
    # total_length = len(unique_list)
    # feats, scores, labels = [], [], []
    # for c, ID in enumerate(unique_list):
    #     if c % 50 == 0: print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
    #     specs = ut.load_data(ID, win_length=params['win_length'], sr=params['sampling_rate'],
    #                          hop_length=params['hop_length'], n_fft=params['nfft'],
    #                          spec_len=params['spec_len'], mode='eval')
    #     specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    
    #     v = network_eval.predict(specs)
    #     feats += [v]
    
    # feats = np.array(feats)

    # # ==> compute the pair-wise similarity.
    # for c, (p1, p2) in enumerate(zip(list1, list2)):
    #     ind1 = np.where(unique_list == p1)[0][0]
    #     ind2 = np.where(unique_list == p2)[0][0]

    #     v1 = feats[ind1, 0]
    #     v2 = feats[ind2, 0]

    #     scores += [np.sum(v1*v2)]
    #     labels += [verify_lb[c]]
    #     print('scores : {}, gt : {}'.format(scores[-1], verify_lb[c]))

    # scores = np.array(scores)
    # labels = np.array(labels)

    # np.save(os.path.join(result_path, 'prediction_scores.npy'), scores)
    # np.save(os.path.join(result_path, 'groundtruth_labels.npy'), labels)

    # eer, thresh = toolkits.calculate_eer(labels, scores)
    # print('==> model : {}, EER: {}'.format(args.resume, eer))


def set_result_path(args):
    model_path = args.resume
    exp_path = model_path.split(os.sep)
    result_path = os.path.join('../result', exp_path[2], exp_path[3])
    if not os.path.exists(result_path): os.makedirs(result_path)
    return result_path


if __name__ == "__main__":
    main()
