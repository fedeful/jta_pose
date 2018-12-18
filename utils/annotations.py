import json
import os
import pickle
import psutil


def frames_elaboration(sequence):
    track_ids = {}
    for frame_info in sequence['annolist']:
        if frame_info['is_labeled'][0]:
            for pose in frame_info['annorect']:
                pose['image'] = frame_info['image'][0]['name']
                if pose['track_id'][0] in track_ids:
                    track_ids[pose['track_id'][0]].append(pose)
                else:
                    track_ids[pose['track_id'][0]] = [pose]
    return track_ids


def main():
    if not os.path.isdir(DESTINATION_PATH):
        os.mkdir(DESTINATION_PATH)
    for file in os.listdir(ANNOTATION_PATH):
        if file.endswith('.json'):
            with open(os.path.join(ANNOTATION_PATH, file)) as jf:
                sequence = json.load(jf)
            feti = frames_elaboration(sequence)
            with open(os.path.join(DESTINATION_PATH, '{}.pickle'.format(file.split('.json')[0])), 'wb') as po:
                pickle.dump(feti, po)
            print('END {}'.format(file))


def generate_sequence_pairs(pickle_sequence):
    nsorg = {}
    for person_counter in pickle_sequence.keys():
        # tic = track id counter
        tic = 0
        while 1:
            if tic >= len(pickle_sequence[person_counter]):
                break
            tic_inc = tic + 1
            while 1:
                if tic_inc >= len(pickle_sequence[person_counter]):
                    break
                if person_counter in nsorg:
                    nsorg[person_counter].append({'pairs': '{}_{}_{}'.format(person_counter, tic, tic_inc),
                                                  'data': [pickle_sequence[person_counter][tic],
                                                           pickle_sequence[person_counter][tic_inc]]
                                                  })

                else:
                    nsorg[person_counter] = [{'pairs':  '{}_{}_{}'.format(person_counter, tic, tic_inc),
                                              'data': [pickle_sequence[person_counter][tic],
                                                       pickle_sequence[person_counter][tic_inc]]
                                              }]
                tic_inc += 1

            tic += 1
    return nsorg


def pairs_from_pickles():
    if not os.path.isdir(DESTINATION_PATH_PAIRS):
        os.mkdir(DESTINATION_PATH_PAIRS)
    for file in os.listdir(DESTINATION_PATH):
        with open(os.path.join(DESTINATION_PATH, file), 'rb') as pf:
            sequence = pickle.load(pf)
        nsorg = generate_sequence_pairs(sequence)
        with open(os.path.join(DESTINATION_PATH_PAIRS, '{}'.format(file.split('.json')[0])), 'wb') as po:
            pickle.dump(nsorg, po)
        print('END {}'.format(file))


def generate_single_sequence_pairs(pickle_sequence, seq_name):
    seq_list = []
    for person_counter in pickle_sequence.keys():
        for index in range(len(pickle_sequence[person_counter])):
            se = pickle_sequence[person_counter][index]
            seq_list.append(se['data'])
    return seq_list


def single_pairs_files():
    final_list = []
    if not os.path.isdir(DESTINATION_PATH_PAIRS_FINAL):
        os.mkdir(DESTINATION_PATH_PAIRS_FINAL)
    for file in os.listdir(DESTINATION_PATH_PAIRS):
        with open(os.path.join(DESTINATION_PATH_PAIRS, file), 'rb') as pf:
            sequence = pickle.load(pf)
        ll = generate_single_sequence_pairs(sequence, file)
        final_list += ll

        print('END {}'.format(file))
    print(psutil.virtual_memory())
    print('END')


ANNOTATION_PATH = '/majinbu/public/capra/Datasets/PoseTrack/posetrack_v0.75_labels/posetrack_data/annotations/val'
DESTINATION_PATH = './destination_test'
DESTINATION_PATH_PAIRS = './pt_validation'
DESTINATION_PATH_PAIRS_FINAL = './pairs_final'

if __name__ == '__main__':
    #main()
    pairs_from_pickles()
    #single_pairs_files()
