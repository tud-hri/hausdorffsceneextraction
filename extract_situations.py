import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import tqdm
from scipy.spatial.distance import directed_hausdorff

from hausdorffscenarioextraction.dataset_relative import DatasetRelative
from processing.encryptiontools import load_encrypted_pickle, save_encrypted_pickle
from hausdorffscenarioextraction.lane import Lane
from hausdorffscenarioextraction.progressprocess import ProgressProcess
from hausdorffscenarioextraction.plotting import *


def get_context_set(dataset: DatasetRelative, vehicle_id, frame_number):
    data_row = dataset.track_data.loc[(dataset.track_data['id'] == vehicle_id) &
                                      (dataset.track_data['frame'] == frame_number), :].iloc[0, :]

    driving_direction = dataset.track_meta_data.at[vehicle_id, 'drivingDirection']

    keys = []
    for other_name in ["preceding", "following", "leftPreceding", "leftAlongside", "leftFollowing", "rightPreceding", "rightAlongside", "rightFollowing"]:
        keys += [other_name + 'RelativeXCenter', other_name + 'RelativeYCenter', other_name + 'XVelocity', other_name + 'YVelocity']

    data_row.replace([np.inf, -np.inf], np.nan, inplace=True)

    # preceding vehicle velocity is set to 0.0 in HighD if no preceding vehicle exists. But is has to be filtered out for the context set so is set to nan
    if np.isnan(data_row['precedingRelativeXCenter']):
        data_row['precedingXVelocity'] = np.nan

    context_set = data_row[keys].dropna(how="all").to_numpy()
    context_set = context_set.reshape(int(len(context_set) / 4), 4)

    # convert velocities to vehicle frame (assuming 0 or pi fixed yaw for vehicles)
    if driving_direction == 1:
        context_set *= np.array([1., 1., -1., -1.])

    return context_set


def calculate_distance_between_contexts(u, v, size_penalty_factor=0):
    size_penalty = abs(len(u) - len(v)) * size_penalty_factor
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0]) + size_penalty


def get_lane_number_mapping(number_of_lane_markings):
    if number_of_lane_markings == 9:
        lane_number_mapping = {Lane.LEFT: [5, 7],
                               Lane.CENTER: [4, 8],
                               Lane.RIGHT: [3, 9],
                               Lane.MERGING: [2]}
    elif number_of_lane_markings == 8:
        lane_number_mapping = {Lane.LEFT: [4, 6],
                               Lane.CENTER: [3, 7],
                               Lane.RIGHT: [2, 8],
                               Lane.MERGING: []}
    elif number_of_lane_markings == 6:
        lane_number_mapping = {Lane.LEFT: [3, 5],
                               Lane.CENTER: [],
                               Lane.RIGHT: [2, 6],
                               Lane.MERGING: []}
    return lane_number_mapping


def get_context_differences_for_one_dataset(dataset_id, selected_context_set, selected_vehicle_lane, selected_dataset_id, selected_ego_id, selected_frame,
                                            y_scale_factor, progress_queue, path_to_data_folder):
    data = load_encrypted_pickle(os.path.join(path_to_data_folder, '%02d_relative.pkl' % dataset_id))

    all_regarded_contexts = []
    if data is None:  # pickle file was not present
        data = DatasetRelative.from_csv_files(dataset_id, path_to_data_folder=path_to_data_folder)
        save_encrypted_pickle(os.path.join(path_to_data_folder, '%02d_relative.pkl' % dataset_id), data)

    number_of_lane_markings = len(data.upper_lane_markings) + len(data.lower_lane_markings)
    lane_number_mapping = get_lane_number_mapping(number_of_lane_markings)

    included_lane_numbers = lane_number_mapping[selected_vehicle_lane]

    id_frame_sets = data.track_data.loc[data.track_data['laneId'].isin(included_lane_numbers), ['id', 'frame']]

    for id_frame_set_index in id_frame_sets.index:
        other_id, other_frame = id_frame_sets.loc[id_frame_set_index, :]
        if not (dataset_id == selected_dataset_id and selected_ego_id == other_id):
            other_context_set = get_context_set(data, other_id, other_frame)
            if other_context_set.any():
                distance = calculate_distance_between_contexts(selected_context_set * np.array([1, y_scale_factor, 1, y_scale_factor]),
                                                               other_context_set * np.array([1, y_scale_factor, 1, y_scale_factor]))

                all_regarded_contexts.append({'dataset_id': dataset_id,
                                              'vehicle_id': other_id,
                                              'frame_number': other_frame,
                                              'distance': distance})

    save_encrypted_pickle(os.path.join(path_to_data_folder, '..', 'context_distances', '%02d_context_distance_wrt_d%d_a%d_f%d.pkl' % (
        dataset_id, selected_dataset_id, selected_ego_id, selected_frame)), all_regarded_contexts)
    if progress_queue is not None:
        progress_queue.put(1)
    return all_regarded_contexts


def get_selected_context(selected_dataset_id, selected_ego_id, selected_frame, path_to_data_folder):
    data = load_encrypted_pickle(os.path.join(path_to_data_folder, '%02d_relative.pkl' % selected_dataset_id))

    if data is None:  # pickle file was not present
        data = DatasetRelative.from_csv_files(selected_dataset_id, path_to_data_folder=path_to_data_folder)
        save_encrypted_pickle(os.path.join(path_to_data_folder, '/%02d_relative.pkl' % selected_dataset_id), data)

    selected_context_set = get_context_set(data, selected_ego_id, selected_frame)

    number_of_lane_markings = len(data.upper_lane_markings) + len(data.lower_lane_markings)
    lane_number_mapping = get_lane_number_mapping(number_of_lane_markings)

    selected_vehicle_lane_id = data.track_data.loc[(data.track_data['id'] == selected_ego_id) &
                                                   (data.track_data['frame'] == selected_frame), 'laneId'].iat[0]
    for lane in Lane:
        if selected_vehicle_lane_id in lane_number_mapping[lane]:
            selected_vehicle_lane = lane
            break
    return selected_context_set, selected_vehicle_lane


def run_multiprocessing(dataset_ids, selected_context_set, selected_vehicle_lane, selected_dataset_id, selected_ego_id, selected_frame, workers=8):
    number_of_sets = len(dataset_ids)
    manager = mp.Manager()
    progress_process = ProgressProcess(number_of_sets, manager)
    progress_process.start()

    arguments = zip(dataset_ids, [selected_context_set] * number_of_sets, [selected_vehicle_lane] * number_of_sets,
                    [selected_dataset_id] * number_of_sets, [selected_ego_id] * number_of_sets, [selected_frame] * number_of_sets, [10] * number_of_sets,
                    [progress_process.queue] * number_of_sets, [os.path.join('..', 'data', 'HighD', 'data')] * number_of_sets)

    with mp.Pool(workers) as p:
        return_values = p.starmap(get_context_differences_for_one_dataset, arguments)

    all_results = [item for sub_list in return_values for item in sub_list]
    all_results_as_dataframe = pd.DataFrame(all_results)

    return all_results_as_dataframe


def sort_and_return_best(all_results, n=100):
    all_results = all_results.sort_values('distance')

    unique_id_results = all_results.drop_duplicates(['vehicle_id', 'dataset_id'])
    best = unique_id_results.iloc[0:n, :]
    best.reset_index(inplace=True)

    return best


def get_all_output_data(path_to_context_data, path_to_data, example_dataset_id, example_ego_id, example_frame, datasets_to_search):
    tag = 'd%d_a%d_f%d' % (example_dataset_id, example_ego_id, example_frame)

    if not os.path.isfile(os.path.join(path_to_context_data, 'context_distance_wrt_' + tag + '.pkl')):
        calculate_and_save_context_distances(example_dataset_id, example_ego_id, example_frame, datasets_to_search, path_to_context_data, path_to_data)

    all_results_as_dataframe = load_encrypted_pickle(os.path.join(path_to_context_data, 'context_distance_wrt_' + tag + '.pkl'))

    return all_results_as_dataframe, tag


def print_set_size_table(best_results, path_to_data_folder):
    set_sizes = []

    for result_index in tqdm.tqdm(best_results.index):
        selected_dataset_id, selected_ego_id, selected_frame = best_results.loc[result_index, ['dataset_id', 'vehicle_id', 'frame_number']]
        context_set, _ = get_selected_context(selected_dataset_id, selected_ego_id, selected_frame, path_to_data_folder)
        set_sizes += [len(context_set)]

    set_sizes = np.array(set_sizes)

    for set_size in np.unique(set_sizes):
        print(str(sum(set_sizes == set_size)) + ' of the selected sets have a set size of ' + str(set_size))


def post_process(all_results, tag, path_to_data_folder, n=100, generate_situation_images=True, generate_distribution_plots=True, plot_context_sets=True,
                 report_number_of_vehicles_in_sets=True):
    tag_as_list = tag.split('_')
    example_dataset = int(tag_as_list[0].replace('d', ''))
    example_ego_id = int(tag_as_list[1].replace('a', ''))
    example_frame = int(tag_as_list[2].replace('f', ''))

    best = sort_and_return_best(all_results, n=n)

    print(best)

    if report_number_of_vehicles_in_sets:
        print_set_size_table(best, path_to_data_folder)

    if plot_context_sets:
        plot_heatmap_of_context_sets(best, example_dataset, example_ego_id, example_frame, path_to_data_folder)

    os.chdir(os.getcwd() + '\\..')

    if generate_situation_images:
        from processing.imagegenerator import generate_images

        generate_images(example_dataset, [example_frame], ego_ids=[example_ego_id], file_names=['d1-a%d-f%d' % (example_ego_id, example_frame)], folder=tag)

        for dataset_id in tqdm.tqdm(best['dataset_id'].unique()):
            frame_numbers = best.loc[best['dataset_id'] == dataset_id, 'frame_number'].to_list()
            vehicle_ids = best.loc[best['dataset_id'] == dataset_id, 'vehicle_id'].to_list()
            ranks = best.loc[best['dataset_id'] == dataset_id, 'vehicle_id'].index.to_list()

            file_names = []
            for frame, vehicle, rank in zip(frame_numbers, vehicle_ids, ranks):
                file_names += ['%03d-d%d-a%d-f%d' % (rank + 1, dataset_id, vehicle, frame)]

            generate_images(dataset_id, frame_ids=frame_numbers, ego_ids=vehicle_ids, file_names=file_names, folder=tag)

    if generate_distribution_plots:
        get_distribution_plots(best, time_stamps=[0, 1, 2, 3], path_to_data_folder=path_to_data_folder)


def calculate_and_save_context_distances(example_dataset_id, example_ego_id, example_frame, datasets_to_search, path_to_context_data, path_to_data):

    example_context_set, example_vehicle_lane = get_selected_context(example_dataset_id, example_ego_id, example_frame, path_to_data)
    all_results = run_multiprocessing(datasets_to_search, example_context_set, example_vehicle_lane, example_dataset_id, example_ego_id, example_frame)

    tag = 'd%d_a%d_f%d' % (example_dataset_id, example_ego_id, example_frame)
    save_encrypted_pickle(os.path.join(path_to_context_data, 'context_distance_wrt_' + tag + '.pkl'), all_results)


if __name__ == '__main__':
    example_dataset_id = 1
    example_ego_id = 21
    example_frame = 379
    datasets_to_search = [1, 2]

    path_to_context_data = os.path.join('..', 'data', 'HighD', 'context_distances')
    path_to_data_folder = os.path.join('..', 'data', 'HighD', 'data')

    if not os.path.isdir(path_to_context_data):
        os.makedirs(path_to_context_data, exist_ok=True)

    all_results, tag = get_all_output_data(path_to_context_data, path_to_data_folder, example_dataset_id, example_ego_id, example_frame, datasets_to_search)

    post_process(all_results, tag=tag, n=250, path_to_data_folder=path_to_data_folder, generate_situation_images=False, generate_distribution_plots=True,
                 plot_context_sets=True, report_number_of_vehicles_in_sets=False)
