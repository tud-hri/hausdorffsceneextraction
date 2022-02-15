import glob
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import tqdm
from scipy.spatial.distance import directed_hausdorff

from .dataset_relative import DatasetRelative
from processing.encryptiontools import load_encrypted_pickle, save_encrypted_pickle
from .lane import Lane
from .progressprocess import ProgressProcess


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
                                            y_scale_factor=10, progress_queue=None, path_to_data_folder='../data/'):
    data = load_encrypted_pickle(path_to_data_folder + '%02d_relative.pkl' % dataset_id)

    all_regarded_contexts = []
    if data is None:  # pickle file was not present
        data = DatasetRelative.from_csv_files(dataset_id, path_to_data_folder=path_to_data_folder)
        save_encrypted_pickle(path_to_data_folder + '%02d_relative.pkl' % dataset_id, data)

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
    save_encrypted_pickle(path_to_data_folder + 'context_distances/%02d_context_distance_wrt_d%d_a%d_f%d.pkl' % (
        dataset_id, selected_dataset_id, selected_ego_id, selected_frame), all_regarded_contexts)
    if progress_queue is not None:
        progress_queue.put(1)
    return all_regarded_contexts


def get_selected_context(selected_dataset_id, selected_ego_id, selected_frame, path_to_data_folder='../data/'):
    data = load_encrypted_pickle(path_to_data_folder + '%02d_relative.pkl' % selected_dataset_id)

    if data is None:  # pickle file was not present
        data = DatasetRelative.from_csv_files(selected_dataset_id, path_to_data_folder=path_to_data_folder)
        save_encrypted_pickle(path_to_data_folder + '/%02d_relative.pkl' % selected_dataset_id, data)

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
                    [progress_process.queue] * number_of_sets)

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


def generate_assignment_file(selected_dataset_id, selected_ego_id, selected_frame, path_to_data_folder, y_scale_factor=10):
    selected_context_set, selected_vehicle_lane = get_selected_context(selected_dataset_id, selected_ego_id, selected_frame)

    dict_to_save = {'selected_dataset': selected_dataset_id,
                    'selected_ego_id': selected_ego_id,
                    'selected_frame': selected_frame,
                    'selected_context_set': selected_context_set,
                    'selected_vehicle_lane': selected_vehicle_lane,
                    'y_scale_factor': y_scale_factor}

    save_encrypted_pickle(path_to_data_folder + 'context_distances/assignment_d%d_a%d_f%d.pkl' % (selected_dataset_id, selected_ego_id, selected_frame),
                          dict_to_save)


def run_from_assignment_file(target_dataset_id, tag, path_to_data_folder):
    assignment = load_encrypted_pickle(path_to_data_folder + 'context_distances/assignment_%s.pkl' % tag)

    get_context_differences_for_one_dataset(dataset_id=target_dataset_id,
                                            selected_context_set=assignment['selected_context_set'],
                                            selected_ego_id=assignment['selected_ego_id'],
                                            selected_frame=assignment['selected_frame'],
                                            selected_vehicle_lane=assignment['selected_vehicle_lane'],
                                            selected_dataset_id=assignment['selected_dataset'],
                                            y_scale_factor=assignment['y_scale_factor'],
                                            path_to_data_folder=path_to_data_folder
                                            )


def get_all_output_data(path_to_data, tag):
    if os.path.isfile(path_to_data + 'context_distance_wrt_' + tag + '.pkl'):
        all_results_as_dataframe = load_encrypted_pickle(path_to_data + 'context_distance_wrt_' + tag + '.pkl')
    else:

        all_files = glob.glob(path_to_data + '*_context_distance_wrt_' + tag + '.pkl')
        all_results = []

        for file_path in all_files:
            all_results += load_encrypted_pickle(file_path)

        all_results_as_dataframe = pd.DataFrame(all_results)
        save_encrypted_pickle(path_to_data + 'context_distance_wrt_' + tag + '.pkl', all_results_as_dataframe)

    return all_results_as_dataframe


def print_set_size_table(best_results):
    set_sizes = []

    for result_index in tqdm.tqdm(best_results.index):
        selected_dataset_id, selected_ego_id, selected_frame = best_results.loc[result_index, ['dataset_id', 'vehicle_id', 'frame_number']]
        context_set, _ = get_selected_context(selected_dataset_id, selected_ego_id, selected_frame)
        set_sizes += [len(context_set)]

    set_sizes = np.array(set_sizes)

    for set_size in np.unique(set_sizes):
        print(str(sum(set_sizes == set_size)) + ' of the selected sets have a set size of ' + str(set_size))


def post_process(all_results, tag, n=100, generate_situation_images=True, generate_distribution_plots=True, plot_context_sets=True,
                 report_number_of_vehicles_in_sets=True):
    tag_as_list = tag.split('_')
    example_dataset = int(tag_as_list[0].replace('d', ''))
    example_ego_id = int(tag_as_list[1].replace('a', ''))
    example_frame = int(tag_as_list[2].replace('f', ''))

    best = sort_and_return_best(all_results, n=n)

    print(best)

    if report_number_of_vehicles_in_sets:
        print_set_size_table(best)

    if plot_context_sets:
        plot_heatmap_of_context_sets(best, example_dataset, example_ego_id, example_frame)

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
        get_distribution_plots(best, tag, time_stamps=[0, 1, 2, 3], path_to_data_folder='data/')


def plot_heatmap_of_context_sets(best_results, example_dataset_id, example_ego_id, example_frame, path_to_data_folder='../data/'):
    import seaborn as sns
    import matplotlib.pyplot as plt

    data = load_encrypted_pickle(path_to_data_folder + '%02d_relative.pkl' % example_dataset_id)
    if data is None:  # pickle file was not present
        data = DatasetRelative.from_csv_files(example_dataset_id, path_to_data_folder=path_to_data_folder)
        save_encrypted_pickle(path_to_data_folder + '%02d_relative.pkl' % example_dataset_id, data)

    example_context_set = get_context_set(data, example_ego_id, example_frame)

    all_context_sets = load_encrypted_pickle('all_context_sets.pkl')

    if all_context_sets is None:
        all_context_sets = []

        for dataset_id in tqdm.tqdm(best_results['dataset_id'].unique()):
            data = load_encrypted_pickle(path_to_data_folder + '%02d_relative.pkl' % dataset_id)
            if data is None:  # pickle file was not present
                data = DatasetRelative.from_csv_files(dataset_id, path_to_data_folder=path_to_data_folder)
                save_encrypted_pickle(path_to_data_folder + '%02d_relative.pkl' % dataset_id, data)

            for index in best_results.loc[best_results['dataset_id'] == dataset_id, :].index:
                vehicle_id = best_results.at[index, 'vehicle_id']
                initial_frame = best_results.at[index, 'frame_number']
                context_set = get_context_set(data, vehicle_id, initial_frame)
                all_context_sets += [context_set]

        all_context_sets = np.concatenate(all_context_sets)
        save_encrypted_pickle('all_context_sets.pkl', all_context_sets)

    f, ax = plt.subplots(2, figsize=(6, 6))
    ax[0].set_aspect(2)
    plt.sca(ax[0])
    sns.scatterplot(x=all_context_sets[:, 0], y=-all_context_sets[:, 1], s=10., label='Selected scenarios')
    sns.scatterplot(x=example_context_set[:, 0], y=-example_context_set[:, 1], marker='*', s=150., label='Example')
    plt.plot([0.0], [0.0], color='tab:green', marker='o', linestyle='none', label='Ego position')

    plt.legend()
    # plt.title('distribution of positions in context-sets')
    plt.xlabel('relative longitudinal position [m]')
    plt.ylabel('relative lateral position [m]')

    plt.sca(ax[1])
    ax[1].set_aspect(2)
    sns.scatterplot(x=all_context_sets[:, 2], y=-all_context_sets[:, 3], s=10., label='Selected scenarios')
    sns.scatterplot(x=example_context_set[:, 2], y=-example_context_set[:, 3], marker='*', s=150., label='Example')

    plt.legend()
    # plt.title('distribution of velocities in context-sets')
    plt.xlabel('x-velocity [m/s]')
    plt.ylabel('y-velocity [m/s]')


def get_distribution_plots(best_results, tag, time_stamps=range(1, 3), path_to_data_folder='../data/'):
    all_positions_after_n_seconds = {'Longitudinal position [m]': [],
                                     'Lateral position [m]': [],
                                     'time [s]': [],
                                     'vehicle id': []}

    for dataset_id in tqdm.tqdm(best_results['dataset_id'].unique()):
        data = load_encrypted_pickle(path_to_data_folder + '%02d.pkl' % dataset_id)

        for index in best_results.loc[best_results['dataset_id'] == dataset_id, :].index:
            vehicle_id = best_results.at[index, 'vehicle_id']
            initial_frame = best_results.at[index, 'frame_number']

            if data.track_meta_data.at[vehicle_id, 'numLaneChanges'] > 0:
                tactical = 'lc'
            else:
                tactical = 'cf'

            for n in time_stamps:
                relative_position = get_position_after_n_seconds(data, vehicle_id, initial_frame, n)
                if relative_position is not None:
                    all_positions_after_n_seconds['Longitudinal position [m]'].append(relative_position[0])
                    all_positions_after_n_seconds['Lateral position [m]'].append(relative_position[1])
                    all_positions_after_n_seconds['time [s]'].append('%.1f s' % float(n))
                    all_positions_after_n_seconds['vehicle id'].append(str(vehicle_id) + '-' + str(dataset_id))

    import seaborn as sns
    import matplotlib.pyplot as plt

    # sns.color_palette("tab10")
    g = sns.jointplot(data=all_positions_after_n_seconds, x='Longitudinal position [m]', y='Lateral position [m]', hue='time [s]', zorder=1.)
    # g.plot_joint(sns.kdeplot, color='tab:orange')
    sns.lineplot(data=all_positions_after_n_seconds, x='Longitudinal position [m]', y='Lateral position [m]', color='lightgray', linestyle='dashed',
                 linewidth=0.5, units='vehicle id', estimator=None, ax=g.ax_joint, zorder=0.)
    # plt.suptitle('Relative vehicle position n seconds after detected situation ')
    plt.xlabel('lon position [m]')
    plt.ylabel('lat position [m]')
    l = g.ax_joint.legend(title='Waypoints after n seconds \n for n = ')
    plt.setp(l.get_title(), multialignment='center')
    plt.show()


def get_position_after_n_seconds(data, vehicle_id, initial_frame, n):
    fps = data.frame_rate

    driving_direction = data.track_meta_data.at[vehicle_id, 'drivingDirection']
    vehicle_length = data.track_meta_data.at[vehicle_id, 'width']
    vehicle_width = data.track_meta_data.at[vehicle_id, 'height']

    initial_position = data.track_data.loc[(data.track_data['id'] == vehicle_id) & (data.track_data['frame'] == initial_frame), ['x', 'y']]
    initial_lane_id = data.track_data.loc[(data.track_data['id'] == vehicle_id) & (data.track_data['frame'] == initial_frame), 'laneId'].iat[0]
    position_after_n_seconds = data.track_data.loc[(data.track_data['id'] == vehicle_id) &
                                                   (data.track_data['frame'] == initial_frame + n * fps), ['x', 'y']]
    initial_position += np.array([vehicle_length / 2, vehicle_width / 2])
    position_after_n_seconds += np.array([vehicle_length / 2, vehicle_width / 2])

    upper_lane_centers = ((np.roll(data.upper_lane_markings, -1) - data.upper_lane_markings)[0:-1] / 2) + data.upper_lane_markings[0:-1]
    lower_lane_centers = ((np.roll(data.lower_lane_markings, -1) - data.lower_lane_markings)[0:-1] / 2) + data.lower_lane_markings[0:-1]
    all_lance_centers = [0.] + [lc for lc in upper_lane_centers] + [0.0] + [lc for lc in lower_lane_centers]

    lane_center = all_lance_centers[initial_lane_id - 1]
    origin = np.array([initial_position.to_numpy()[0][0], lane_center])

    try:
        relative_position = position_after_n_seconds.to_numpy()[0] - origin
        relative_position *= np.array([1.0, -1.0])

        if driving_direction == 1:
            relative_position *= -1.

        return relative_position
    except IndexError:
        # n seconds is out of sight
        pass


def run_main():
    selected_dataset_id = 1
    selected_ego_id = 11
    selected_frame = 35

    selected_context_set, selected_vehicle_lane = get_selected_context(selected_dataset_id, selected_ego_id, selected_frame)

    all_results = run_multiprocessing([1, 2], selected_context_set, selected_vehicle_lane, selected_dataset_id, selected_ego_id, selected_frame)

    tag = 'd%d_a%d_f%d_with_v' % (selected_dataset_id, selected_ego_id, selected_frame)
    save_encrypted_pickle('../data/context_distances/context_distance_wrt_' + tag + '.pkl', all_results)
    # Sort results
    post_process(all_results, tag)


if __name__ == '__main__':
    # run_main()
    # # generate_assignment_file(1, 21, 379, '../data/', y_scale_factor=10)
    tag = 'd1_a21_f379'
    all_results = get_all_output_data('..\\data\\context_distances\\', tag=tag)
    post_process(all_results, tag=tag, n=250, generate_situation_images=False, generate_distribution_plots=True, plot_context_sets=True,
                 report_number_of_vehicles_in_sets=False)
