"""
Copyright 2022, Olger Siebinga (o.siebinga@tudelft.nl)
This file is part of the module hausdorffsceneextraction.

hausdorffsceneextraction is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

hausdorffsceneextraction is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with hausdorffsceneextraction.  If not, see <https://www.gnu.org/licenses/>.
"""

import tqdm
import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from hausdorffsceneextraction.dataset_relative import DatasetRelative
from hausdorffsceneextraction.extract_situations import get_context_set
from processing.encryptiontools import save_encrypted_pickle, load_encrypted_pickle


def _load_dataset(dataset_id, path_to_data_folder):
    data = load_encrypted_pickle(os.path.join(path_to_data_folder, '%02d_relative.pkl' % dataset_id))
    if data is None:  # pickle file was not present
        data = DatasetRelative.from_csv_files(dataset_id, path_to_data_folder=path_to_data_folder)
        save_encrypted_pickle(os.path.join(path_to_data_folder, '%02d_relative.pkl' % dataset_id), data)

    return data


def plot_spread_of_context_sets(best_results, example_dataset_id, example_ego_id, example_frame, path_to_data_folder):
    """
    generates a scatter plot of the found context sets.
    """

    data = _load_dataset(example_dataset_id, path_to_data_folder)

    example_context_set = get_context_set(data, example_ego_id, example_frame)

    all_context_sets = []

    for dataset_id in tqdm.tqdm(best_results['dataset_id'].unique()):
        data = _load_dataset(dataset_id, path_to_data_folder)

        for index in best_results.loc[best_results['dataset_id'] == dataset_id, :].index:
            vehicle_id = best_results.at[index, 'vehicle_id']
            initial_frame = best_results.at[index, 'frame_number']
            context_set = get_context_set(data, vehicle_id, initial_frame)
            all_context_sets += [context_set]

    all_context_sets = np.concatenate(all_context_sets)

    f, ax = plt.subplots(2, figsize=(6, 6))
    ax[0].set_aspect(2)
    plt.sca(ax[0])
    sns.scatterplot(x=all_context_sets[:, 0], y=-all_context_sets[:, 1], s=10., label='Selected scenes')
    sns.scatterplot(x=example_context_set[:, 0], y=-example_context_set[:, 1], marker='*', s=150., label='Example')
    plt.plot([0.0], [0.0], color='tab:green', marker='o', linestyle='none', label='Ego position')

    plt.legend()
    # plt.title('distribution of positions in context-sets')
    plt.xlabel('relative longitudinal position [m]')
    plt.ylabel('relative lateral position [m]')

    plt.sca(ax[1])
    ax[1].set_aspect(2)
    sns.scatterplot(x=all_context_sets[:, 2], y=all_context_sets[:, 3], s=10., label='Selected scenes')
    sns.scatterplot(x=example_context_set[:, 2], y=example_context_set[:, 3], marker='*', s=150., label='Example')

    plt.legend()
    # plt.title('distribution of velocities in context-sets')
    plt.xlabel('x-velocity [m/s]')
    plt.ylabel('y-velocity [m/s]')


def plot_variability_in_responses(best_results, time_stamps, path_to_data_folder):
    """
    generates plot of the responses of human drivers to the selected scenes
    """

    all_positions_after_n_seconds = {'Longitudinal position [m]': [],
                                     'Lateral position [m]': [],
                                     'time [s]': [],
                                     'vehicle id': []}

    for dataset_id in tqdm.tqdm(best_results['dataset_id'].unique()):
        data = load_encrypted_pickle(os.path.join(path_to_data_folder, '%02d_relative.pkl' % dataset_id))

        for index in best_results.loc[best_results['dataset_id'] == dataset_id, :].index:
            vehicle_id = best_results.at[index, 'vehicle_id']
            initial_frame = best_results.at[index, 'frame_number']

            for n in time_stamps:
                relative_position = _get_position_after_n_seconds(data, vehicle_id, initial_frame, n)
                if relative_position is not None:
                    all_positions_after_n_seconds['Longitudinal position [m]'].append(relative_position[0])
                    all_positions_after_n_seconds['Lateral position [m]'].append(relative_position[1])
                    all_positions_after_n_seconds['time [s]'].append('%.1f s' % float(n))
                    all_positions_after_n_seconds['vehicle id'].append(str(vehicle_id) + '-' + str(dataset_id))

    g = sns.jointplot(data=all_positions_after_n_seconds, x='Longitudinal position [m]', y='Lateral position [m]', hue='time [s]', zorder=1.)

    sns.lineplot(data=all_positions_after_n_seconds, x='Longitudinal position [m]', y='Lateral position [m]', color='lightgray', linestyle='dashed',
                 linewidth=0.5, units='vehicle id', estimator=None, ax=g.ax_joint, zorder=0.)

    plt.xlabel('lon position [m]')
    plt.ylabel('lat position [m]')
    l = g.ax_joint.legend(title='Waypoints after n seconds \n for n = ')
    plt.setp(l.get_title(), multialignment='center')
    plt.show()


def _get_position_after_n_seconds(data, vehicle_id, initial_frame, n):
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
