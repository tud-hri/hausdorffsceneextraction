import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets

from encryptiontools import save_encrypted_pickle
from gui import TrafficVisualizerGui, SimMaster
from variability.lcdirection import LaneChangeDirection
from variability.one_time_data_conversion_tools import *
from variability.plotdata import PlotData


def find_vehicles_in_context(x_bounds, v_bounds, context_initial_lane, vehicle_type: VehicleType, data: DatasetRelative):
    x_bounds = sorted(x_bounds)
    v_bounds = sorted(v_bounds)

    number_of_lane_markings = len(data.upper_lane_markings) + len(data.lower_lane_markings)
    lane_number_mapping = {Lane.LEFT: [],
                           Lane.CENTER: [],
                           Lane.RIGHT: [],
                           Lane.MERGING: []}

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

    included_lane_numbers = lane_number_mapping[context_initial_lane]

    situations_list = []
    dataset_id = data.recording_id

    vehicles_of_correct_type = data.track_meta_data.index[data.track_meta_data['class'] == vehicle_type.as_csv_string()]

    data_within_bounds = data.track_data.loc[(data.track_data['dhw'] >= x_bounds[0]) &
                                             (data.track_data['dhw'] <= x_bounds[1]) &
                                             (data.track_data['precedingRelativeXVelocity'] >= v_bounds[0]) &
                                             (data.track_data['precedingRelativeXVelocity'] <= v_bounds[1]) &
                                             (data.track_data['laneId'].isin(included_lane_numbers)) &
                                             (data.track_data['id'].isin(vehicles_of_correct_type)), :]

    for vehicle_id in data_within_bounds['id'].unique():
        first_frames_in_situation = data_within_bounds.loc[data_within_bounds['id'] == vehicle_id, 'frame'].to_numpy()[0]
        situations_list.append((dataset_id, vehicle_id, first_frames_in_situation))

    return situations_list


def plot_in_six_figures(plot_data_objects, state: ManeuverState, title_suffix='', use_time_gap=False):
    longest_data_trace = 0
    title = 'All ' + str(state).lower()
    if title_suffix:
        title += ' ' + title_suffix

    for data_object in plot_data_objects:
        length = data_object.last_frame - data_object.first_frame
        if length > longest_data_trace:
            longest_data_trace = length

    color_map = mpl.cm.cool
    color_map_norm = mpl.colors.Normalize(vmin=0, vmax=longest_data_trace)
    color_map_time_norm = mpl.colors.Normalize(vmin=0.0, vmax=longest_data_trace / 25)

    fig = plt.figure(title, figsize=(10, 10))
    grid = mpl.gridspec.GridSpec(ncols=2, nrows=4, height_ratios=[1., 1., 1., 0.1])

    top_left_plot = fig.add_subplot(grid[0])
    top_right_plot = fig.add_subplot(grid[1])

    center_left_plot = fig.add_subplot(grid[2])
    center_right_plot = fig.add_subplot(grid[3])

    bottom_left_plot = fig.add_subplot(grid[4])
    bottom_right_plot = fig.add_subplot(grid[5])

    color_bar_plot = fig.add_subplot(grid[-2:])
    grid.update(hspace=0.5)
    plt.suptitle(title)

    # plot the data
    for data_object in plot_data_objects:
        data_object.plot(color_map, color_map_norm, top_left_plot, top_right_plot, center_left_plot, center_right_plot, bottom_left_plot,
                         bottom_right_plot, use_time_gap=use_time_gap)

    color_bar = mpl.colorbar.ColorbarBase(color_bar_plot, cmap=color_map, norm=color_map_time_norm, orientation='horizontal')
    color_bar.set_label('time [s]')

    if use_time_gap:
        xmin = 5.
        xmax = 7.5
    else:
        xmin = 15.
        xmax = 60.
    ylim = 0.5

    top_left_plot.set_xlim([-xmax, xmin])
    top_right_plot.set_xlim([-xmin, xmax])
    center_left_plot.set_xlim([-xmax, xmin])
    center_right_plot.set_xlim([-xmin, xmax])
    bottom_left_plot.set_xlim([-xmax, xmin])
    bottom_right_plot.set_xlim([-xmin, xmax])

    top_left_plot.set_ylim([-ylim, ylim])
    top_right_plot.set_ylim([-ylim, ylim])
    center_left_plot.set_ylim([-ylim, ylim])
    center_right_plot.set_ylim([-ylim, ylim])
    bottom_left_plot.set_ylim([-ylim, ylim])
    bottom_right_plot.set_ylim([-ylim, ylim])

    return fig


def get_plot_data(dataset_ids):
    data: DatasetRelative

    all_data_objects = {ManeuverState.NO_LANE_CHANGE: [],
                        ManeuverState.BEFORE_LANE_CHANGE: [],
                        ManeuverState.AFTER_LANE_CHANGE: []}

    for dataset_index in tqdm.tqdm(dataset_ids):
        data = load_encrypted_pickle('../data/%02d_relative.pkl' % dataset_index)

        if data is None:  # pickle file was not present
            data = DatasetRelative.from_csv_files(dataset_index, path_to_data_folder='../data/')
            save_encrypted_pickle('../data/%02d_relative.pkl' % dataset_index, data)

        # define constant dimensions of context
        preceding_relative_position_bounds = [20, 25]
        preceding_relative_velocity_bounds = [-2, -1]
        context_initial_lane = Lane.RIGHT
        vehicle_type = VehicleType.CAR

        # select included vehicles
        vehicle_ids = find_vehicles_in_context(preceding_relative_position_bounds,
                                               preceding_relative_velocity_bounds,
                                               context_initial_lane,
                                               vehicle_type,
                                               data)

        # Divide trajectories
        for vehicle_of_interest in vehicle_ids:
            _, vehicle_id, first_frame = vehicle_of_interest

            trajectory = data.track_data.loc[(data.track_data['id'] == vehicle_id) & (data.track_data['frame'] >= first_frame), :]
            driving_direction = data.track_meta_data.at[vehicle_id, 'drivingDirection']

            all_lane_ids = trajectory['laneId'].to_numpy()
            number_of_lane_changes = sum(abs(all_lane_ids - np.roll(all_lane_ids, -1))[:-1])

            if number_of_lane_changes == 0:
                lane_id = trajectory['laneId'].iat[0]

                plot_data = PlotData(vehicle_id, dataset_index, lane_id, driving_direction, ManeuverState.NO_LANE_CHANGE, LaneChangeDirection.NONE)
                last_frame = trajectory['frame'].max()

                horizon_data = data.track_data.loc[(data.track_data['frame'] >= first_frame) & (data.track_data['frame'] <= last_frame), :]

                plot_data.extract_data_from_dataframe(horizon_data)
                all_data_objects[ManeuverState.NO_LANE_CHANGE].append(plot_data)
            elif number_of_lane_changes == 1:
                initial_lane = trajectory['laneId'].iat[0]
                final_lane = trajectory['laneId'].iat[-1]

                if driving_direction == 1:
                    lane_change_direction = LaneChangeDirection.LEFT if final_lane > initial_lane else LaneChangeDirection.RIGHT
                else:  # driving_direction == 2
                    lane_change_direction = LaneChangeDirection.LEFT if final_lane < initial_lane else LaneChangeDirection.RIGHT

                for lane_id in trajectory['laneId'].unique():
                    if lane_id == initial_lane:
                        state = ManeuverState.BEFORE_LANE_CHANGE
                    else:
                        state = ManeuverState.AFTER_LANE_CHANGE

                    plot_data = PlotData(vehicle_id, dataset_index, lane_id, driving_direction, state, lane_change_direction)
                    last_frame = trajectory.loc[trajectory['laneId'] == lane_id]['frame'].max()
                    first_frame = trajectory.loc[trajectory['laneId'] == lane_id]['frame'].min()

                    horizon_data = data.track_data.loc[(data.track_data['frame'] >= first_frame) & (data.track_data['frame'] <= last_frame), :]
                    plot_data.extract_data_from_dataframe(horizon_data)

                    all_data_objects[state].append(plot_data)

    return all_data_objects


def scatter_of_start_points(data_objects, state):
    fig = plt.figure('Scatter start points ' + str(state).lower())
    left_following_plot = fig.add_subplot(1, 2, 1)
    ego_preceding_plot = fig.add_subplot(1, 2, 2)

    for data_object in data_objects:
        try:
            ego_preceding_plot.scatter(data_object.ego_lane_preceding['gap'].iat[0], 1 / data_object.ego_lane_preceding['ttc'].iat[0])
        except IndexError:
            pass
        try:
            left_following_plot.scatter(data_object.left_lane_following['gap'].iat[0], 1 / data_object.left_lane_following['ttc'].iat[0])
        except IndexError:
            pass

    return fig


def compare_plotting_methods(batch, title_suffix=''):
    longest_data_trace = 0
    title = 'Compare plotting forms'
    if title_suffix:
        title += ' ' + title_suffix

    for data_object in batch:
        length = data_object.last_frame - data_object.first_frame
        if length > longest_data_trace:
            longest_data_trace = length

    for subset in ['ego_lane_preceding', 'right_lane_preceding']:
        title += ' ' + subset
        color_map = mpl.cm.cool
        color_map_norm = mpl.colors.Normalize(vmin=0, vmax=longest_data_trace)
        color_map_time_norm = mpl.colors.Normalize(vmin=0.0, vmax=longest_data_trace / 25)

        fig = plt.figure(title, figsize=(10, 10))
        grid = mpl.gridspec.GridSpec(ncols=2, nrows=3, height_ratios=[1., 1., 0.1])

        top_left_plot = fig.add_subplot(grid[0])
        top_right_plot = fig.add_subplot(grid[1])

        center_left_plot = fig.add_subplot(grid[2])
        center_right_plot = fig.add_subplot(grid[3])

        color_bar_plot = fig.add_subplot(grid[-2:])
        grid.update(hspace=0.5)
        plt.suptitle(title)

        last_id = 0
        raw_data = None

        # plot the data
        for data_object in batch:
            for other_id in data_object.__dict__[subset]['other_id'].unique():
                first_frame = data_object.first_frame
                last_frame = data_object.last_frame
                other_id = int(other_id)

                data_to_plot = data_object.__dict__[subset].loc[data_object.__dict__[subset]['other_id'] == other_id]
                plot_in_frame(data_to_plot['gap'], 1 / data_to_plot['ttc'], data_to_plot['frame'] - first_frame, top_left_plot, color_map, color_map_norm,
                              xlabel='gap [m]', ylabel='1/ttc [1/s]')

                if data_object.dataset_id != last_id:
                    last_id = data_object.dataset_id
                    raw_data = load_encrypted_pickle('../data/%02d.pkl' % last_id)

                initial_ego_position = raw_data.track_data.loc[(raw_data.track_data['id'] == data_object.ego_id) &
                                                               (raw_data.track_data['frame'] == first_frame), ['x', 'y']].to_numpy()
                driving_direction = data_object.driving_direction
                other_length = raw_data.track_meta_data.at[other_id, 'width']
                other_width = raw_data.track_meta_data.at[other_id, 'height']
                ego_length = raw_data.track_meta_data.at[data_object.ego_id, 'width']
                ego_width = raw_data.track_meta_data.at[data_object.ego_id, 'height']

                other_position = raw_data.track_data.loc[(raw_data.track_data['id'] == other_id) &
                                                         (raw_data.track_data['frame'] >= first_frame) &
                                                         (raw_data.track_data['frame'] <= last_frame), ['x', 'y']].to_numpy()
                other_frames = raw_data.track_data.loc[(raw_data.track_data['id'] == other_id) &
                                                       (raw_data.track_data['frame'] >= first_frame) &
                                                       (raw_data.track_data['frame'] <= last_frame), 'frame'].to_numpy()

                if driving_direction == 1:
                    relative_other_position = ((other_position + np.array([other_length, other_width / 2])) - (
                            initial_ego_position + np.array([0.0, ego_width / 2]))) * np.array([-1., 1.])
                else:  # driving_direction == 2
                    relative_other_position = ((other_position + np.array([0., other_width / 2])) - (
                            initial_ego_position + np.array([ego_length, ego_width / 2]))) * np.array([1., -1.])

                plot_in_frame(relative_other_position[:, 0], relative_other_position[:, 1], other_frames - first_frame, top_right_plot, color_map,
                              color_map_norm,
                              xlabel='position [m]', ylabel='position [m]')

                other_actions = raw_data.track_data.loc[(raw_data.track_data['id'] == other_id) &
                                                        (raw_data.track_data['frame'] >= first_frame) &
                                                        (raw_data.track_data['frame'] <= last_frame), ['xAcceleration', 'yAcceleration']].to_numpy()

                plot_in_frame(other_actions[:, 0], other_actions[:, 1], other_frames - first_frame, center_left_plot, color_map, color_map_norm,
                              xlabel='x-acceleration [m/s^2]', ylabel='y-acceleration [m/s^2]')

                plot_in_frame(data_to_plot['gap'], data_to_plot['relative_velocity'], data_to_plot['frame'] - first_frame, center_right_plot, color_map,
                              color_map_norm,
                              xlabel='gap [m]', ylabel='relative v [m/s]')

        top_left_plot.set_ylim([-1., 2.5])

        color_bar = mpl.colorbar.ColorbarBase(color_bar_plot, cmap=color_map, norm=color_map_time_norm, orientation='horizontal')
        color_bar.set_label('time [s]')


def plot_positions(data_objects, title=''):
    last_data_id = 0
    data = None

    longest_data_trace = 0

    for data_object in data_objects:
        length = data_object.last_frame - data_object.first_frame
        if length > longest_data_trace:
            longest_data_trace = length
    color_map = mpl.cm.cool
    color_map_norm = mpl.colors.Normalize(vmin=0, vmax=longest_data_trace)
    color_map_time_norm = mpl.colors.Normalize(vmin=0.0, vmax=longest_data_trace / 25)

    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    plt.title(title)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for data_object in data_objects:
        all_other_ids = set(list(data_object.left_lane_following['other_id'].unique()) + list(data_object.left_lane_preceding['other_id'].unique()) + list(
            data_object.ego_lane_following['other_id'].unique()) + list(data_object.ego_lane_preceding['other_id'].unique()) + list(
            data_object.right_lane_following['other_id'].unique()) + list(data_object.right_lane_preceding['other_id'].unique()))

        if data_object.dataset_id != last_data_id:
            last_data_id = data_object.dataset_id
            data = load_encrypted_pickle('../data/%02d.pkl' % last_data_id)

        # step 1 convert everything to center vehicle points
        ego_position = data.track_data.loc[(data.track_data['id'] == data_object.ego_id) &
                                           (data.track_data['frame'] >= data_object.first_frame) &
                                           (data.track_data['frame'] <= data_object.last_frame), ['x', 'y', 'frame']].to_numpy()
        ego_length, ego_width = data.track_meta_data.loc[data_object.ego_id, ['width', 'height']].to_numpy()
        ego_position[:, 0:2] = ego_position[:, 0:2] + np.array([ego_length / 2, ego_width / 2])
        ego_position[:, 2] = ego_position[:, 2] - data_object.first_frame

        all_other_positions = []
        for other_id in all_other_ids:
            other_positions = data.track_data.loc[(data.track_data['id'] == other_id) &
                                                  (data.track_data['frame'] >= data_object.first_frame) &
                                                  (data.track_data['frame'] <= data_object.last_frame), ['x', 'y', 'frame']].to_numpy()
            other_length, other_width = data.track_meta_data.loc[other_id, ['width', 'height']].to_numpy()
            other_positions[:, 0:2] = other_positions[:, 0:2] + np.array([other_length / 2, other_width / 2])
            other_positions[:, 2] = other_positions[:, 2] - data_object.first_frame
            all_other_positions.append(other_positions)

        # step 2 find local origin (rotation + translation) and convert all
        driving_direction = data.track_meta_data.at[data_object.ego_id, 'drivingDirection']

        if driving_direction == 1:
            rotation_factor = np.array([-1., 1.])
            y_origin = data.upper_lane_markings[0]
        else:
            rotation_factor = np.array([1., -1.])
            y_origin = data.lower_lane_markings[-1]

        x_origin = ego_position[0, 0]
        origin = np.array([x_origin, y_origin])

        ego_position[:, 0:2] = (ego_position[:, 0:2] - origin) * rotation_factor
        for other_position in all_other_positions:
            other_position[:, 0:2] = (other_position[:, 0:2] - origin) * rotation_factor

        # step 3 plot all
        plot_in_frame(ego_position[:, 0], ego_position[:, 1], ego_position[:, 2], ax, color_map, color_map_norm, line_color=colors[0])

        for color_index, other_position in enumerate(all_other_positions):
            plot_in_frame(other_position[:, 0], other_position[:, 1], other_position[:, 2], ax, color_map, color_map_norm,
                          line_color=colors[color_index + 1])
    return figure


def plot_positions_of_complete_lc(before_lc_data_objects, after_lc_data_objects, title=''):
    last_data_id = 0
    data = None

    figure = plt.figure(title)
    ax = figure.add_subplot(1, 1, 1)
    plt.title(title)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    color_index = 0

    for before_lc_data_object, after_lc_data_object in zip(before_lc_data_objects, after_lc_data_objects):
        assert before_lc_data_object.ego_id == after_lc_data_object.ego_id and before_lc_data_object.dataset_id == after_lc_data_object.dataset_id

        all_other_ids = set(list(before_lc_data_object.left_lane_following['other_id'].unique()) +
                            list(before_lc_data_object.left_lane_preceding['other_id'].unique()) +
                            list(before_lc_data_object.ego_lane_following['other_id'].unique()) +
                            list(before_lc_data_object.ego_lane_preceding['other_id'].unique()) +
                            list(before_lc_data_object.right_lane_following['other_id'].unique()) +
                            list(before_lc_data_object.right_lane_preceding['other_id'].unique()) +
                            list(before_lc_data_object.left_lane_following['other_id'].unique()) +
                            list(before_lc_data_object.left_lane_preceding['other_id'].unique()) +
                            list(before_lc_data_object.ego_lane_following['other_id'].unique()) +
                            list(before_lc_data_object.ego_lane_preceding['other_id'].unique()) +
                            list(before_lc_data_object.right_lane_following['other_id'].unique()) +
                            list(before_lc_data_object.right_lane_preceding['other_id'].unique())
                            )

        if before_lc_data_object.dataset_id != last_data_id:
            last_data_id = before_lc_data_object.dataset_id
            data = load_encrypted_pickle('../data/%02d.pkl' % last_data_id)

        # step 1 convert everything to center vehicle points
        ego_position = data.track_data.loc[(data.track_data['id'] == before_lc_data_object.ego_id) &
                                           (data.track_data['frame'] >= before_lc_data_object.first_frame) &
                                           (data.track_data['frame'] <= after_lc_data_object.last_frame), ['x', 'y', 'frame']].to_numpy()
        ego_length, ego_width = data.track_meta_data.loc[before_lc_data_object.ego_id, ['width', 'height']].to_numpy()
        ego_position[:, 0:2] = ego_position[:, 0:2] + np.array([ego_length / 2, ego_width / 2])
        ego_position[:, 2] = ego_position[:, 2] - before_lc_data_object.first_frame

        all_other_positions = []
        for other_id in all_other_ids:
            other_positions = data.track_data.loc[(data.track_data['id'] == other_id) &
                                                  (data.track_data['frame'] >= before_lc_data_object.first_frame) &
                                                  (data.track_data['frame'] <= after_lc_data_object.last_frame), ['x', 'y', 'frame']].to_numpy()
            other_length, other_width = data.track_meta_data.loc[other_id, ['width', 'height']].to_numpy()
            other_positions[:, 0:2] = other_positions[:, 0:2] + np.array([other_length / 2, other_width / 2])
            other_positions[:, 2] = other_positions[:, 2] - before_lc_data_object.first_frame
            all_other_positions.append(other_positions)

        # step 2 find local origin (rotation + translation) and convert all
        driving_direction = data.track_meta_data.at[before_lc_data_object.ego_id, 'drivingDirection']

        if driving_direction == 1:
            rotation_factor = np.array([-1., 1.])
            y_origin = data.upper_lane_markings[0]
        else:
            rotation_factor = np.array([1., -1.])
            y_origin = data.lower_lane_markings[-1]

        x_origin = ego_position[0, 0]
        origin = np.array([x_origin, y_origin])

        ego_position[:, 0:2] = (ego_position[:, 0:2] - origin) * rotation_factor
        for other_position in all_other_positions:
            other_position[:, 0:2] = (other_position[:, 0:2] - origin) * rotation_factor

        # step 3 plot all
        plt.plot(ego_position[:, 0], ego_position[:, 1], color=colors[color_index], linestyle='--')

        for other_position in all_other_positions:
            plt.plot(other_position[:, 0], other_position[:, 1], color=colors[color_index], linestyle='-')

        color_index += 1
    plt.xlabel('longitudinal position [m]')
    plt.ylabel('lateral position [m]')
    return figure


def plot_in_frame(x_data, y_data, normalized_frames, plot_frame, time_color_map, color_map_norm, xlabel='', ylabel='', title='', line_color=''):
    if line_color:
        plot_frame.plot(x_data, y_data, line_color)
    else:
        plot_frame.plot(x_data, y_data)

    plot_frame.scatter(x_data[::25], y_data[::25], c=normalized_frames[::25], cmap=time_color_map, norm=color_map_norm)
    plot_frame.set_xlabel(xlabel)
    plot_frame.set_ylabel(ylabel)
    plot_frame.set_title(title)


def all_surrounding_ids_are_unique(plot_data_object):
    return (len(plot_data_object.left_lane_following['other_id'].unique()) <= 1) & \
           (len(plot_data_object.left_lane_preceding['other_id'].unique()) <= 1) & \
           (len(plot_data_object.ego_lane_following['other_id'].unique()) <= 1) & \
           (len(plot_data_object.ego_lane_preceding['other_id'].unique()) <= 1) & \
           (len(plot_data_object.right_lane_following['other_id'].unique()) <= 1) & \
           (len(plot_data_object.right_lane_preceding['other_id'].unique()) <= 1)


def find_unique_others(data_objects):
    unique_context_ids = {}
    for state in ManeuverState:
        unique_context_ids[state] = []
        for data_object in data_objects[state]:
            if all_surrounding_ids_are_unique(data_object):
                unique_context_ids[state].append(data_object)
    return unique_context_ids


def print_statistics(data_objects):
    total_no_lane_change = len(data_objects[ManeuverState.NO_LANE_CHANGE])
    total_lane_change = len(data_objects[ManeuverState.BEFORE_LANE_CHANGE])
    total_number_of_trajectories = total_no_lane_change + total_lane_change

    all_lane_change_sets = zip(data_objects[ManeuverState.BEFORE_LANE_CHANGE], data_objects[ManeuverState.AFTER_LANE_CHANGE])

    lengths_for_no_lc = [do.last_frame - do.first_frame for do in all_data_objects[ManeuverState.NO_LANE_CHANGE]]
    lengths_for_lc = [(do_blc.last_frame - do_blc.first_frame) + (do_alc.last_frame - do_alc.first_frame) for do_blc, do_alc in all_lane_change_sets]

    length_per_trajectory = lengths_for_no_lc + lengths_for_lc
    frames_until_lane_change = [do.last_frame - do.first_frame for do in all_data_objects[ManeuverState.BEFORE_LANE_CHANGE]]

    number_of_passing_cars_before_lane_change = []

    for data_object in data_objects[ManeuverState.BEFORE_LANE_CHANGE]:
        passing_ids = np.intersect1d(data_object.left_lane_following['other_id'].unique(), data_object.left_lane_preceding['other_id'].unique())
        number_of_passing_cars_before_lane_change.append(len(passing_ids))

    average_velocities = {ManeuverState.NO_LANE_CHANGE: [],
                          ManeuverState.BEFORE_LANE_CHANGE: [],
                          ManeuverState.AFTER_LANE_CHANGE: []}

    initial_left_follower_gap = {ManeuverState.NO_LANE_CHANGE: [],
                                 ManeuverState.BEFORE_LANE_CHANGE: [],
                                 ManeuverState.AFTER_LANE_CHANGE: []}
    initial_left_follower_ttc = {ManeuverState.NO_LANE_CHANGE: [],
                                 ManeuverState.BEFORE_LANE_CHANGE: [],
                                 ManeuverState.AFTER_LANE_CHANGE: []}

    initial_left_preceding_gap = {ManeuverState.NO_LANE_CHANGE: [],
                                  ManeuverState.BEFORE_LANE_CHANGE: [],
                                  ManeuverState.AFTER_LANE_CHANGE: []}
    initial_left_preceding_ttc = {ManeuverState.NO_LANE_CHANGE: [],
                                  ManeuverState.BEFORE_LANE_CHANGE: [],
                                  ManeuverState.AFTER_LANE_CHANGE: []}

    number_of_lane_changes = []
    last_dataset_id = 0
    data = None

    vehicles_tracked_from_first_frame = 0

    for state in ManeuverState:
        for data_object in tqdm.tqdm(all_data_objects[state]):
            if data_object.dataset_id != last_dataset_id:
                last_dataset_id = data_object.dataset_id
                data = load_encrypted_pickle('../data/%02d.pkl' % last_dataset_id)

            average_velocity = data.track_data.loc[(data.track_data['id'] == data_object.ego_id) &
                                                   (data.track_data['frame'] >= data_object.first_frame) &
                                                   (data.track_data['frame'] <= data_object.last_frame), 'xVelocity'].abs().mean()
            average_velocities[state].append(average_velocity)

            if state in [ManeuverState.NO_LANE_CHANGE, ManeuverState.BEFORE_LANE_CHANGE]:
                if data_object.first_frame == data.track_data.loc[data.track_data['id'] == data_object.ego_id, 'frame'].min():
                    vehicles_tracked_from_first_frame += 1
                try:
                    left_follower_gap = data_object.left_lane_following['gap'].iat[0]
                    left_follower_ttc = data_object.left_lane_following['ttc'].iat[0]
                    initial_left_follower_ttc[state].append(left_follower_ttc)
                except IndexError:
                    left_follower_gap = 100.

                try:
                    left_preceding_gap = data_object.left_lane_preceding['gap'].iat[0]
                    left_preceding_ttc = data_object.left_lane_preceding['ttc'].iat[0]
                    initial_left_preceding_ttc[state].append(left_preceding_ttc)
                except IndexError:
                    left_preceding_gap = 100.

                initial_left_follower_gap[state].append(left_follower_gap)
                initial_left_preceding_gap[state].append(left_preceding_gap)

    unique_context_ids = find_unique_others(data_objects)
    for state in ManeuverState:
        print('The total number of %s vehicles with unique context is %d' % (state, len(unique_context_ids[state])))

    print('The total number of trajectories is %d' % total_number_of_trajectories)
    print('The total number of lane changes is %d' % total_lane_change)
    print('The total number of car following is %d' % total_no_lane_change)
    print('Number of vehicles that are tracked from their initial frame is %d' % vehicles_tracked_from_first_frame)

    plt.figure()
    time_bins = [t for t in range(0, 100, 2)]
    plt.hist(np.array(length_per_trajectory) / 25, label='All data', bins=time_bins)
    plt.hist(np.array(lengths_for_no_lc) / 25, label='No lane change', bins=time_bins)
    plt.hist(np.array(lengths_for_lc) / 25, label='Lane change', bins=time_bins)
    plt.hist(np.array(frames_until_lane_change) / 25, label='Frames until lane change', bins=time_bins)
    plt.legend()
    plt.xlabel('time [s]')
    plt.title('Trajectory duration')

    plt.figure()
    plt.hist(number_of_passing_cars_before_lane_change, bins=4)
    plt.title('Number of overtaking cars before lane change')

    plt.figure()
    for state in ManeuverState:
        plt.hist(average_velocities[state], label=str(state))
    plt.legend()
    plt.title('Average velocity of the monitored vehicle')
    plt.xlabel('velocity in [m/s]')

    plt.figure()
    bins = [-100, -90, -80, -70, -60, -50, -40, -30, -20, - 15, -10, -5, 0, 5, 10, 15]
    for state in [ManeuverState.NO_LANE_CHANGE, ManeuverState.BEFORE_LANE_CHANGE]:
        plt.hist(np.clip(initial_left_follower_gap[state], bins[0], bins[-1]), bins=bins, label=str(state))
    plt.legend()
    plt.title('Initial gap with left following vehicle')
    plt.xlabel('gap [m]. Lowest bin is open, no left following is highest bin (+15m)')

    plt.figure()
    ttc_bins = [-10, 0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60]
    for state in [ManeuverState.NO_LANE_CHANGE, ManeuverState.BEFORE_LANE_CHANGE]:
        plt.hist(np.clip(np.array(initial_left_follower_ttc[state]), ttc_bins[0], ttc_bins[-1]), bins=ttc_bins, label=str(state))
    plt.legend()
    plt.title('Initial ttc with left following vehicle')
    plt.xlabel('ttc [s]')

    plt.figure()
    bins = [-15, -10, -5, 0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for state in [ManeuverState.NO_LANE_CHANGE, ManeuverState.BEFORE_LANE_CHANGE]:
        plt.hist(np.clip(initial_left_preceding_gap[state], bins[0], bins[-1]), bins=bins, label=str(state))
    plt.legend()
    plt.title('Initial gap with left preceding vehicle')
    plt.xlabel('gap [m]. Lowest bin is open, no left preceding is highest bin (+100m)')

    plt.figure()
    for state in [ManeuverState.NO_LANE_CHANGE, ManeuverState.BEFORE_LANE_CHANGE]:
        plt.hist(np.clip(np.array(initial_left_preceding_ttc[state]), ttc_bins[0], ttc_bins[-1]), bins=ttc_bins, label=str(state))
    plt.legend()
    plt.title('Initial ttc with left preceding vehicle')
    plt.xlabel('ttc [s]')


def show_linked_gui(dataset, plot_object):
    os.chdir(os.getcwd() + '\\..')
    app = QtWidgets.QApplication(sys.argv)

    dataset.track_data = dataset.track_data.loc[(dataset.track_data['frame'] >= plot_object.first_frame) &
                                                (dataset.track_data['frame'] <= plot_object.last_frame), :]
    dataset.duration = (plot_object.last_frame - plot_object.first_frame) / dataset.frame_rate

    plot_object.plot_this_vehicle_in_own_figure()
    plt.show(block=False)

    gui = TrafficVisualizerGui(dataset)
    gui.selected_invisible_car_id = plot_object.ego_id
    sim = SimMaster(dataset, gui)
    gui.register_sim_master(sim)

    exit_code = app.exec_()

    sys.exit(exit_code)


if __name__ == '__main__':
    figures = []

    all_data_objects = load_encrypted_pickle('..\\data\\context_plot_all_data.pkl')

    # plot_object = all_data_objects[ManeuverState.BEFORE_LANE_CHANGE][62]
    # dataset = load_encrypted_pickle('../data/%02d.pkl' % plot_object.dataset_id)

    # show_linked_gui(dataset, plot_object)

    # data_per_dataset = load_encrypted_pickle('..\\data\\context_plot_data_per_set_wid.pkl')

    print_statistics(all_data_objects)

    batch_size = 10
    #
    # batch = all_data_objects[ManeuverState.AFTER_LANE_CHANGE][40: 60]
    # compare_plotting_methods(batch)
    #
    # for maneuver_state in [ManeuverState.BEFORE_LANE_CHANGE, ManeuverState.AFTER_LANE_CHANGE]:
    #     total_batches = int(np.ceil(len(all_data_objects[maneuver_state]) / batch_size))
    #     last_batch_size = len(all_data_objects[maneuver_state]) % batch_size
    #
    #     for batch_number in range(total_batches):
    #         if batch_number == total_batches - 1 and last_batch_size:
    #             batch = all_data_objects[maneuver_state][batch_number * batch_size: batch_number * batch_size + last_batch_size]
    #         else:
    #             batch = all_data_objects[maneuver_state][batch_number * batch_size: (batch_number + 1) * batch_size]
    #
    #         figures.append(plot_in_six_figures(batch, maneuver_state, title_suffix=str(batch_number) + ' - time gap', use_time_gap=True))
    # #
    # unique_context_ids = find_unique_others(all_data_objects)
    # for maneuver_state in ManeuverState:
    #     figures.append(plot_in_six_figures(unique_context_ids[maneuver_state], maneuver_state, use_time_gap=True, title_suffix='unique context'))
    #
    # for maneuver_state in ManeuverState:
    #     figures.append(plot_in_six_figures(all_data_objects[maneuver_state], maneuver_state, use_time_gap=True, title_suffix='all data'))
    #
    # for maneuver_state in ManeuverState:
    #     figures.append(plot_positions(all_data_objects[maneuver_state][0:10], title=str(maneuver_state)))

    # total_batches = int(np.ceil(len(all_data_objects[ManeuverState.AFTER_LANE_CHANGE]) / batch_size))
    # last_batch_size = len(all_data_objects[ManeuverState.AFTER_LANE_CHANGE]) % batch_size
    #
    # for batch_number in range(total_batches):
    #     if batch_number == total_batches - 1 and last_batch_size:
    #         batch_before_lc = all_data_objects[ManeuverState.BEFORE_LANE_CHANGE][batch_number * batch_size: batch_number * batch_size + last_batch_size]
    #         batch_after_lc = all_data_objects[ManeuverState.AFTER_LANE_CHANGE][batch_number * batch_size: batch_number * batch_size + last_batch_size]
    #     else:
    #         batch_before_lc = all_data_objects[ManeuverState.BEFORE_LANE_CHANGE][batch_number * batch_size: (batch_number + 1) * batch_size]
    #         batch_after_lc = all_data_objects[ManeuverState.AFTER_LANE_CHANGE][batch_number * batch_size: (batch_number + 1) * batch_size]
    #
    #         figures.append(plot_positions_of_complete_lc(batch_before_lc,batch_after_lc,
    #                                                      title=' 10 trajectory traces, color is a matching sample - batch ' + str(batch_number + 1)))

    # for dataset_id, dataset_data in data_per_dataset.items():
    #     for maneuver_state in ManeuverState:
    #         figures.append(plot_in_six_figures(dataset_data[maneuver_state], maneuver_state, title_suffix=dataset_id))
    # figures.append(scatter_of_start_points(dataset_data[maneuver_state], maneuver_state))

    # for index in range(1, 11):
    #     figure = all_data_objects[ManeuverState.NO_LANE_CHANGE][index].plot_this_vehicle_in_own_figure()
    #     figures.append(figure)

    plt.show()
