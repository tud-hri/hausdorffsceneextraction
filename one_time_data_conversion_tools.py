import numpy as np
import tqdm

from dataobjects.dataset_relative import DatasetRelative
from dataobjects.vehicletype import VehicleType
from encryptiontools import load_encrypted_pickle
from variability.lane import Lane
from variability.maneuver_state import ManeuverState


def append_time_gap_info(plotting_data_object, dataset):
    ego_id = plotting_data_object.ego_id

    for attribute_name in ['left_lane_following', 'left_lane_preceding', 'ego_lane_following', 'ego_lane_preceding', 'right_lane_following',
                           'right_lane_preceding']:
        data_frame = plotting_data_object.__dict__[attribute_name]

        frames = data_frame['frame']

        ego_v = dataset.track_data.loc[(dataset.track_data['id'] == ego_id) &
                                       (dataset.track_data['frame'].isin(frames)), 'xVelocity'].abs().to_numpy()

        data_frame['time_gap'] = data_frame['gap'] / ego_v


def fill_missing_dataset_ids(all_data_objects, data_per_dataset):
    reconstructed_object_dict = {ManeuverState.NO_LANE_CHANGE: [],
                                 ManeuverState.AFTER_LANE_CHANGE: [],
                                 ManeuverState.BEFORE_LANE_CHANGE: []}

    for dataset_id, object_dict in data_per_dataset.items():
        for key, object_list in object_dict.items():
            for data_object in object_list:
                data_object.dataset_id = int(dataset_id)
            reconstructed_object_dict[key].extend(object_list)

    for state in ManeuverState:
        for index, item in enumerate(reconstructed_object_dict[state]):
            assert item.ego_id == all_data_objects[state][index].ego_id
            all_data_objects[state][index].dataset_id = item.dataset_id


def add_no_lc_to_sorted_data_dict(all_data_objects, data_per_dataset):
    data: DatasetRelative

    all_no_lc_ids = {}

    for dataset_index in tqdm.tqdm(range(1, 61)):
        data = load_encrypted_pickle('../data/%02d_relative.pkl' % dataset_index)

        all_no_lc_ids[str(dataset_index)] = []
        # define constant dimensions of context
        preceding_relative_position_bounds = [20, 25]
        preceding_relative_velocity_bounds = [-2, -1]
        context_initial_lane = Lane.RIGHT
        vehicle_type = VehicleType.CAR

        # select included vehicles
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

        vehicle_ids = find_vehicles_in_context(preceding_relative_position_bounds,
                                               preceding_relative_velocity_bounds,
                                               included_lane_numbers,
                                               vehicle_type,
                                               data)

        # Divide trajectories
        for vehicle_of_interest in vehicle_ids:
            _, vehicle_id, first_frame = vehicle_of_interest

            trajectory = data.track_data.loc[(data.track_data['id'] == vehicle_id) & (data.track_data['frame'] >= first_frame), :]
            driving_direction = data.track_meta_data.at[vehicle_id, 'drivingDirection']

            all_lane_ids = trajectory['laneId'].unique()
            number_of_lane_changes = sum(abs(all_lane_ids - np.roll(all_lane_ids, -1))[:-1])

            if number_of_lane_changes == 0:
                all_no_lc_ids[str(dataset_index)].append(vehicle_id)

    index = 0
    for inner_dataset_id, ids_list in all_no_lc_ids.items():
        for ego_id in ids_list:
            assert all_data_objects[ManeuverState.NO_LANE_CHANGE][index].ego_id == ego_id
            data_per_dataset[inner_dataset_id][ManeuverState.NO_LANE_CHANGE].append(all_data_objects[ManeuverState.NO_LANE_CHANGE][index])
            index += 1
