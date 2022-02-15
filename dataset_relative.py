import datetime

import numpy as np
import pandas as pd


class DatasetRelative:
    """
    This dataset object stores the HighD as relative positions, velocities and accelerations with respect to an ego vehicle. It is therefore bigger in size
    then the regular dataset and should only be used if needed.
    """
    def __init__(self, ):
        self.recording_id = 0
        self.frame_rate = 0
        self.location_id = 0
        self.speed_limit = 0
        self.month = 0
        self.week_day = 0
        self.start_time = datetime.time()
        self.duration = 0
        self.total_driven_distance = 0
        self.total_driven_time = 0
        self.num_vehicles = 0
        self.num_cars = 0
        self.num_trucks = 0
        self.upper_lane_markings = []
        self.lower_lane_markings = []

        self.top_lanes_congestion_per_frame = np.array([])
        self.bottom_lanes_congestion_per_frame = np.array([])

        self.track_meta_data = pd.DataFrame()
        self.track_data = pd.DataFrame()

    @staticmethod
    def from_csv_files(dataset_index: int, path_to_data_folder='data/'):
        dataset = DatasetRelative()

        try:
            recording_meta_data = pd.read_csv(path_to_data_folder + '%02d_recordingMeta.csv' % dataset_index)
        except FileNotFoundError:
            raise ValueError('The dataset with index %d could not be loaded because the data is missing.' % dataset_index)

        dataset.recording_id = int(recording_meta_data.at[0, 'id'])
        dataset.frame_rate = int(recording_meta_data.at[0, 'frameRate'])
        dataset.location_id = int(recording_meta_data.at[0, 'locationId'])
        dataset.speed_limit = recording_meta_data.at[0, 'speedLimit']
        dataset.month = recording_meta_data.at[0, 'month']
        dataset.week_day = recording_meta_data.at[0, 'weekDay']
        dataset.start_time = datetime.datetime.strptime(recording_meta_data.at[0, 'startTime'], '%H:%M')
        dataset.duration = recording_meta_data.at[0, 'duration']
        dataset.total_driven_distance = recording_meta_data.at[0, 'totalDrivenDistance']
        dataset.total_driven_time = recording_meta_data.at[0, 'totalDrivenTime']
        dataset.num_vehicles = int(recording_meta_data.at[0, 'numVehicles'])
        dataset.num_cars = int(recording_meta_data.at[0, 'numCars'])
        dataset.num_trucks = int(recording_meta_data.at[0, 'numTrucks'])
        dataset.upper_lane_markings = [float(value) for value in recording_meta_data.at[0, 'upperLaneMarkings'].split(';')]
        dataset.lower_lane_markings = [float(value) for value in recording_meta_data.at[0, 'lowerLaneMarkings'].split(';')]

        track_meta_data = pd.read_csv(path_to_data_folder + '%02d_tracksMeta.csv' % dataset_index)
        dataset.track_meta_data = track_meta_data.astype({"id": int,
                                                          "width": float,
                                                          "height": float,
                                                          "initialFrame": int,
                                                          "finalFrame": int,
                                                          "numFrames": int,
                                                          "class": str,
                                                          "drivingDirection": int,
                                                          "traveledDistance": float,
                                                          "minXVelocity": float,
                                                          "maxXVelocity": float,
                                                          "meanXVelocity": float,
                                                          "minDHW": float,
                                                          "minTHW": float,
                                                          "minTTC": float,
                                                          "numLaneChanges": int})
        dataset.track_meta_data = dataset.track_meta_data.set_index('id')

        track_data = pd.read_csv(path_to_data_folder + '%02d_tracks.csv' % dataset_index)
        dataset.track_data = track_data.astype({"frame": int,
                                                "id": int,
                                                "x": float,
                                                "y": float,
                                                "width": float,
                                                "height": float,
                                                "xVelocity": float,
                                                "yVelocity": float,
                                                "xAcceleration": float,
                                                "yAcceleration": float,
                                                "frontSightDistance": float,
                                                "backSightDistance": float,
                                                "dhw": float,
                                                "thw": float,
                                                "ttc": float,
                                                "precedingXVelocity": float,
                                                "precedingId": int,
                                                "followingId": int,
                                                "leftPrecedingId": int,
                                                "leftAlongsideId": int,
                                                "leftFollowingId": int,
                                                "rightPrecedingId": int,
                                                "rightAlongsideId": int,
                                                "rightFollowingId": int,
                                                "laneId": int})

        # add heading angle
        dataset.track_data = pd.merge(dataset.track_data, dataset.track_meta_data.loc[:, 'drivingDirection'], left_on='id', right_index=True, how='left')
        dataset.track_data['heading'] = (dataset.track_data['drivingDirection'] * -1 + 2) * np.pi
        dataset.track_data.drop(columns='drivingDirection', inplace=True)
        dataset.track_data['xCenter'] = dataset.track_data['x'] + dataset.track_data['width'] / 2.
        dataset.track_data['yCenter'] = dataset.track_data['y'] + dataset.track_data['height'] / 2.

        # add relative positions and velocities to data
        for other_name in ["preceding", "following", "leftPreceding", "leftAlongside", "leftFollowing", "rightPreceding", "rightAlongside", "rightFollowing"]:

            data_copy = dataset.track_data.loc[:, ["id", "frame", "xCenter", "yCenter", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].copy()

            column_mapping = {"id": other_name + "Id",
                              "xCenter": other_name + "XCenter",
                              "yCenter": other_name + "YCenter",
                              "xVelocity": other_name + "XVelocity",
                              "yVelocity": other_name + "YVelocity",
                              "xAcceleration": other_name + "XAcceleration",
                              "yAcceleration": other_name + "YAcceleration",
                              }

            data_copy.rename(columns=column_mapping, inplace=True)

            if other_name == "preceding":
                data_copy.drop(columns="precedingXVelocity", inplace=True)

            dataset.track_data = pd.merge(dataset.track_data, data_copy, how="left", on=[other_name + "Id", "frame"])

            # calculate relative states
            # translate
            for lower_var, upper_var in [('xCenter', 'XCenter'), ('yCenter', 'YCenter'), ('xVelocity', 'XVelocity'), ('yVelocity', 'YVelocity'),
                                         ('xAcceleration', 'XAcceleration'), ('yAcceleration', 'YAcceleration')]:
                dataset.track_data[other_name + 'Relative' + upper_var] = dataset.track_data[other_name + upper_var] - dataset.track_data[lower_var]

            # rotate
            for x, y in [('XCenter', 'YCenter'), ('XVelocity', 'YVelocity'), ('XAcceleration', 'YAcceleration')]:
                dataset.track_data[other_name + 'Relative' + x] = np.cos(-dataset.track_data['heading']) * dataset.track_data[other_name + 'Relative' + x] - \
                                                                  np.sin(-dataset.track_data['heading']) * dataset.track_data[other_name + 'Relative' + y]
                dataset.track_data[other_name + 'Relative' + y] = np.cos(-dataset.track_data['heading']) * dataset.track_data[other_name + 'Relative' + y] + \
                                                                  np.sin(-dataset.track_data['heading']) * dataset.track_data[other_name + 'Relative' + x]

        dataset.track_data.replace(np.nan, np.inf, inplace=True)

        return dataset
