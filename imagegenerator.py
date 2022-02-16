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

import sys
import os
import datetime

from processing.encryptiontools import load_encrypted_pickle, save_encrypted_pickle
from dataobjects.dataset import Dataset
from PyQt5 import QtWidgets, QtCore
from gui import TrafficVisualizerGui
from visualisation import HighDVisualisationMaster


def generate_images(dataset_index, frame_ids, path_to_data_folder, path_to_context_data, tag, ego_ids=None, file_names=None):
    try:
        [i for i in frame_ids]
    except TypeError:
        frame_ids = [frame_ids]

    if ego_ids is not None:
        if len(frame_ids) != len(ego_ids):
            raise ValueError("The number of provided frame ID's should be equal to the number of provided ego ID's")
    if file_names is not None:
        if len(frame_ids) != len(file_names):
            raise ValueError("The number of provided file names should be equal to the number of provided ego ID's")

    app = QtWidgets.QApplication(sys.argv)

    data = load_encrypted_pickle(os.path.join(path_to_data_folder, '%02d.pkl' % dataset_index))

    if data is None:  # pickle file was not present
        data = Dataset.from_csv_files(dataset_index)
        save_encrypted_pickle(os.path.join(path_to_data_folder, '%02d.pkl' % dataset_index), data)

    old_wd = os.getcwd()
    new_wd = os.getcwd()

    while os.path.split(new_wd)[1] != 'travia':
        new_wd, _ = os.path.split(new_wd)

    os.chdir(new_wd)
    gui = TrafficVisualizerGui(data)
    start_time = data.start_time
    end_time = start_time + datetime.timedelta(milliseconds=int(data.duration * 1000))
    first_frame = data.track_data['frame'].min()
    number_of_frames = data.track_data['frame'].max() - first_frame
    dt = datetime.timedelta(seconds=1 / data.frame_rate)
    sim = HighDVisualisationMaster(data, gui, start_time, end_time, number_of_frames, first_frame, dt)
    gui.register_visualisation_master(sim)

    os.chdir(old_wd)

    export_timer = QtCore.QTimer()
    export_timer.setInterval(0)
    export_timer.setSingleShot(True)
    export_timer.timeout.connect(lambda: _save_images_and_close(frame_ids, ego_ids, gui, file_names, path_to_context_data, tag))
    export_timer.start()
    app.exec()


def _save_images_and_close(frame_ids, ego_ids, gui, file_names, folder, tag):
    sub_folder = os.path.join(folder, 'images', tag)

    os.makedirs(sub_folder, exist_ok=True)

    if file_names is None:
        file_names = [None] * len(frame_ids)
    if ego_ids is None:
        ego_ids = [None] * len(frame_ids)

    for index, frame_number in enumerate(frame_ids):
        gui.visualisation_master.frame_number = frame_number
        gui.visualisation_master.do_time_step()

        if ego_ids[index]:
            if not (gui.selected_vehicle and gui.selected_vehicle.id == ego_ids[index]):
                gui.select_vehicle(gui.vehicles[str(ego_ids[index])])

        file_name = file_names[index]
        if not file_name:
            file_name = str(datetime.datetime.now().timestamp())

        gui.get_image_of_current_view().save(os.path.join(sub_folder, file_name + '.png'))

    gui.close()
