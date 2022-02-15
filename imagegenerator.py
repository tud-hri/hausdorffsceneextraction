import sys
import os
import datetime

from processing.encryptiontools import load_encrypted_pickle, save_encrypted_pickle
from dataobjects.dataset import Dataset
from PyQt5 import QtWidgets, QtCore
from gui import TrafficVisualizerGui
from gui import SimMaster


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

    gui = TrafficVisualizerGui(data)
    sim = SimMaster(data, gui)
    gui.register_sim_master(sim)

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
        gui.set_frame_number(frame_number)

        if ego_ids[index]:
            if not (gui.selected_car and gui.selected_car.id == ego_ids[index]):
                gui.select_car(gui.cars[str(ego_ids[index])])

        file_name = file_names[index]
        if not file_name:
            file_name = str(datetime.datetime.now().timestamp())

        gui.get_image_of_current_view().save(os.path.join(sub_folder, file_name + '.png'))

    gui.close()
