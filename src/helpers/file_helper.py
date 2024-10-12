import os

def get_filepath(filename, sibiling_folder='datasets'):
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, '..', sibiling_folder, filename)
    file_path = os.path.abspath(file_path)
    return file_path