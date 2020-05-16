import pickle
from pathlib import Path
import os


class Cache(object):
    def __init__(self, destination_folder, filename, enabled=False):
        self.destination_folder = destination_folder
        self.filename = filename
        self.enabled = enabled

    def __call__(self, function):
        def wrapped_function(*args, **kwargs):
            if self.enabled is True:
                destination_file = os.path.join(self.destination_folder, self.filename)
                # get result from destination file
                if os.path.exists(destination_file):
                    return pickle.load(open(destination_file, "rb"))
                else:
                    Path(self.destination_folder).mkdir(parents=True, exist_ok=True)
                    pickle_file = open(destination_file, "wb")
                    result = function(*args, **kwargs)
                    pickle.dump(result, pickle_file)
                    pickle_file.close()
                    return result
            else:
                return function(*args, **kwargs)

        return wrapped_function
