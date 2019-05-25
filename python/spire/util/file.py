import os
import shutil

#
# General file management
#

# Makes the directory if it does not exist.
#
# dir: str
def ensure_dir(dir):
    if not exists_dir(dir):
        make_dir(dir)

# Clears the directory if it exists, and then remakes it.
#
# dir: str
def clear_and_make_dir(dir):
    if exists_dir(dir):
        clear_dir(dir)
    make_dir(dir)

# Checks if the file exists.
#
# dir: str
# fname: str
# return: bool
def exists_file(dir, fname):
    return os.path.isfile('{}/{}'.format(dir, fname))

# Returns whether the directory already exists.
#
# dir: str
# return: bool
def exists_dir(dir):
    return os.path.isdir(dir)

# Clears the directory.
#
# dir: str
def clear_dir(dir):
    shutil.rmtree(dir, ignore_errors=True)

# Makes the directory.
#
# dir: str
def make_dir(dir):
    os.makedirs(dir)
