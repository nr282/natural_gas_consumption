import os


def get_base_path():

    potential_paths = ["C:\\Users\\nr282\\PycharmProjects\\PythonProject4",
                       "/home/ec2-user/natural_gas_consumption"]

    actual_path = None
    for pot_path in potential_paths:
        if os.path.exists(pot_path):
            actual_path = pot_path
            break

    if actual_path is None:
        raise ValueError("Could not find base path")

    return actual_path