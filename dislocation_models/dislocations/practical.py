from typing import Union
import numpy as np
from dislocations.faults import Patch as FaultPatch
from dislocations.displacements import DisplacementTable, DisplacementGrid
from matplotlib import pyplot as plt
import datetime

fixed_y1 = 50.
fixed_y2 = -50.
fixed_x1 = 0.
fixed_x2 = 0.
fixed_top = 0.
fixed_bottom = -20.
fixed_rake = 90.
param_ranges = {
    "y1": (-100., 100.),
    "y2": (-100., 100.),
    "x1": (-100., 100.),
    "x2": (-100., 100.),
    "dip": (5., 85.),
    "top_depth": (-10., 0),
    "bottom_depth": (-20., -5.),
    "slip_magnitude": (0, 50.),
    "rake": (0., 360.),
}

param_units = {
    "y1": "km",
    "y2": "km",
    "x1": "km",
    "x2": "km",
    "dip": "degrees",
    "top_depth": "km",
    "bottom_depth": "km",
    "slip_magnitude": "m",
    "rake": "degrees"}

def randomize_parameters_and_write_to_pickle(pickle_file: str,
                                             y1: Union[float, int] = None, y2: Union[float, int] = None, 
                                                x1: Union[float, int] = None, x2: Union[float, int] = None,
                                                dip: Union[float, int] = None,
                                                top_depth: Union[float, int] = None, bottom_depth: Union[float, int] = None,
                                                slip_magnitude: Union[float, int] = None, rake: Union[float, int] = None):
    """
    Function to randomize parameters for a fault and write to a pickle file.
    :param pickle_file: Path to the pickle file to write to.
    :param y1: Y coordinate of the first point.
    :param y2: Y coordinate of the second point.
    :param x1: X coordinate of the first point.
    :param x2: X coordinate of the second point.
    :param dip: Dip angle of the fault.
    :param top_depth: Top depth of the fault.
    :param bottom_depth: Bottom depth of the fault.
    :param slip_magnitude: Slip magnitude of the fault.
    :param rake: Rake angle of the fault.
    :return: None
    """
    output_dict = {}
    # randomize bottom depth if not specified
    if bottom_depth is None:
        bottom_depth = np.random.uniform(param_ranges["bottom_depth"][0], param_ranges["bottom_depth"][1])
        output_dict["bottom_depth"] = bottom_depth
    else:
        # check if bottom depth is within the specified range
        assert param_ranges["bottom_depth"][0] <= bottom_depth <= param_ranges["bottom_depth"][1], \
            f"Bottom depth {bottom_depth} is out of range {param_ranges['bottom_depth']}"
        try:
            bottom_depth = float(bottom_depth)
        except ValueError:
            raise ValueError(f"Bottom depth {bottom_depth} is not a valid float")
        output_dict["bottom_depth"] = bottom_depth
    # randomize top depth if not specified
    if top_depth is None:
        min_top_depth = max(bottom_depth, param_ranges["top_depth"][0])
        max_top_depth = param_ranges["top_depth"][1]
        top_depth = np.random.uniform(min_top_depth, max_top_depth)
        output_dict["top_depth"] = top_depth
    else:
        # check if top depth is within the specified range
        assert param_ranges["top_depth"][0] <= top_depth <= param_ranges["top_depth"][1], \
            f"Top depth {top_depth} is out of range {param_ranges['top_depth']}"
        # check if top depth is greater than bottom depth
        assert top_depth > bottom_depth, f"Top depth {top_depth} must be greater than bottom depth {bottom_depth}"
        try:
            top_depth = float(top_depth)
        except ValueError:
            raise ValueError(f"Top depth {top_depth} is not a valid float")
        output_dict["top_depth"] = top_depth

    for param, value in zip(["y1", "y2", "x1", "x2"], [y1, y2, x1, x2]):
        if value is None:
            value = np.random.uniform(param_ranges[param][0], param_ranges[param][1])
            output_dict[param] = value
        else:
            # check if value is within the specified range
            assert param_ranges[param][0] <= value <= param_ranges[param][1], \
                f"{param} {value} is out of range {param_ranges[param]}"
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"{param} {value} is not a valid float")
            output_dict[param] = value
    
    # randomize dip if not specified
    if dip is None:
        dip = np.random.uniform(param_ranges["dip"][0], param_ranges["dip"][1])
        output_dict["dip"] = dip
    else:
        # check if dip is within the specified range
        assert param_ranges["dip"][0] <= dip <= param_ranges["dip"][1], \
            f"Dip {dip} is out of range {param_ranges['dip']}"
        try:
            dip = float(dip)
        except ValueError:
            raise ValueError(f"Dip {dip} is not a valid float")
        output_dict["dip"] = dip

    # randomize slip magnitude if not specified
    if slip_magnitude is None:
        slip_magnitude = np.random.uniform(0, 10)
        output_dict["slip_magnitude"] = slip_magnitude
    else:
        # check if slip magnitude is within the specified range
        assert slip_magnitude > 0, f"Slip magnitude {slip_magnitude} must be greater than 0"
        try:
            slip_magnitude = float(slip_magnitude)
        except ValueError:
            raise ValueError(f"Slip magnitude {slip_magnitude} is not a valid float")
        output_dict["slip_magnitude"] = slip_magnitude

    # randomize rake if not specified
    if rake is None:
        rake = np.random.uniform(param_ranges["rake"][0], param_ranges["rake"][1])
        output_dict["rake"] = rake

    else:
        # check if rake is within the specified range
        assert 0 <= rake < 360, f"Rake {rake} is out of range (0, 360)"
        try:
            rake = float(rake)
        except ValueError:
            raise ValueError(f"Rake {rake} is not a valid float")
        output_dict["rake"] = rake

    # write the output dictionary to a pickle file
    with open(pickle_file, "wb") as f:
        import pickle
        pickle.dump(output_dict, f)


def randomize_parameters_and_write_to_pickle_fixed_top_bottom(pickle_file: str, dip: Union[float, int] = None,                                                
                                                              slip_magnitude: Union[float, int] = None):
    """
    Function to randomize parameters for a fault with fixed top and bottom depth and write to a pickle file.
    :param pickle_file: Path to the pickle file to write to.
    :param dip: Dip angle of the fault.
    :param
    slip_magnitude: Slip magnitude of the fault.
    :return: None
    """
    randomize_parameters_and_write_to_pickle(pickle_file=pickle_file,
        y1=fixed_y1, y2=fixed_y2, x1=fixed_x1, x2=fixed_x2,
        top_depth=fixed_top, bottom_depth=fixed_bottom,
        rake=fixed_rake)
    
                                                              


def read_parameters_from_pickle(pickle_file: str) -> dict:
    """
    Function to read parameters from a pickle file.
    :param pickle_file: Path to the pickle file to read from.
    :return: Dictionary of parameters.
    """
    with open(pickle_file, "rb") as f:
        import pickle
        params = pickle.load(f)
    # check if all required parameters are present
    required_params = ["y1", "y2", "x1", "x2", "dip", "top_depth", "bottom_depth", "slip_magnitude", "rake"]
    for param in required_params:
        if param not in params:
            raise ValueError(f"Parameter {param} is missing from the pickle file")
    return params


def profile_2d_displacement(min_x: float = -100., max_x: float = 100., x_inc: float = 1., top_depth: float = 0.,
    bottom_depth: float = -20., dip: float = 45., slip_magnitude: float = 10., rake: float = 90.) -> np.ndarray:
    
    x_vals = np.arange(min_x, max_x + x_inc, x_inc)
    x_vals_m = np.column_stack([x_vals * 1000.0, np.zeros_like(x_vals)])  # convert to meters

    y1, y2, x1, x2 = 1000. * np.array([fixed_y1, fixed_y2, fixed_x1, fixed_x2])
    top_depth, bottom_depth = 1000. * np.array([top_depth, bottom_depth])
    fault_patch = FaultPatch.from_top_endpoints(y1=y1, y2=y2, x1=x1, x2=x2,
        dip=dip, top_z=top_depth, bottom_z=bottom_depth)
    displacement_table = DisplacementTable.from_xy_array(fault_patch, x_vals_m)
    displacements = slip_magnitude * displacement_table.greens_functions_array(rake=rake, vertical_only=True).flatten()
    displacements[x_vals == 0.] = np.nan  # set displacements to zero at the origin
    return x_vals, displacements


def practical_general_2d_generic(
    y1: Union[float, int], y2: Union[float, int], x1: Union[float, int], x2: Union[float, int],
    dip: Union[float, int], top_depth: Union[float, int], bottom_depth: Union[float, int],
    slip_magnitude: Union[float, int], rake: Union[float, int], pickle_file: str = None,
    min_x: float = -100., max_x: float = 100., x_inc: float = 1.) -> np.ndarray:
    """
    Function to calculate the dislocation displacement for a given set of parameters.
    :param y1: Y coordinate of the first point.
    :param y2: Y coordinate of the second point.
    :param x1: X coordinate of the first point.
    :param x2: X coordinate of the second point.
    :param dip: Dip angle of the fault.
    :param top_depth: Top depth of the fault.
    :param bottom_depth: Bottom depth of the fault.
    :param slip_magnitude: Slip magnitude of the fault.
    :param rake: Rake angle of the fault.
    :param pickle_file: Path to the pickle file to read parameters from.
    :param grid: DisplacementGrid object to calculate displacements on a grid.
    :param table: DisplacementTable object to calculate displacements at specific points.
    :param fault_patch: FaultPatch object to represent the fault patch.
    :param min_x: Minimum x coordinate for the profile.
    :param max_x: Maximum x coordinate for the profile.
    :param x_inc: Increment for the x coordinate.
    :return: Displacement array or table depending on input parameters.

    """
    
    # Check if all required parameters are provided
    if any(param is None for param in [y1, y2, x1, x2, dip, top_depth, bottom_depth, slip_magnitude, rake]):
        raise ValueError("All parameters must be provided")
    
    # read parameters from pickle file if provided
    if pickle_file is not None:
        params = read_parameters_from_pickle(pickle_file)

    # convert to m from km
    y1, y2, x1, x2, top_depth, bottom_depth = 1000. * np.array([y1, y2, x1, x2, top_depth, bottom_depth])
    for param in ["y1", "y2", "x1", "x2", "top_depth", "bottom_depth"]:
        if param in params:
            params[param] = 1000. * params[param]

        

    # create fault patch
    fault_patch_user = FaultPatch.from_top_endpoints(
        y1=y1, y2=y2, x1=x1, x2=x2, dip=dip, top_z=top_depth, bottom_z=bottom_depth)
    if pickle_file is not None:
        fault_patch_dict = FaultPatch.from_top_endpoints(y1=params["y1"], y2=params["y2"],
            x1=params["x1"], x2=params["x2"], dip=params["dip"], top_z=params["top_depth"],
            bottom_z=params["bottom_depth"])
    
    disp_x = np.arange(min_x, max_x + x_inc, x_inc)
    disp_x_m = np.column_stack([disp_x * 1000.0, np.zeros_like(disp_x)])  # convert to meters

    # create displacement tables
    user_table = DisplacementTable.from_xy_array(fault_patch_user, disp_x_m)
    if pickle_file is not None:
        pickle_table = DisplacementTable.from_xy_array(fault_patch_dict, disp_x_m)

    user_disps = slip_magnitude * user_table.greens_functions_array(rake=rake, vertical_only=True).flatten()
    user_disps[disp_x == 0.] = 0.0  # set displacements to zero at the origin
    if pickle_file is not None:
        pickle_disps = params["slip_magnitude"] * pickle_table.greens_functions_array(rake=params["rake"], vertical_only=True).flatten()
        pickle_disps[disp_x == 0.] = 0.0  # set displacements to zero at the origin
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(disp_x, user_disps, label="Your Displacement", color='orange')
    if pickle_file is not None:
        ax.plot(disp_x, pickle_disps, label="Target Displacement", color='black')

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Displacement (m)")

    max_y = np.max(np.abs(np.hstack([user_disps, pickle_disps]))) * 1.2
    ax.set_ylim(-max_y, max_y)
    ax.set_xlim(min_x, max_x)
    ax.legend()
    # Print the current timestamp
    print("Timestamp:", datetime.datetime.now())

def read_and_print_parameters(pickle_file: str, in3d: bool = False) -> None:
    """
    Function to read parameters from a pickle file and print them.
    :param pickle_file: Path to the pickle file to read from.
    :return: None
    """
    print('WARNING: running this cell will display the answer:')
    answer = input('-> Are you sure you want to run it? [yes| no]: ' )
    if answer.lower() == 'yes':
        params = read_parameters_from_pickle(pickle_file)
        if in3d:
            for param, value in params.items():
                print(f"{param}: {value} {param_units[param]}")
        else:
            for param, value in params.items():
                if param not in ["y1", "y2", "x1", "x2"]:
                    print(f"{param}: {value} {param_units[param]}")
        # Print the current timestamp
        print("Timestamp:", datetime.datetime.now())


def practical_general_2d_fixed_top_only(
    bottom_depth: Union[float, int], dip: Union[float, int], 
    slip_magnitude: Union[float, int], pickle_file: str = None,
    min_x: float = -100., max_x: float = 100., x_inc: float = 1.) -> np.ndarray:
    """
    Function to calculate the dislocation displacement for a given set of parameters.
    :param y1: Y coordinate of the first point.
    :param y2: Y coordinate of the second point.
    :param x1: X coordinate of the first point.
    :param x2: X coordinate of the second point.
    :param dip: Dip angle of the fault.
    :param top_depth: Top depth of the fault.
    :param bottom_depth: Bottom depth of the fault.
    :param slip_magnitude: Slip magnitude of the fault.
    :param rake: Rake angle of the fault.
    :param pickle_file: Path to the pickle file to read parameters from.
    :param grid: DisplacementGrid object to calculate displacements on a grid.
    :param table: DisplacementTable object to calculate displacements at specific points.
    :param fault_patch: FaultPatch object to represent the fault patch.
    :param min_x: Minimum x coordinate for the profile.
    :param max_x: Maximum x coordinate for the profile.
    :param x_inc: Increment for the x coordinate.
    :return: Displacement array or table depending on input parameters.

    """
    
    # Check if all required parameters are provided
    if any(param is None for param in [dip, bottom_depth, slip_magnitude]):
        raise ValueError("All parameters must be provided")
    
    # read parameters from pickle file if provided
    if pickle_file is not None:
        params = read_parameters_from_pickle(pickle_file)

    practical_general_2d_generic(y1=fixed_y1, y2=fixed_y2, x1=fixed_x1, x2=fixed_x2,
        dip=dip, top_depth=fixed_top, bottom_depth=bottom_depth,
        slip_magnitude=slip_magnitude, rake=fixed_rake, pickle_file=pickle_file,
        min_x=min_x, max_x=max_x, x_inc=x_inc)
    




    
    