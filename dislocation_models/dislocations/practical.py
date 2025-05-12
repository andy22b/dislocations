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
    "x1": (-50., 50.),
    "x2": (-50., 50.),
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
                                                slip_magnitude: Union[float, int] = None, rake: Union[float, int] = None,
                                                x1_eq_x2: bool = False) -> None:
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
    if x1_eq_x2:
        output_dict["x1"] = output_dict["x2"]
    
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
    

def randomize_parameters_and_write_to_pickle_2d(pickle_file: str):
    """
    Function to randomize parameters for a fault and write to a pickle file.
    :param pickle_file: Path to the pickle file to write to.
    :return: None
    """
    rake = np.random.choice([90., 270.])
    randomize_parameters_and_write_to_pickle(pickle_file=pickle_file,
        y1=fixed_y1, y2=fixed_y2, x1_eq_x2=True, rake=rake)                              


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
    min_x: float = -100., max_x: float = 100., x_inc: float = 1., data_only: bool = False) -> np.ndarray:
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
    if pickle_file is not None:
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
    if not data_only:
        user_disps[disp_x == 0.] = 0.0  # set displacements to zero at the origin
    if pickle_file is not None:
        pickle_disps = params["slip_magnitude"] * pickle_table.greens_functions_array(rake=params["rake"], vertical_only=True).flatten()
        if not data_only:
            pickle_disps[disp_x == 0.] = 0.0  # set displacements to zero at the origin
    
    if data_only:
        return disp_x, user_disps
    
    else:
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
    :param x2: X coordinate of the first point.
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
    

tt_dip = 12.
tt_slip = 10.
tt_rake = 90.
tt_min_x = -150.
tt_max_x = 150.
tt_x_inc = 1.
depth_x0 = -30.
subduction_top_x = (-1. * depth_x0) / np.tan(np.radians(tt_dip))

min_tt_y = -50.
max_tt_y = -10.

max_disp_y = 5.
min_disp_y = -5.

def sz_depth_x(depth: float) -> float:
    """
    Function to calculate the x coordinate for a given depth.
    :param depth: Depth of the fault.
    :return: X coordinate for the given depth.
    """
    return subduction_top_x + depth / np.tan(np.radians(tt_dip))



def tremor_trigger(bottom_depth: float = -50., top_depth: float = -10.):
    """
    Function to calculate the dislocation displacement for a given set of parameters.
    :param bottom_depth: Bottom depth of the fault.
    :param top_depth: Top depth of the fault.
    :return: Displacement array or table depending on input parameters.
    """
    
    # Check if all required parameters are provided
    if any(param is None for param in [bottom_depth, top_depth]):
        raise ValueError("Make sure you provide both top and bottom depth")
    
    if any([param > 0. for param in [bottom_depth, top_depth]]):
        raise ValueError("Make sure you provide negative values for both top and bottom depth")
    
    for name, param in zip(["top_depth", "bottom_depth"], [top_depth, bottom_depth]):
        if not min_tt_y <= param <= max_tt_y:
            raise ValueError(f"Make sure you provide a {name} between {min_tt_y} and {max_tt_y}")
        
    
    if bottom_depth >= top_depth:
        raise ValueError("Make sure you provide a bottom depth deeper than top depth")
    

    
    # read parameters from pickle file if provided
    top_x = sz_depth_x(top_depth)
    bottom_x = sz_depth_x(bottom_depth)
    xvals, disps = practical_general_2d_generic(y1=fixed_y1, y2=fixed_y2, x1=top_x, x2=top_x,
                                                    dip=tt_dip, top_depth=top_depth, bottom_depth=bottom_depth,
                                                    slip_magnitude=tt_slip, rake=tt_rake, pickle_file=None,
                                                    min_x=tt_min_x, max_x=tt_max_x, x_inc=tt_x_inc, data_only=True)
    
    disp0 = disps[xvals == 0.][0]
    


    # plot the displacements
    plt.close('all')
    fig, ax = plt.subplots(2, 1,figsize=(10, 5), sharex=True)
    ax[0].vlines(x=0., color='black', linestyle='--', ymin=min_disp_y, ymax=max_disp_y)
    ax[0].set_ylim(min_disp_y, max_disp_y)
    ax[0].plot(xvals, disps, label="Uplift", color='red')
    ax[0].plot(xvals[disps < 0.], disps[disps < 0.], color='blue', label="Subsidence")
    ax[0].set_xlim(tt_min_x, tt_max_x)
    ax[0].text(x=0., y=4., s="Madeup Island", ha='center', va='top', fontsize=12,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    text_string = f"{disp0:.2f} m Uplift at Madeup Island" if disp0 >= 0. else f"{-disp0:.2f} m Subsidence at Madeup Island"
    ax[0].text(x=145., y=-4.5, s=text_string, ha='right', va='bottom', fontsize=12,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    ax[0].scatter(x=0., y=disp0, color='black', marker='o', s=30)
    
    ax[1].set_xlabel("Distance from Madeup Island (km)")
    #ax[0].set_ylabel("Modelled displacement (m)")
    ax[0].legend()

    ax[1].vlines(x=0., color='black', linestyle='--', ymin=min_tt_y - 10., ymax=0.)
    ax[1].set_ylim(min_tt_y - 10., 0.)
    ax[1].plot([sz_depth_x(min_tt_y), sz_depth_x(max_tt_y)], [min_tt_y, max_tt_y], color='0.7', linestyle='-')
    ax[1].plot([sz_depth_x(bottom_depth), sz_depth_x(top_depth)], [bottom_depth, top_depth], color='black', linestyle='-', lw=3)
    ax[0].set_ylabel("Modelled displacement (m)")
    ax[1].set_ylabel("SZ Depth (km)")
    
    return xvals, disps


def practical_general_2d_no_fixed_top(top_x: float = 0., top_depth: float = 0., bottom_depth: float = -20.,
    dip: float = 45., slip_magnitude: float = 10., rake: float = 90., pickle_file: str = None,
    min_x: float = -100., max_x: float = 100., x_inc: float = 1.) -> np.ndarray:
    """
    Function to calculate the dislocation displacement for a given set of parameters.
    :param y1: Y coordinate of the first point.
    :param y2: Y coordinate of the second point.
    :param x1: X coordinate of the first point.
    :param x2: X coordinate of the first point.
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
    if any(param is None for param in [top_x, top_depth, bottom_depth, dip, slip_magnitude, rake]):
        raise ValueError("All parameters must be provided")
    
    assert top_depth > bottom_depth, f"Top depth {top_depth} must be greater than bottom depth {bottom_depth}"
    assert rake in [90, 270.], f"Rake {rake} must be either 90 (reverse) or 270 (normal) degrees"

    practical_general_2d_generic(y1=fixed_y1, y2=fixed_y2, x1=top_x, x2=top_x,
        dip=dip, top_depth=top_depth, bottom_depth=bottom_depth,
        slip_magnitude=slip_magnitude, rake=rake, pickle_file=pickle_file,
        min_x=min_x, max_x=max_x, x_inc=x_inc)















