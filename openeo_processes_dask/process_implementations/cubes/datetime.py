import numpy as np
import xarray as xr

def sin_cos_doy(data: xr.DataArray) -> xr.DataArray:
    time_dim = data.coords.get("t") or data.coords.get("time")
    if time_dim is None:
        raise ValueError("Time coordinate 't' or 'time' not found")

    time_dim_name = time_dim.dims[0]
    doy = time_dim.dt.dayofyear
    is_leap = time_dim.dt.is_leap_year
    angle = 2 * np.pi * (doy - 1) / xr.where(is_leap, 366, 365)

    sin_vals = np.sin(angle)
    cos_vals = np.cos(angle)

    # Build base arrays with time dimension
    sin_doy = xr.DataArray(sin_vals, coords={time_dim_name: time_dim}, dims=[time_dim_name])
    cos_doy = xr.DataArray(cos_vals, coords={time_dim_name: time_dim}, dims=[time_dim_name])

    # Broadcast to input data dims except bands
    dims_to_expand = [dim for dim in data.dims if dim != "bands"]
    sin_doy = sin_doy.broadcast_like(data.isel(bands=0).drop_vars("bands"))
    cos_doy = cos_doy.broadcast_like(data.isel(bands=0).drop_vars("bands"))

    # Now expand to add bands dim as dim_0
    sin_doy = sin_doy.expand_dims("bands")
    cos_doy = cos_doy.expand_dims("bands")

    # Assign band names for sin and cos
    sin_doy = sin_doy.assign_coords(bands=["sin_doy"])
    cos_doy = cos_doy.assign_coords(bands=["cos_doy"])

    # Concatenate along bands dim
    result = xr.concat([sin_doy, cos_doy], dim="bands")

    return result

