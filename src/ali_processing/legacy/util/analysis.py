from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def resolution_from_averaging_kernel(averaging_kernel: xr.Dataset) -> xr.DataArray:
    """
    Determine the full width half-maximum vertical resolution of the retrieval from the averaging kernel

    Parameters
    ----------
    A : xr.Dataset
        Averaging Kernel. Should have the dimension `altitude`

    Returns
    -------
    fwhm : xr.DataArray
        full width half max of the averaging kernel
    """

    A = averaging_kernel
    alts = A.altitude.values
    fwhm = np.ones_like(alts) * np.nan
    hires_alts = np.linspace(alts[0] * 0.9, alts[-1] * 1.1, len(alts) * 50)
    a_area = np.ones_like(alts) * np.nan
    for aidx, alt in enumerate(alts):
        An = np.interp(hires_alts, alts, A.sel(altitude=alt, method="nearest").values)
        try:
            A_max = np.max(An) + np.min(An)
            max_idx = np.argmax(An)
            lb = hires_alts[:max_idx][np.argmin(np.abs(An[:max_idx] - A_max / 2))]
            ub = hires_alts[max_idx:][np.argmin(np.abs(An[max_idx:] - A_max / 2))]
            fwhm[aidx] = ub - lb
            a_area[aidx] = np.trapz(A.sel(altitude=alt, method="nearest").values, alts)
        except:
            pass
    return xr.DataArray(fwhm, dims=["altitude"], coords=[A.altitude.values])


def encode_multiindex(ds, idxnames):
    """
    Encode a MultiIndexed dimension using the "compression by gathering" CF convention.

    https://cf-xarray.readthedocs.io/en/latest/generated/cf_xarray.encode_multi_index_as_compress.html#cf_xarray.encode_multi_index_as_compress

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with at least one MultiIndexed dimension
    idxnames : hashable or iterable of hashable, optional
        Dimensions that are MultiIndex-ed. If None, will detect all MultiIndex-ed dimensions.

    Returns
    -------
    xarray.Dataset
        Encoded Dataset with ``name`` as a integer coordinate with a ``"compress"`` attribute.

    References
    ----------
    CF conventions on `compression by gathering <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#compression-by-gathering>`_
    """
    if idxnames is None:
        # After the flexible indexes refactor, all MultiIndex Levels
        # have a MultiIndex but the name won't match.
        # Prior to that refactor, there is only a single MultiIndex with name=None
        idxnames = tuple(
            name
            for name, idx in ds.indexes.items()
            if isinstance(idx, pd.MultiIndex)
            and (idx.name == name if idx.name is not None else True)
        )
    elif isinstance(idxnames, str):
        idxnames = (idxnames,)

    if not idxnames:
        raise ValueError("No MultiIndex-ed dimensions found in Dataset.")

    encoded = ds.reset_index(idxnames)
    for idxname in idxnames:
        mindex = ds.indexes[idxname]
        coords = dict(zip(mindex.names, mindex.levels, strict=False))
        encoded.update(coords)
        for c in coords:
            encoded[c].attrs = ds[c].attrs
            encoded[c].encoding = ds[c].encoding
        encoded[idxname] = np.ravel_multi_index(mindex.codes, mindex.levshape)
        encoded[idxname].attrs = ds[idxname].attrs.copy()
        if (
            "compress" in encoded[idxname].encoding
            or "compress" in encoded[idxname].attrs
        ):
            raise ValueError(
                f"Does not support the 'compress' attribute in {idxname}.encoding or {idxname}.attrs. "
                "This is generated automatically."
            )
        encoded[idxname].attrs["compress"] = " ".join(mindex.names)
    return encoded


def decode_to_multiindex(encoded, idxnames):
    """
    Decode a compressed variable to a pandas MultiIndex.

    https://cf-xarray.readthedocs.io/en/latest/_modules/cf_xarray/coding.html#decode_compress_to_multi_index

    Parameters
    ----------
    encoded : xarray.Dataset
        Encoded Dataset with variables that use "compression by gathering".capitalize
    idxnames : hashable or iterable of hashable, optional
        Variable names that represents a compressed dimension. These variables must have
        the attribute ``"compress"``. If None, will detect all indexes with a ``"compress"``
        attribute and decode those.

    Returns
    -------
    xarray.Dataset
        Decoded Dataset with ``name`` as a MultiIndexed dimension.

    References
    ----------
    CF conventions on `compression by gathering <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#compression-by-gathering>`_
    """
    decoded = xr.Dataset(data_vars=encoded.data_vars, attrs=encoded.attrs.copy())
    if idxnames is None:
        idxnames = tuple(
            name for name in encoded.indexes if "compress" in encoded[name].attrs
        )
    elif isinstance(idxnames, str):
        idxnames = (idxnames,)

    for idxname in idxnames:
        if "compress" not in encoded[idxname].attrs:
            raise ValueError("Attribute 'compress' not found in provided Dataset.")

        if not isinstance(encoded, xr.Dataset):
            raise ValueError(
                f"Must provide a Dataset. Received {type(encoded)} instead."
            )

        names = encoded[idxname].attrs["compress"].split(" ")
        shape = [encoded.sizes[dim] for dim in names]
        indices = np.unravel_index(encoded[idxname].data, shape)
        try:
            from xarray.indexes import PandasMultiIndex

            variables = {
                dim: encoded[dim].isel({dim: xr.Variable(data=index, dims=idxname)})
                for dim, index in zip(names, indices, strict=False)
            }
            decoded = decoded.assign_coords(variables).set_xindex(
                names, PandasMultiIndex
            )
        except ImportError:
            arrays = [
                encoded[dim].data[index]
                for dim, index in zip(names, indices, strict=False)
            ]
            mindex = pd.MultiIndex.from_arrays(arrays, names=names)
            decoded.coords[idxname] = mindex

        decoded[idxname].attrs = encoded[idxname].attrs.copy()
        for coord in names:
            variable = encoded._variables[coord]
            decoded[coord].attrs = variable.attrs.copy()
            decoded[coord].encoding = variable.encoding.copy()
        del decoded[idxname].attrs["compress"]

    return decoded


def encode_multi_index_as_compress(ds, idxnames=None):
    """
    Encode a MultiIndexed dimension using the "compression by gathering" CF convention.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with at least one MultiIndexed dimension
    idxnames : hashable or iterable of hashable, optional
        Dimensions that are MultiIndex-ed. If None, will detect all MultiIndex-ed dimensions.

    Returns
    -------
    xarray.Dataset
        Encoded Dataset with ``name`` as a integer coordinate with a ``"compress"`` attribute.

    References
    ----------
    CF conventions on `compression by gathering <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#compression-by-gathering>`_
    """
    if idxnames is None:
        # After the flexible indexes refactor, all MultiIndex Levels
        # have a MultiIndex but the name won't match.
        # Prior to that refactor, there is only a single MultiIndex with name=None
        idxnames = tuple(
            name
            for name, idx in ds.indexes.items()
            if isinstance(idx, pd.MultiIndex)
            and (idx.name == name if idx.name is not None else True)
        )
    elif isinstance(idxnames, str):
        idxnames = (idxnames,)

    if not idxnames:
        raise ValueError("No MultiIndex-ed dimensions found in Dataset.")

    encoded = ds.reset_index(idxnames)
    for idxname in idxnames:
        mindex = ds.indexes[idxname]
        coords = dict(zip(mindex.names, mindex.levels, strict=False))
        encoded.update(coords)
        for c in coords:
            encoded[c].attrs = ds[c].attrs
            encoded[c].encoding = ds[c].encoding
        encoded[idxname] = np.ravel_multi_index(mindex.codes, mindex.levshape)
        encoded[idxname].attrs = ds[idxname].attrs.copy()
        if (
            "compress" in encoded[idxname].encoding
            or "compress" in encoded[idxname].attrs
        ):
            raise ValueError(
                f"Does not support the 'compress' attribute in {idxname}.encoding or {idxname}.attrs. "
                "This is generated automatically."
            )
        encoded[idxname].attrs["compress"] = " ".join(mindex.names)
    return encoded


def decode_compress_to_multi_index(encoded, idxnames=None):
    """
    Decode a compressed variable to a pandas MultiIndex.

    Parameters
    ----------
    encoded : xarray.Dataset
        Encoded Dataset with variables that use "compression by gathering".capitalize
    idxnames : hashable or iterable of hashable, optional
        Variable names that represents a compressed dimension. These variables must have
        the attribute ``"compress"``. If None, will detect all indexes with a ``"compress"``
        attribute and decode those.

    Returns
    -------
    xarray.Dataset
        Decoded Dataset with ``name`` as a MultiIndexed dimension.

    References
    ----------
    CF conventions on `compression by gathering <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#compression-by-gathering>`_
    """
    decoded = xr.Dataset(data_vars=encoded.data_vars, attrs=encoded.attrs.copy())
    if idxnames is None:
        idxnames = tuple(
            name for name in encoded.indexes if "compress" in encoded[name].attrs
        )
    elif isinstance(idxnames, str):
        idxnames = (idxnames,)

    for idxname in idxnames:
        if "compress" not in encoded[idxname].attrs:
            raise ValueError("Attribute 'compress' not found in provided Dataset.")

        if not isinstance(encoded, xr.Dataset):
            raise ValueError(
                f"Must provide a Dataset. Received {type(encoded)} instead."
            )

        names = encoded[idxname].attrs["compress"].split(" ")
        shape = [encoded.sizes[dim] for dim in names]
        indices = np.unravel_index(encoded[idxname].data, shape)
        try:
            from xarray.indexes import PandasMultiIndex

            variables = {
                dim: encoded[dim].isel({dim: xr.Variable(data=index, dims=idxname)})
                for dim, index in zip(names, indices, strict=False)
            }
            decoded = decoded.assign_coords(variables).set_xindex(
                names, PandasMultiIndex
            )
        except ImportError:
            arrays = [
                encoded[dim].data[index]
                for dim, index in zip(names, indices, strict=False)
            ]
            mindex = pd.MultiIndex.from_arrays(arrays, names=names)
            decoded.coords[idxname] = mindex

        decoded[idxname].attrs = encoded[idxname].attrs.copy()
        for coord in names:
            variable = encoded._variables[coord]
            decoded[coord].attrs = variable.attrs.copy()
            decoded[coord].encoding = variable.encoding.copy()
        del decoded[idxname].attrs["compress"]

    return decoded
