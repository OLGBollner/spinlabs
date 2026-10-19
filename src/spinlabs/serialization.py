"""Generic npz save/load for the spinlabs dataclasses.

Each dataclass declares a SCHEMA: a list of FieldSpec entries describing
its fields (kind, optional npz key, symbolic shape, optionality, legacy
aliases, optional encode/decode hooks). One generic serializer uses the
schema for both directions:

* save: None-valued optional fields are OMITTED from the file (never
  written as 0-d object arrays); every legacy alias key is written with
  the same value.
* load: missing optional fields come back as None; a 0-d object array
  holding None (legacy files) is normalized to None; missing required
  fields raise SchemaError.
* load: array kind and symbolic shapes are validated. Shape strings
  ("n_modes", ...) must be consistent across fields of one object.
* load: when the file carries keys not in the schema, a warning is
  emitted (forward compatibility: old readers still open new files;
  new readers tell you when a file has unknown content).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union
import warnings
from pathlib import Path

import numpy as np


class SchemaError(ValueError):
    """Raised when a file does not match the declared schema."""


@dataclass(frozen=True)
class FieldSpec:
    name: str  # dataclass attribute name
    kind: str  # array|int|float|str|bool|dict|list_str|list_bool|zfstensor
    npz_key: str = None  # key in the file (defaults to name)
    shape: Tuple = ()  # symbolic shape; ints fixed, strings matched across fields
    optional: bool = False  # None allowed; key omitted on save when None
    legacy_keys: Tuple[str, ...] = ()  # fallback npz keys on load
    save_aliases: Tuple[str, ...] = ()  # extra npz keys written with the same value
    encode: Callable[[Any], Any] = None  # attribute -> file value (default identity)
    decode: Callable[[Any, Any], Any] = None  # (file value, whole npz) -> attribute (default identity)

    @property
    def key(self) -> str:
        return self.npz_key if self.npz_key is not None else self.name


def _npz_value(value: Any) -> Any:
    """Make a value storable by np.savez.

    np.savez (numpy >= 1.24) refuses ragged sequences (lists of
    variable-length lists, e.g. deg_groups) with ValueError instead of
    storing them. Convert such values to object arrays; object arrays
    are written and read back faithfully (with allow_pickle=True).
    """
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except ValueError:
        arr = np.empty(len(value), dtype=object)
        arr[:] = [np.asarray(v) for v in value]
        return arr


def _unwrap(value: Any) -> Any:
    """Normalize np.load artefacts: a 0-d object array holding None -> None."""
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        inner = value.item()
        if inner is None:
            return None
    return value


_KINDS = {
    "array",
    "int",
    "float",
    "str",
    "bool",
    "dict",
    "list_str",
    "list_bool",
    "list_int",
    "zfstensor",
}


def _check_shape(kind: str, arr: np.ndarray, spec: FieldSpec, dims: Dict[str, int]) -> Optional[np.ndarray]:
    if kind != "array" or not spec.shape:
        return arr
    if arr.ndim != len(spec.shape):
        raise SchemaError(
            f"field '{spec.name}': expected {len(spec.shape)}-d array with shape {spec.shape}, got shape {arr.shape}"
        )
    for axis, dim in enumerate(spec.shape):
        if isinstance(dim, str):
            if dim in dims and dims[dim] != arr.shape[axis]:
                raise SchemaError(
                    f"field '{spec.name}': dim '{dim}' inconsistent: expected {dims[dim]}, got {arr.shape[axis]}"
                )
            dims[dim] = arr.shape[axis]
        elif arr.shape[axis] != dim:
            raise SchemaError(f"field '{spec.name}': expected shape {spec.shape}, got {arr.shape}")
    return arr


def _validate_value(spec: FieldSpec, value: Any, dims: Dict[str, int]) -> Any:
    kind = spec.kind
    if value is None:
        return None
    if kind == "array":
        if isinstance(value, np.ndarray) and value.dtype == object:
            return value
        try:
            arr = np.asarray(value)
        except ValueError:
            return value  # ragged stored object array; shape unchecked
        return _check_shape(kind, arr, spec, dims)
    if kind == "int":
        if isinstance(value, (int, np.integer)):
            return int(value)
        raise SchemaError(f"field '{spec.name}': expected int, got {type(value).__name__}")
    if kind == "float":
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        raise SchemaError(f"field '{spec.name}': expected float, got {type(value).__name__}")
    if kind == "str":
        if isinstance(value, str):
            return value
        raise SchemaError(f"field '{spec.name}': expected str, got {type(value).__name__}")
    if kind == "bool":
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        raise SchemaError(f"field '{spec.name}': expected bool, got {type(value).__name__}")
    if kind in ("dict", "list_str", "list_bool", "list_int"):
        return value
    if kind == "zfstensor":
        mat = getattr(value, "matrix", None)
        if mat is None or np.asarray(mat).shape != (3, 3):
            raise SchemaError(f"field '{spec.name}': ZFSTensor matrix must be 3x3, got {np.shape(mat)}")
        return value
    raise SchemaError(f"field '{spec.name}': unknown kind '{kind}'")


def save_npz(
    obj: Any,
    path: Union[str, Path],
    schema: Sequence[FieldSpec],
    overrides: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    """Serialize obj to .npz according to schema. Returns the path written."""
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {}
    for spec in schema:
        if overrides and spec.name in overrides:
            value = overrides[spec.name]
        else:
            value = getattr(obj, spec.name)
        if spec.encode is not None:
            value = spec.encode(value)
        if value is None:
            if not spec.optional:
                raise SchemaError(f"field '{spec.name}' is required but None")
            continue  # omitted from the file entirely
        if spec.kind == "zfstensor":
            from .models import ZFSTensor  # lazy: serialization <-> models cycle

            if not isinstance(value, ZFSTensor):
                raise SchemaError(f"field '{spec.name}': expected ZFSTensor, got {type(value).__name__}")
            payload[f"{spec.key}_matrix"] = value.matrix
            payload[f"{spec.key}_unit"] = value.unit
            continue
        value = _npz_value(value)
        payload[spec.key] = value
        for alias in spec.save_aliases:
            payload[alias] = value

    if extras:
        payload.update(extras)

    # Compressed: the dense N x N second-order coupling arrays are mostly zeros
    # (NV_512: 226 MB -> 0.4 MB). np.load reads compressed and plain files alike.
    np.savez_compressed(path, **payload)
    return path


def _zfstensor_from(data: Any, key: str) -> Any:
    from .models import ZFSTensor  # lazy: serialization <-> models cycle

    mk = f"{key}_matrix"
    if mk in data.files:
        return ZFSTensor(matrix=data[mk], unit=str(data.get(f"{key}_unit", "MHz")))
    # legacy: tensor stored as a single array under the field key itself
    return ZFSTensor(matrix=np.asarray(data[key]), unit="MHz")


def _fields_from_npz(
    data: Any,
    cls: Type,
    schema: Sequence[FieldSpec],
    strict: bool = True,
) -> Dict[str, Any]:
    """Parse one np.load object into field values according to schema."""
    present = set(data.files)
    for spec in schema:
        for alias in spec.save_aliases:
            present.discard(alias)

    kwargs: Dict[str, Any] = {}
    dims: Dict[str, int] = {}

    for spec in schema:
        raw = None
        mk = f"{spec.key}_matrix"
        if spec.kind == "zfstensor" and mk in present:
            # tensor fields are stored as <key>_matrix / <key>_unit subkeys
            found = True
            raw = None
        else:
            found = spec.key in present
            if not found:
                for lk in spec.legacy_keys:
                    if lk in present:
                        raw = data[lk]
                        found = True
                        break
            else:
                raw = data[spec.key]

        if not found:
            if spec.optional:
                kwargs[spec.name] = None
            elif strict:
                raise SchemaError(f"{cls.__name__}.load: missing required field '{spec.key}'")
            continue

        if spec.kind == "zfstensor":
            if "zfs_relaxed" in present and mk not in present:
                from spinlabs.models import ZFSTensor  # lazy: models imports this module

                value = ZFSTensor(matrix=data["zfs_relaxed"], unit="J")  # legacy: Joules
            else:
                value = _zfstensor_from(data, spec.key)
        else:
            value = _unwrap(raw)
            if value is None:
                if spec.optional:
                    kwargs[spec.name] = None
                    continue
                if strict:
                    raise SchemaError(f"{cls.__name__}.load: required field '{spec.key}' is None in file")
                continue

            # ragged object arrays (e.g. deg_groups) read back as lists
            if isinstance(value, np.ndarray) and value.dtype == object and value.ndim >= 1:
                value = value.tolist()

            # kind conversion
            if spec.kind == "int":
                value = int(value)
            elif spec.kind == "float":
                value = float(value)
            elif spec.kind == "str":
                value = str(value)
            elif spec.kind == "bool":
                value = bool(value)
            elif spec.kind == "list_str":
                value = list(str(x) for x in value)
            elif spec.kind == "list_bool":
                value = list(bool(x) for x in value)
            elif spec.kind == "list_int":
                value = list(int(x) for x in value)
            elif spec.kind == "dict":
                value = value[()] if isinstance(value, np.ndarray) and value.shape == () else value

        value = _validate_value(spec, value, dims)

        if spec.decode is not None:
            value = spec.decode(value, data)

        kwargs[spec.name] = value

    return kwargs


def load_npz(
    cls: Type,
    in_path: Union[str, Path],
    schema: Sequence[FieldSpec],
    pre_decode: Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]] = None,
    post_decode: Optional[Callable[[Any, Any], Any]] = None,
) -> Any:
    """Reconstruct cls from an .npz file according to schema.

    pre_decode(data, kwargs) may rewrite the whole kwargs dict before
    instantiation (legacy remaps); post_decode(obj, data) may adapt the
    final object (returns the instance).
    """
    data = np.load(str(in_path), allow_pickle=True)
    kwargs = _fields_from_npz(data, cls, schema)
    if pre_decode is not None:
        kwargs = pre_decode(data, kwargs)
    obj = cls(**kwargs)
    if post_decode is not None:
        obj = post_decode(obj, data)
    return obj


def warn_unknown_keys(
    data_files: Sequence[str],
    schema: Sequence[FieldSpec],
    extra_ignore: Sequence[str] = (),
) -> None:
    """Warn about keys present in the file but absent from the schema."""
    known = set(extra_ignore)
    for spec in schema:
        known.add(spec.key)
        known.update(spec.legacy_keys)
        known.update(spec.save_aliases)
        if spec.kind == "zfstensor":
            known.add(f"{spec.key}_matrix")
            known.add(f"{spec.key}_unit")
    unknown = [k for k in data_files if k not in known]
    if unknown:
        warnings.warn(
            f"unknown keys in file (newer schema or hand-made file): {unknown}",
            stacklevel=3,
        )
