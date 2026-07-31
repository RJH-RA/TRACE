from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CTPreprocessConfig:
    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 3.0)
    roi_shape_voxels: tuple[int, int, int] = (16, 128, 128)
    hu_window: tuple[float, float] = (-150.0, 350.0)


def _sitk():
    try:
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("CT preprocessing requires SimpleITK") from exc
    return sitk


def resample(
    image,
    spacing: tuple[float, float, float],
    interpolator,
    reference=None,
):
    sitk = _sitk()
    if reference is not None:
        return sitk.Resample(
            image,
            reference,
            sitk.Transform(),
            interpolator,
            0.0,
            image.GetPixelID(),
        )
    old_spacing, old_size = image.GetSpacing(), image.GetSize()
    new_size = [
        round(old_size[index] * old_spacing[index] / spacing[index])
        for index in range(3)
    ]
    transform = sitk.Transform()
    return sitk.Resample(
        image,
        new_size,
        transform,
        interpolator,
        image.GetOrigin(),
        spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def rigid_register(moving, fixed):
    sitk = _sitk()
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.05)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial, inPlace=False)
    transform = registration.Execute(fixed, moving)
    registered = sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, -1024.0, moving.GetPixelID()
    )
    return registered, transform


def scale_hu(array: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lower, upper = window
    clipped = np.clip(np.asarray(array, dtype=np.float32), lower, upper)
    return ((clipped - lower) / (upper - lower)).astype(np.float32)


def crop_around_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    lesion = np.asarray(mask) > 0
    center = (
        np.round(np.argwhere(lesion).mean(axis=0)).astype(int)
        if lesion.any()
        else np.asarray(volume.shape) // 2
    )
    crop = np.zeros(shape, dtype=np.float32)
    source, target = [], []
    for axis, length in enumerate(shape):
        start = int(center[axis] - length // 2)
        end = start + length
        source_start = max(start, 0)
        source_end = min(end, volume.shape[axis])
        target_start = source_start - start
        target_end = target_start + source_end - source_start
        source.append(slice(source_start, source_end))
        target.append(slice(target_start, target_end))
    crop[tuple(target)] = volume[tuple(source)]
    return crop


def preprocess_ct_case(
    patient_id: str,
    noncontrast_path: str | Path,
    arterial_path: str | Path,
    mask_path: str | Path,
    output_dir: str | Path,
    config: CTPreprocessConfig | None = None,
) -> dict[str, str]:
    """Register, resample, window, and crop paired CT for one patient."""

    config = config or CTPreprocessConfig()
    sitk = _sitk()
    noncontrast = sitk.ReadImage(str(noncontrast_path), sitk.sitkFloat32)
    arterial = sitk.ReadImage(str(arterial_path), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)

    arterial = resample(arterial, config.target_spacing_mm, sitk.sitkLinear)
    mask = resample(mask, config.target_spacing_mm, sitk.sitkNearestNeighbor, reference=arterial)
    noncontrast, _ = rigid_register(noncontrast, arterial)

    arterial_array = scale_hu(sitk.GetArrayFromImage(arterial), config.hu_window)
    noncontrast_array = scale_hu(sitk.GetArrayFromImage(noncontrast), config.hu_window)
    mask_array = sitk.GetArrayFromImage(mask) > 0

    destination = Path(output_dir) / str(patient_id)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {"patient_id": str(patient_id)}
    for name, volume in (("noncontrast", noncontrast_array), ("arterial", arterial_array)):
        crop = crop_around_mask(volume, mask_array, config.roi_shape_voxels)
        path = destination / f"{name}.npz"
        np.savez_compressed(path, volume=crop.astype(np.float32))
        outputs[name] = str(path)
    return outputs
