# Peaks Tensor Column Mapping

This document describes the column indices used in the `peaks` tensor throughout the diffraction simulation pipeline.

## Original Peaks from Polycrystal (25 columns)

The `Polycrystal.compute_peaks()` method returns a tensor with the following columns:

| Index | Name | Description |
|-------|------|-------------|
| 0 | grain_index | Index of the scattering grain/element |
| 1 | phase_number | Phase identifier |
| 2 | h | Miller index h |
| 3 | k | Miller index k |
| 4 | l | Miller index l |
| 5 | structure_factors | Structure factor magnitude |
| 6 | diffraction_times | Time of diffraction event |
| 7 | G0_x | Incident scattering vector x-component |
| 8 | G0_y | Incident scattering vector y-component |
| 9 | G0_z | Incident scattering vector z-component |
| 10 | Gx | Scattered scattering vector x-component |
| 11 | Gy | Scattered scattering vector y-component |
| 12 | Gz | Scattered scattering vector z-component |
| 13 | K_out_x | Outgoing wave vector x-component |
| 14 | K_out_y | Outgoing wave vector y-component |
| 15 | K_out_z | Outgoing wave vector z-component |
| 16 | Source_x | X-ray source position x-component |
| 17 | Source_y | X-ray source position y-component |
| 18 | Source_z | X-ray source position z-component |
| 19 | lorentz_factors | Lorentz polarization factor |
| 20 | polarization_factors | Polarization factor |
| 21 | volumes | Scattering volume |
| 22 | 2theta | Two-theta diffraction angle |
| 23 | scherrer_fwhm | Scherrer line broadening FWHM |
| 24 | peak_index | Unique peak identifier |

## Added by Detector._peaks_detector_intersection() (4 columns added)

The detector intersection function adds columns through two operations:

### Step 1: get_intersection() - adds 3 columns
```python
zd_yd_angle = self.get_intersection(peaks[:, 13:16], peaks[:, 16:19])
peaks = torch.cat((peaks, zd_yd_angle), dim=1)
```

| Index | Name | Description |
|-------|------|-------------|
| 25 | zd | Detector z-coordinate (pixel units) |
| 26 | yd | Detector y-coordinate (pixel units) |
| 27 | incident_angle | Incident angle in degrees |

### Step 2: Frame assignment - adds 1 column
```python
frame = torch.zeros((peaks.shape[0], 1)) if frames_to_render <= 1 else ...
peaks = torch.cat((peaks, frame), dim=1)
```

| Index | Name | Description |
|-------|------|-------------|
| 28 | frame | Frame index for time-resolved rendering |

### Step 3: Intensity calculation - adds 1 column
```python
intensity = peaks[:, 21]  # volumes (base intensity)
if self.structure_factor: intensity *= peaks[:, 5]
if self.polarization_factor: intensity *= peaks[:, 20]
if self.lorentz_factor: intensity *= peaks[:, 19]
peaks = torch.cat((peaks, intensity), dim=1)
```

| Index | Name | Description |
|-------|------|-------------|
| 29 | intensity | Combined intensity (volume × structure × polarization × lorentz) |

## Usage in Rendering Functions

### _render_gauss_peaks()
```python
pos_z = peaks[:, 25] / self.pixel_size_z  # zd coordinate
pos_y = peaks[:, 26] / self.pixel_size_y  # yd coordinate
frame_idx = peaks[:, 28].long()            # Frame index
intensities = peaks[:, 29]                 # Combined intensity
```

### _render_voigt_peaks()
```python
frames_n = int(peaks[:, 28].max() + 1)     # Frame index for counting
frame_mask = peaks[:, 28] == frame_idx     # Frame index for masking
fwhm_rad = batch_peaks[:, 23]              # Scherrer FWHM
incident_angles = batch_peaks[:, 27]       # Incident angle
zd = batch_peaks[:, 25] / self.pixel_size_z    # zd coordinate
yd = batch_peaks[:, 26] / self.pixel_size_y    # yd coordinate
intensities = batch_peaks[:, 29]           # Combined intensity
```

## Important Notes

1. **Column 6 (diffraction_times)** is used for frame assignment via time-based bucketing
2. **Columns 21, 5, 20, 19** are combined to create column 29 (intensity)
3. **Columns 13-16** (K_out) are extracted for scattered wave vectors in volume projection
4. After filtering by detector bounds, the peaks tensor maintains its column structure
5. All column indices are 0-indexed

## Verification Checklist

- [x] Gauss rendering uses correct columns for zd (25), yd (26), frame (28), intensity (29)
- [x] Voigt rendering uses correct columns for frame (28), incident_angle (27), zd (25), yd (26), intensity (29)
- [x] Frame counting uses column 28 (frame), not column 27 (incident_angle)
- [x] Intensity calculation combines columns 21, 5, 20, 19 into column 29
- [x] All column references have been validated
