from pathlib import Path

import numpy as np
import tyro


DEFAULT_COLUMNS = [
  "x",
  "y",
  "z",
  "qx",
  "qy",
  "qz",
  "qw",
  "Left_Hip_Pitch",
  "Left_Hip_Roll",
  "Left_Hip_Yaw",
  "Left_Knee_Pitch",
  "Left_Ankle_Pitch",
  "Left_Ankle_Roll",
  "Right_Hip_Pitch",
  "Right_Hip_Roll",
  "Right_Hip_Yaw",
  "Right_Knee_Pitch",
  "Right_Ankle_Pitch",
  "Right_Ankle_Roll",
  "Waist",
  "Left_Shoulder_Pitch",
  "Left_Shoulder_Roll",
  "Left_Elbow_Pitch",
  "Left_Elbow_Yaw",
  "Right_Shoulder_Pitch",
  "Right_Shoulder_Roll",
  "Right_Elbow_Pitch",
  "Right_Elbow_Yaw",
  "AAHead_yaw",
  "Head_pitch",
]

# For left/right mirrored motion on the sagittal plane:
# - swap left<->right joints
# - negate joints rotating around x (roll) or z (yaw)
LEFT_RIGHT_SWAP = [
  ("Left_Hip_Pitch", "Right_Hip_Pitch"),
  ("Left_Hip_Roll", "Right_Hip_Roll"),
  ("Left_Hip_Yaw", "Right_Hip_Yaw"),
  ("Left_Knee_Pitch", "Right_Knee_Pitch"),
  ("Left_Ankle_Pitch", "Right_Ankle_Pitch"),
  ("Left_Ankle_Roll", "Right_Ankle_Roll"),
  ("Left_Shoulder_Pitch", "Right_Shoulder_Pitch"),
  ("Left_Shoulder_Roll", "Right_Shoulder_Roll"),
  ("Left_Elbow_Pitch", "Right_Elbow_Pitch"),
  ("Left_Elbow_Yaw", "Right_Elbow_Yaw"),
]

NEGATE_AFTER_SWAP = {
  "Left_Hip_Roll",
  "Right_Hip_Roll",
  "Left_Hip_Yaw",
  "Right_Hip_Yaw",
  "Left_Ankle_Roll",
  "Right_Ankle_Roll",
  "Left_Shoulder_Roll",
  "Right_Shoulder_Roll",
  "Left_Elbow_Yaw",
  "Right_Elbow_Yaw",
}

# Root/global quantities that must change sign under y-axis reflection.
NEGATE_NO_SWAP = {"y", "qx", "qz", "Waist", "AAHead_yaw"}


def _parse_header(path: Path) -> list[str] | None:
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      stripped = line.strip()
      if not stripped:
        continue
      if stripped.startswith("#"):
        header = stripped.lstrip("#").strip()
        if "," in header and any(ch.isalpha() for ch in header):
          return [part.strip() for part in header.split(",")]
        continue
      break
  return None


def _load_csv(path: Path) -> tuple[np.ndarray, list[str] | None]:
  header = _parse_header(path)
  data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)
  if data.ndim == 1:
    data = data[np.newaxis, :]
  return data, header


def _resolve_columns(ncols: int, header: list[str] | None) -> list[str]:
  if header is not None and len(header) == ncols:
    return header

  if ncols < len(DEFAULT_COLUMNS):
    raise ValueError(
      f"CSV has {ncols} columns, but Booster T1 format needs at least {len(DEFAULT_COLUMNS)}"
    )

  extra_cols = [f"extra_{i}" for i in range(ncols - len(DEFAULT_COLUMNS))]
  return DEFAULT_COLUMNS + extra_cols


def _mirror_matrix(data: np.ndarray, columns: list[str]) -> np.ndarray:
  col_to_idx = {name: idx for idx, name in enumerate(columns)}
  required = set(NEGATE_NO_SWAP)
  for left, right in LEFT_RIGHT_SWAP:
    required.add(left)
    required.add(right)

  missing = sorted(name for name in required if name not in col_to_idx)
  if missing:
    raise ValueError(
      "Missing required columns for mirroring: "
      + ", ".join(missing)
      + "."
      + " CSV must follow Booster T1 23-DoF naming/order."
    )

  mirrored = data.copy()

  for left, right in LEFT_RIGHT_SWAP:
    li = col_to_idx[left]
    ri = col_to_idx[right]
    sign = -1.0 if left in NEGATE_AFTER_SWAP else 1.0
    mirrored[:, li] = sign * data[:, ri]
    mirrored[:, ri] = sign * data[:, li]

  for name in NEGATE_NO_SWAP:
    mirrored[:, col_to_idx[name]] *= -1.0

  return mirrored


def _save_csv(path: Path, data: np.ndarray, columns: list[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(
    path,
    data,
    delimiter=",",
    fmt="%.6f",
    header=",".join(columns),
    comments="# ",
  )


def _default_output_path(input_path: Path) -> Path:
  return input_path.with_name(f"{input_path.stem}_mirror{input_path.suffix}")


def _iter_csv_files(root_dir: Path, include_already_mirrored: bool) -> list[Path]:
  files = sorted(root_dir.rglob("*.csv"))
  if include_already_mirrored:
    return files
  return [path for path in files if not path.stem.endswith("_mirror")]


def _mirror_one(input_path: Path, output_path: Path, overwrite: bool) -> str:
  if output_path.exists() and not overwrite:
    return "skipped_exists"

  data, header = _load_csv(input_path)
  columns = _resolve_columns(data.shape[1], header)
  mirrored = _mirror_matrix(data, columns)
  _save_csv(output_path, mirrored, columns)
  return "written"


def main(
  input_file: str | None = None,
  root_dir: str | None = None,
  output_file: str | None = None,
  output_root: str | None = None,
  overwrite: bool = False,
  include_already_mirrored: bool = False,
):
  """Create mirrored Booster T1 CSV motions (left/right swapped).

  Args:
    input_file: Path to a single CSV file.
    root_dir: Path to a folder; all *.csv are processed recursively.
    output_file: Optional output path in single-file mode.
    output_root: Optional output root in batch mode (preserves subfolders).
    overwrite: If True, overwrite existing output files.
    include_already_mirrored: In batch mode, also process *_mirror.csv inputs.
  """
  if (input_file is None) == (root_dir is None):
    raise ValueError("Provide exactly one of `input_file` or `root_dir`.")

  if input_file is not None and output_root is not None:
    raise ValueError("`output_root` is only valid with `root_dir`.")
  if root_dir is not None and output_file is not None:
    raise ValueError("`output_file` is only valid with `input_file`.")

  if input_file is not None:
    src = Path(input_file)
    if not src.is_file():
      raise FileNotFoundError(f"Input CSV not found: {src}")
    dst = Path(output_file) if output_file is not None else _default_output_path(src)
    result = _mirror_one(src, dst, overwrite=overwrite)
    print(f"[{result.upper()}] {src} -> {dst}")
    return

  root = Path(root_dir)  # type: ignore[arg-type]
  if not root.is_dir():
    raise FileNotFoundError(f"Input folder not found: {root}")

  batch_files = _iter_csv_files(root, include_already_mirrored=include_already_mirrored)
  if not batch_files:
    raise FileNotFoundError(f"No CSV files found in: {root}")

  out_root = Path(output_root) if output_root is not None else None
  written = 0
  skipped = 0

  for src in batch_files:
    if out_root is None:
      dst = _default_output_path(src)
    else:
      rel = src.relative_to(root)
      dst = out_root / rel.parent / f"{rel.stem}_mirror{rel.suffix}"
    result = _mirror_one(src, dst, overwrite=overwrite)
    print(f"[{result.upper()}] {src} -> {dst}")
    if result == "written":
      written += 1
    else:
      skipped += 1

  print(f"[DONE] written={written}, skipped_existing={skipped}, total={len(batch_files)}")


if __name__ == "__main__":
  tyro.cli(main)
