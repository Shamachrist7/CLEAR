import os
import sys
from pathlib import Path

import numpy as np
from einops import rearrange
from torch.utils.data import Dataset

sys.path.append("../")
from Utils.utils_datasl import load_mat, read_params_csv
from Utils.utils_flow import k2i_numpy

from Utils.misc_utils import mriAdjointOp


REQUIRED_FILES = ("coilmap.mat", "segmask.mat", "params.csv")

DEFAULT_OPTS = {
    "val_roots": [
        "/LOCAL/CMRx4Dflow2026/TaskR1R2/ValidationSet/",
    ],
    "test_roots": [
        "/LOCAL/CMRx4Dflow2026/TaskR1R2/TestSet/",
    ],
}


def _compute_out_dir(case_dir, in_base_dir, out_base_dir):
    case_dir = str(case_dir)

    if out_base_dir is None or str(out_base_dir) == "":
        return case_dir

    out_base_dir = str(out_base_dir)

    if in_base_dir is None or str(in_base_dir) == "":
        return str(Path(out_base_dir) / Path(case_dir).name)

    in_base_dir = str(in_base_dir)
    rel = os.path.relpath(case_dir, in_base_dir)
    return str(Path(out_base_dir) / rel)


def find_valid_cases(roots, required_files=REQUIRED_FILES, anchor="params.csv"):
    req = tuple(required_files)
    out, seen = [], set()

    for r in roots:
        root = Path(r)
        if not root.exists():
            continue

        for kpath in root.rglob(anchor):
            case_dir = kpath.parent
            if all((case_dir / f).is_file() for f in req):
                p = str(case_dir)
                if p not in seen:
                    seen.add(p)
                    out.append(p)

    out.sort()
    return out


def load_usmask_ktGaussian(case_dir, usrate, Nt, SPE, PE):
    mask_path = Path(case_dir) / f"usmask_ktGaussian{usrate}.mat"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    m = load_mat(str(mask_path), "usmask_ktGaussian")[()]
    expected = (1, Nt, 1, SPE, PE, 1)

    if m.shape != expected:
        raise ValueError(f"Unexpected mask shape {m.shape}, expected {expected}")

    m = np.squeeze(m, axis=(0, 2, 5))      # [Nt, SPE, PE]
    m = np.transpose(m, (1, 2, 0))         # [SPE, PE, Nt]

    mask = rearrange(m, "spe pe t -> 1 1 t 1 pe spe").astype(np.float32)
    return mask


class CMRx4DFlowEvalDataSet(Dataset):
    """
    Evaluation-only dataset for validation/test.

    It loads undersampled k-space and preprocesses it into a format directly usable by
    the network.

    Raw loaded k-space:
        kdata_ktGaussianR.mat:
            [Nv, Nt, Nc, SPE, PE, FE]

    Returned network-ready shapes:
        imdata_p1:
            [Nv, 1, Nt, FE, PE, SPE]

        kdata_p1:
            [Nv, 1, Nc, Nt, FE, PE, SPE]

        coil_sens:
            [Nv, Nc, FE, PE, SPE]

        mask:
            [Nv, 1, 1, Nt, 1, PE, SPE]

    Here Nv=4 is treated as the batch dimension.
    """

    def __init__(self, **kwargs):
        options = DEFAULT_OPTS.copy()
        options.update(kwargs)

        self.options = options
        self.mode = options.get("mode", "val")

        if self.mode not in ("val", "test"):
            raise ValueError(f"CMRx4DFlowEvalDataSet only supports mode='val' or mode='test', got {self.mode}")

        self.usrate_list = [10, 20, 30, 40, 50]
        self.input = options.get("input", None)

        self.in_base_dir = options.get("in_base_dir", None)
        self.out_base_dir = options.get("out_base_dir", None)

        self.eval_usrate = options.get("usrate", None)

        if self.mode == "val":
            if self.eval_usrate is None:
                self.eval_usrate = self.usrate_list
            elif isinstance(self.eval_usrate, int):
                self.eval_usrate = [int(self.eval_usrate)]
            else:
                self.eval_usrate = [int(u) for u in self.eval_usrate]

        if self.mode == "test":
            if self.eval_usrate is None:
                raise ValueError("In test mode you must pass usrate, e.g. usrate=10 or usrate=[10, 20, 30]")
            if isinstance(self.eval_usrate, int):
                self.eval_usrate = [int(self.eval_usrate)]
            else:
                self.eval_usrate = [int(u) for u in self.eval_usrate]

        self.filename = []

        def add_case(case_dir):
            case_dir = str(case_dir)
            out_dir = _compute_out_dir(case_dir, self.in_base_dir, self.out_base_dir)

            for u in self.eval_usrate:
                mask_ok = (Path(case_dir) / f"usmask_ktGaussian{int(u)}.mat").is_file()
                k_ok = (Path(case_dir) / f"kdata_ktGaussian{int(u)}.mat").is_file()

                if mask_ok and k_ok:
                    self.filename.append([case_dir, int(u), out_dir])

        if self.input is not None and str(self.input) != "":
            case_dir = Path(self.input)
            if not case_dir.exists():
                raise FileNotFoundError(f"Input path does not exist: {case_dir}")
            add_case(str(case_dir))
        else:
            roots = options.get(f"{self.mode}_roots", [])
            subjects = find_valid_cases(
                roots,
                required_files=REQUIRED_FILES,
                anchor="params.csv",
            )

            for patient_dir in subjects:
                add_case(patient_dir)

        self.filename.sort(key=lambda x: (str(x[0]), int(x[1])))

    def __len__(self):
        return len(self.filename)

    @staticmethod
    def _normalize_coils(c):
        denom = np.sqrt(np.sum(np.abs(c) ** 2, axis=0, keepdims=True) + 1e-12)
        return c / denom, denom # in the end, the network outputs z = denom * x, so we need to return denom as well for proper scaling of the output as z / denom = x (Especially before saving the validation and test reconstructions)

    @staticmethod
    def _compute_norm_per_encoding(f):
        """
        Compute one scalar normalization per encoding / batch element.

        Input
        -----
        f:
            Complex undersampled k-space in network layout:
                [Nv, Nc, Nt, FE, PE, SPE]

        Output
        ------
        norm:
            [Nv] float32, one normalization value per encoding.
        """
        Nv = f.shape[0]
        f_flat = f.reshape(Nv, -1)

        numerator = np.linalg.norm(f_flat, axis=1)
        denominator = np.linalg.norm(np.abs(f_flat) != 0, axis=1)

        norm = numerator / np.maximum(denominator, 1.0)
        norm = np.maximum(norm, 1e-8)

        return norm.astype(np.float32)

    def _prepare_eval_sample_for_network(
        self,
        case_dir,
        out_dir,
        subj,
        usrate,
        f_corrupt_raw,
        c_raw,
        s_raw,
        params,
    ):
        Nv, Nt, Nc, SPE, PE, FE = f_corrupt_raw.shape

        mask_model = load_usmask_ktGaussian(
            case_dir=case_dir,
            usrate=int(usrate),
            Nt=Nt,
            SPE=SPE,
            PE=PE,
        )
        # mask_model: [1, 1, Nt, 1, PE, SPE]

        mask_raw = load_mat(
            str(Path(case_dir) / f"usmask_ktGaussian{int(usrate)}.mat"),
            "usmask_ktGaussian",
        )[()]
        # mask_raw: [1, Nt, 1, SPE, PE, 1]

        # Transform FE/readout direction to image space.
        # f_corrupt_raw shape: [Nv, Nt, Nc, SPE, PE, FE]
        f = k2i_numpy(f_corrupt_raw, ax=[-1]).astype(np.complex64)

        c = c_raw.astype(np.complex64)
        s = s_raw.astype(np.uint8)

        # Normalize coils before adjoint reconstruction.
        c, denom = self._normalize_coils(c)
        c = c.astype(np.complex64)

        # Standard layout used by the network/operators.
        f = rearrange(f, "nv nt nc spe pe fe -> nv nc nt fe pe spe").astype(np.complex64)
        c = rearrange(c, "nc spe pe fe -> nc fe pe spe").astype(np.complex64)
        s_model = rearrange(s, "spe pe fe -> fe pe spe").astype(np.uint8)

        # denom originally has shape [1, SPE, PE, FE].
        # Put it in the same spatial convention as the reconstruction: [1, FE, PE, SPE].
        denom_model = rearrange(denom, "one spe pe fe -> one fe pe spe").astype(np.float32)

        # The loaded validation/test k-space is already undersampled.
        # Multiplying by mask again is harmless if the file and mask agree,
        # and protects us against tiny nonzero values outside the official mask.
        f_corrupt = f * mask_model
        # f_corrupt: [Nv, Nc, Nt, FE, PE, SPE]

        # Compute one normalization scalar per encoding / batch element.
        norm = self._compute_norm_per_encoding(f_corrupt)
        # norm: [Nv]

        # Zero-filled reconstruction / adjoint reconstruction.
        imdata_p1 = mriAdjointOp(
            f_corrupt,
            c[np.newaxis, :, np.newaxis, :, :, :],
            mask_model,
        ).astype(np.complex64)
        # imdata_p1: [Nv, Nt, FE, PE, SPE]

        # Normalize each encoding separately.
        imdata_p1 = (
            imdata_p1 / norm[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        ).astype(np.complex64)

        kdata_p1 = (
            f_corrupt / norm[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        ).astype(np.complex64)

        # Treat the 4 encodings as the batch dimension for evaluation.
        #
        # Before:
        #   imdata_p1:  [Nv, Nt, FE, PE, SPE]
        #   kdata_p1:   [Nv, Nc, Nt, FE, PE, SPE]
        #   c:          [Nc, FE, PE, SPE]
        #   mask_model: [1, 1, Nt, 1, PE, SPE]
        #   denom_model:[1, FE, PE, SPE]
        #
        # After:
        #   imdata_p1:       [Nv, 1, Nt, FE, PE, SPE]
        #   kdata_p1:        [Nv, 1, Nc, Nt, FE, PE, SPE]
        #   coil_sens:       [Nv, Nc, FE, PE, SPE]
        #   mask:            [Nv, 1, 1, Nt, 1, PE, SPE]
        #   final_im_divisor:[Nv, 1, 1, FE, PE, SPE]
        imdata_p1 = imdata_p1[:, np.newaxis, ...]
        kdata_p1 = kdata_p1[:, np.newaxis, ...]

        coil_sens_batched = np.repeat(c[np.newaxis, ...], Nv, axis=0)

        mask_batched = np.repeat(mask_model[np.newaxis, ...], Nv, axis=0)

        final_im_divisor = np.repeat(
            denom_model[np.newaxis, :, np.newaxis, :, :, :],
            Nv,
            axis=0,
        ).astype(np.float32)
        # final_im_divisor: [Nv, 1, 1, FE, PE, SPE]

        usrate_true_scalar = np.float32(1.0 / max(float(np.mean(mask_model)), 1e-8))
        usrate_true_batch = np.full((Nv,), usrate_true_scalar, dtype=np.float32)
        usrate_batch = np.full((Nv,), int(usrate), dtype=np.int64)

        return {
            # Network-ready data.
            "imdata_p1": imdata_p1.astype(np.complex64),
            "kdata_p1": kdata_p1.astype(np.complex64),
            "coil_sens": coil_sens_batched.astype(np.complex64),
            "mask": mask_batched.astype(np.float32),

            # Per-encoding normalization.
            "norm": norm.astype(np.float32),
            "final_im_divisor": final_im_divisor.astype(np.float32),

            # Raw data, kept for traceability/debugging.
            "kdata_raw": f_corrupt_raw.astype(np.complex64),
            "coil_sens_raw": c_raw.astype(np.complex64),
            "mask_raw": mask_raw.astype(np.float32),
            "mask_model_single": mask_model.astype(np.float32),

            # Metadata and useful quantities.
            "segmentation": s_model.astype(np.uint8),
            "segmentation_raw": s.astype(np.uint8),
            "case_dir": case_dir,
            "subj": subj,
            "slice_start": np.int64(0),
            "seg_idx": np.int64(-1),
            "usrate": np.int64(usrate),
            "usrate_batch": usrate_batch,
            "usrate_true": usrate_true_batch,
            "usrate_true_scalar": usrate_true_scalar,
            "bins": np.arange(Nt, dtype=np.int64),
            "Nt": np.int64(Nt),
            "SPE": np.int64(SPE),
            "PE": np.int64(PE),
            "FE": np.int64(FE),
            "VENC": np.asarray(params["VENC"]),
            "out_dir": out_dir,
        }

    def __getitem__(self, idx):
        case_dir, usrate, out_dir = self.filename[idx]
        subj = Path(case_dir).name

        f_corrupt_raw = load_mat(
            str(Path(case_dir) / f"kdata_ktGaussian{int(usrate)}.mat"),
            "kdata_ktGaussian",
        )[()]

        c_raw = load_mat(str(Path(case_dir) / "coilmap.mat"), "coilmap")[()]
        s_raw = load_mat(str(Path(case_dir) / "segmask.mat"), "segmask")[()]
        params = read_params_csv(str(Path(case_dir) / "params.csv"))

        # For evaluation, everything is now preprocessed and directly usable by the network.
        # There are 3 things to be taken into account:
        # 1) The normalization of this raw k-space data;
        # 2) The normalization of the coil sensitivity maps;
        # 3) The final scaling of the output reconstructions (especially important for validation and testing, since we want the final reconstructions to be properly scaled for quantitative evaluation and comparison with the GT).
        #
        # The network output is in normalized-coil space:
        #     z_out = denom * x_out
        #
        # and also in intensity-normalized scale.
        # To return to the original image scale:
        #     x_out = (network_output * norm) / denom

        return self._prepare_eval_sample_for_network(
            case_dir=case_dir,
            out_dir=out_dir,
            subj=subj,
            usrate=usrate,
            f_corrupt_raw=f_corrupt_raw,
            c_raw=c_raw,
            s_raw=s_raw,
            params=params,
        )