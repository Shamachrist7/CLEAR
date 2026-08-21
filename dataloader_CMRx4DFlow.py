import os
import sys
import random
from pathlib import Path

import numpy as np
from einops import rearrange
from torch.utils.data import Dataset

sys.path.append("../")
from Utils.utils_datasl import load_mat, read_params_csv
from Utils.utils_flow import k2i_numpy

sys.path.append("../../")
from CMRx4DFlowMaskGeneration import fun_mask_gen_2d
from Utils.misc_utils import mriAdjointOp


REQUIRED_FILES = ("coilmap.mat", "segmask.mat", "params.csv") #"kdata_full.mat", 

DEFAULT_OPTS = {
    "train_roots": [
        "/LOCAL/CMRx4Dflow2026/TaskR1R2/TrainSet/",
    ],
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


def find_valid_cases(roots, required_files=REQUIRED_FILES, anchor="params.csv"): #"kdata_full.mat"):
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


def sorted_read_then_gather(x, axis, idx, fixed_slices=None):
    idx = np.asarray(idx, dtype=np.int64)
    uniq, inv = np.unique(idx, return_inverse=True)

    ndim = len(x.shape)
    key = [slice(None)] * ndim

    if fixed_slices:
        for ax, sl in fixed_slices.items():
            key[ax] = sl

    key[axis] = uniq
    out_uniq = x[tuple(key)]
    out = np.take(out_uniq, inv, axis=axis)

    return out


class CMRx4DFlowDataSet(Dataset):
    def __init__(self, **kwargs):
        options = DEFAULT_OPTS.copy()
        options.update(kwargs)

        self.options = options
        self.mode = options["mode"]

        if self.mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be one of ['train', 'val', 'test'], got {self.mode}")

        self.usrate_list = [10, 20, 30, 40, 50]

        self.D_size = int(options["D_size"])
        self.T_size = int(options["T_size"])
        self.input = options.get("input", None)

        self.in_base_dir = options.get("in_base_dir", None)
        self.out_base_dir = options.get("out_base_dir", None)

        # Validation and testing must behave the same:
        # full encodings, full time, full FE.
        if self.mode in ("val", "test"):
            self.D_size = -1
            self.T_size = -1

        self.test_usrate = options.get("usrate", None)
        if self.mode == "test":
            if self.test_usrate is None:
                raise ValueError("In test mode you must pass --usrate")
            if isinstance(self.test_usrate, int):
                self.test_usrate = [int(self.test_usrate)]
            else:
                self.test_usrate = [int(u) for u in self.test_usrate]

        self.filename = []

        def add_case(case_dir):
            case_dir = str(case_dir)
            out_dir = _compute_out_dir(case_dir, self.in_base_dir, self.out_base_dir)

            if self.mode == "train":
                kdata = load_mat(str(Path(case_dir) / "kdata_full.mat"), "kdata_full")
                Nx = int(kdata.shape[-1])

                if self.D_size == -1:
                    slice_starts = [0]
                else:
                    slice_starts = list(range(0, Nx - self.D_size + 1))

                for i in slice_starts:
                    self.filename.append([case_dir, int(i), None, out_dir])

                return

            # if self.mode == "val":
            #     for u in self.usrate_list:
            #         mask_ok = (Path(case_dir) / f"usmask_ktGaussian{int(u)}.mat").is_file()
            #         if mask_ok:
            #             self.filename.append([case_dir, 0, int(u), out_dir])
            #     return

            if self.mode in {"val", "test"}:
                for u in self.test_usrate:
                    mask_ok = (Path(case_dir) / f"usmask_ktGaussian{int(u)}.mat").is_file()
                    k_ok = (Path(case_dir) / f"kdata_ktGaussian{int(u)}.mat").is_file()
                    if mask_ok and k_ok:
                        self.filename.append([case_dir, 0, int(u), out_dir])
                return

        if self.input is not None and str(self.input) != "":
            case_dir = Path(self.input)
            if not case_dir.exists():
                raise FileNotFoundError(f"Input path does not exist: {case_dir}")
            add_case(str(case_dir))
        else:
            roots = options.get(f"{self.mode}_roots", [])
            subjects = find_valid_cases(roots)
            for patient_dir in subjects:
                add_case(patient_dir)

        if self.mode != "train":
            self.filename.sort(key=lambda x: (str(x[0]), int(x[2]), int(x[1])))

    def __len__(self):
        return len(self.filename)

    @staticmethod
    def _normalize_coils(c):
        denom = np.sqrt(np.sum(np.abs(c) ** 2, axis=0, keepdims=True) + 1e-12)
        return c / denom, denom # in the end, the network outputs z = denom * x, so we need to return denom as well for proper scaling of the output as z / denom = x (Especially before saving the validation and test reconstructions)

    @staticmethod
    def _compute_norm(f):
        denom = np.linalg.norm(np.abs(f) != 0)
        norm = np.linalg.norm(f) / (denom if denom != 0 else 1.0)
        return np.float32(max(float(norm), 1e-8))

    def _make_mask_train(self, PE, SPE, Nt, usrate):
        total_points = (PE * SPE) // int(usrate)

        masks_spe_pe_t = fun_mask_gen_2d(
            mask_size=(PE, SPE),
            center_radius_x=0.5,
            center_radius_y=0.5,
            total_points=total_points,
            pattern_num=Nt,
            sigma_x=PE / 5,
            sigma_y=SPE / 5,
            min_dist_factor=3,
            rep_decay_factor=0.5,
        )

        mask = rearrange(
            masks_spe_pe_t,
            "spe pe t -> 1 1 t 1 pe spe",
        ).astype(np.float32)

        return mask

    def _prepare_common_arrays(
        self,
        case_dir,
        f_raw,
        c,
        s,
        bins,
        slice_start,
        slice_end,
        seg_idx=None,
    ):
        fixed_slices = None if seg_idx is None else {0: slice(seg_idx, seg_idx + 1)}

        f = sorted_read_then_gather(
            f_raw,
            axis=1,
            idx=bins,
            fixed_slices=fixed_slices,
        )

        # Transform FE/readout direction to image space.
        f = k2i_numpy(f, ax=[-1])[..., slice_start:slice_end]

        c = c[..., slice_start:slice_end].astype(np.complex64)
        s = s[..., slice_start:slice_end]

        # Normalize coils before both GT and adjoint reconstruction.
        c, denom = self._normalize_coils(c)
        c = c.astype(np.complex64)

        # Fully sampled coil-combined target from f.
        # f shape before rearrange: [Nv, Nt, Nc, SPE, PE, FE]
        im = np.sum(k2i_numpy(f, ax=[-2, -3]) * np.conj(c), axis=-4) # z = denom * x as it uses the normalized coils, so no need to divide by denom here, since the network will learn to output reconstructions in the normalized coil space (i.e. z_out = denom * x_out, so that the loss is computed as ||z - z_out||)

        # Standard layout used by the network/operators.
        f = rearrange(f, "nv nt nc spe pe fe -> nv nc nt fe pe spe").astype(np.complex64)
        c = rearrange(c, "nc spe pe fe -> nc fe pe spe").astype(np.complex64)
        s = rearrange(s, "spe pe fe -> fe pe spe")
        im = rearrange(im, "nv nt spe pe fe -> nv nt fe pe spe").astype(np.complex64)

        # denom originally has shape [1, SPE, PE, FE].
        # Put it in the same spatial convention as the reconstruction: [1, FE, PE, SPE].
        denom = rearrange(denom, "one spe pe fe -> one fe pe spe").astype(np.float32)

        return f, c, s, im, denom

    def _finalize_sample(
        self,
        case_dir,
        out_dir,
        subj,
        slice_start,
        seg_idx,
        usrate,
        mask,
        f_clean_model,
        c_model,
        s_model,
        im_gt,
        denom, # final scaling of the output reconstructions (especially important for validation and testing, since we want the final reconstructions to be properly scaled for quantitative evaluation and comparison with the GT)
        bins,
        params,
    ):
        f_corrupt = f_clean_model * mask

        imdata_p1 = mriAdjointOp(
            f_corrupt,
            c_model[np.newaxis, :, np.newaxis, :, :, :],
            mask,
        ).astype(np.complex64)

        norm = self._compute_norm(f_corrupt)

        imdata_p1 = (imdata_p1 / norm).astype(np.complex64)
        gt = (im_gt / norm).astype(np.complex64)
        kdata_p1 = (f_corrupt / norm).astype(np.complex64)

        return {
            "imdata_p1": imdata_p1,
            "gt": gt,
            "kdata_p1": kdata_p1,
            "coil_sens": c_model.astype(np.complex64),
            "mask": mask.astype(np.float32),
            "norm": np.float32(norm),
            "segmentation": s_model.astype(np.uint8),
            "case_dir": case_dir,
            "subj": subj,
            "slice_start": np.int64(slice_start),
            "seg_idx": np.int64(-1 if seg_idx is None else seg_idx),
            "usrate": np.int64(usrate),
            "usrate_true": np.float32(1.0 / max(float(np.mean(mask)), 1e-8)),
            "bins": bins.astype(np.int64),
            "Nt": np.int64(im_gt.shape[1]),
            "SPE": np.int64(im_gt.shape[-1]),
            "PE": np.int64(im_gt.shape[-2]),
            "FE": np.int64(im_gt.shape[-3]),
            "VENC": np.asarray(params["VENC"]),
            "out_dir": out_dir,
            "final_im_divisor": denom.astype(np.float32),
        }

    @staticmethod
    def _stack_quintet(samples):
        out = {}

        keys = samples[0].keys()

        for k in keys:
            vals = [s[k] for s in samples]

            if isinstance(vals[0], np.ndarray):
                out[k] = np.stack(vals, axis=0)
            elif isinstance(vals[0], np.generic):
                out[k] = np.stack(vals, axis=0)
            elif isinstance(vals[0], (int, float)):
                out[k] = np.asarray(vals)
            else:
                # strings / paths / metadata
                out[k] = vals

        return out

    def _getitem_train(self, idx):
        case_dir, slice_start, _, out_dir = self.filename[idx]
        subj = Path(case_dir).name

        f_raw = load_mat(str(Path(case_dir) / "kdata_full.mat"), "kdata_full")
        c = load_mat(str(Path(case_dir) / "coilmap.mat"), "coilmap")
        s = load_mat(str(Path(case_dir) / "segmask.mat"), "segmask")
        params = read_params_csv(str(Path(case_dir) / "params.csv"))

        Nv, Nt_full, Nc, SPE_full, PE_full, FE_full = f_raw.shape

        if self.T_size == -1:
            cardiac_bins = list(range(Nt_full))
        else:
            first_bin = random.randint(-self.T_size + 1, Nt_full - self.T_size)
            cardiac_bins = list(range(first_bin, first_bin + self.T_size))

        bins = np.mod(cardiac_bins, Nt_full).astype(np.int64)

        slice_end = None if self.D_size == -1 else slice_start + self.D_size

        samples = []
        for usrate in self.usrate_list:
            # Training alone draws a single velocity encoding / segment.
            # Here, each element of the quintet draws its own random segment.
            seg_idx = random.randint(0, Nv - 1)

            f_clean_model, c_model, s_model, im_gt, denom = self._prepare_common_arrays(
                case_dir=case_dir,
                f_raw=f_raw,
                c=c,
                s=s,
                bins=bins,
                slice_start=slice_start,
                slice_end=slice_end,
                seg_idx=seg_idx,
            )

            _, _, Nt, FE, PE, SPE = f_clean_model.shape

            mask = self._make_mask_train(PE=PE, SPE=SPE, Nt=Nt, usrate=usrate)

            sample = self._finalize_sample(
                case_dir=case_dir,
                out_dir=out_dir,
                subj=subj,
                slice_start=slice_start,
                seg_idx=seg_idx,
                usrate=usrate,
                mask=mask,
                f_clean_model=f_clean_model,
                c_model=c_model,
                s_model=s_model,
                im_gt=im_gt,
                denom=denom,
                bins=bins,
                params=params,
            )
            samples.append(sample)

        return self._stack_quintet(samples)

    def _getitem_eval(self, idx):
        case_dir, slice_start, usrate, out_dir = self.filename[idx]
        subj = Path(case_dir).name

        f_corrupt_raw = load_mat(str(Path(case_dir) / f"kdata_ktGaussian{int(usrate)}.mat"), "kdata_ktGaussian")[()]
        mask = load_mat(str(Path(case_dir) / f"usmask_ktGaussian{int(usrate)}.mat"), "usmask_ktGaussian")[()]
        c = load_mat(str(Path(case_dir) / "coilmap.mat"), "coilmap")[()]
        s = load_mat(str(Path(case_dir) / "segmask.mat"), "segmask")[()]
        params = read_params_csv(str(Path(case_dir) / "params.csv"))
        
        Nv, Nt, Nc, SPE, PE, FE = f_corrupt_raw.shape
        # For evaluation, everything is raw. SO there are 3 things to be taken into account in the main loop of the evaluation script:
        # 1) The normalization of this raw k-space data;
        # 2) The normalization of the coil sensitivity maps;
        # 3) The final scaling of the output reconstructions (especially important for validation and testing, since we want the final reconstructions to be properly scaled for quantitative evaluation and comparison with the GT). This is done by returning the same normalization factor for both the input k-space data and the GT images, so that the network learns to output reconstructions in the normalized coil space (i.e. z_out = denom * x_out, so that the loss is computed as ||z - z_out||), and then dividing the final output reconstructions by this normalization factor as well (i.e. x_out = z_out / denom) to get properly scaled reconstructions for evaluation and comparison with the GT.

        return {
            "kdata_raw": f_corrupt_raw.astype(np.complex64),
            "coil_sens_raw": c.astype(np.complex64),
            "mask": mask.astype(np.float32),
            "segmentation": s.astype(np.uint8),
            "case_dir": case_dir,
            "subj": subj,
            "usrate": np.int64(usrate),
            "usrate_true": np.float32(1.0 / max(float(np.mean(mask)), 1e-8)),
            "Nt": np.int64(Nt),
            "SPE": np.int64(SPE),
            "PE": np.int64(PE),
            "FE": np.int64(FE),
            "VENC": np.asarray(params["VENC"]),
            "out_dir": out_dir,
        }

    def __getitem__(self, idx):
        if self.mode == "train":
            return self._getitem_train(idx)

        # Deterministic validation/testing.
        np.random.seed(0)
        random.seed(0)

        return self._getitem_eval(idx)
    
    





