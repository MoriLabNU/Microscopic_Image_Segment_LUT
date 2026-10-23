#@ String(label="Python executable", value="python") PYTHON_EXE
#@ String(label="Segment script (segment_lut.py)", value="") SEG_LUT_SCRIPT
#@ String(label="LUT .npy path", value="") SEG_LUT_NPY

from ij import IJ, ImagePlus
from ij.plugin.filter import PlugInFilter
import subprocess
import os

try:
    from java.lang import System
except ImportError:
    System = None


def _env(key):
    if System is None:
        return None
    v = System.getenv(key)
    return v if v else None


def _read_config(imagej_dir):
    cfg = {}
    path = os.path.join(imagej_dir, "plugins", "Seg-LUT", "config.txt")
    if not os.path.isfile(path):
        return cfg
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def _resolve_path(imagej_dir, p):
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(imagej_dir, p))


class SegLUTPlugin(PlugInFilter):
    def setup(self, arg, imp):
        return PlugInFilter.DOES_ALL + PlugInFilter.NO_CHANGES

    def run(self, ip):
        imp = IJ.getImage()
        if imp is None:
            IJ.error("SEG-LUT", "Open a grayscale image first.")
            return

        imagej_dir = IJ.getDirectory("imagej")
        cfg = _read_config(imagej_dir)

        python_exe = _env("PYTHON_EXE") or cfg.get("PYTHON_EXE") or PYTHON_EXE
        script = _env("SEG_LUT_SCRIPT") or cfg.get("SEG_LUT_SCRIPT") or SEG_LUT_SCRIPT
        lut_npy = _env("SEG_LUT_NPY") or cfg.get("SEG_LUT_NPY") or SEG_LUT_NPY

        if not script:
            script = os.path.join(
                imagej_dir, "plugins", "Seg-LUT", "py_lut", "segment_lut.py"
            )
        script = _resolve_path(imagej_dir, script)

        if not lut_npy:
            lut_npy = os.path.join(
                imagej_dir,
                "plugins",
                "Seg-LUT",
                "lut",
                "Seg2_LUT_HV_2_4.npy",
            )
        lut_npy = _resolve_path(imagej_dir, lut_npy)

        if not os.path.isfile(script):
            IJ.error("SEG-LUT", "segment_lut.py not found:\n" + script)
            return
        if not os.path.isfile(lut_npy):
            IJ.error("SEG-LUT", "LUT not found:\n" + lut_npy)
            return

        temp_dir = os.path.join(imagej_dir, "temp_seg_lut")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        temp_in = os.path.join(temp_dir, "input.png")
        temp_out = os.path.join(temp_dir, "output.png")

        temp_imp = ImagePlus("Temp", ip.duplicate())
        IJ.saveAs(temp_imp, "PNG", temp_in)

        cmd = [
            python_exe,
            script,
            "--input",
            temp_in,
            "--output",
            temp_out,
            "--lut",
            lut_npy,
        ]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = proc.communicate()
        except Exception as e:
            IJ.error("SEG-LUT", "Failed to run Python:\n" + str(e))
            return

        if proc.returncode != 0:
            msg = stderr.decode("utf-8", errors="replace") if stderr else ""
            if stdout:
                msg = msg + "\n" + stdout.decode("utf-8", errors="replace")
            IJ.error("SEG-LUT", "Python failed (code {}):\n{}".format(proc.returncode, msg))
            return

        if not os.path.isfile(temp_out):
            IJ.error("SEG-LUT", "No output produced at:\n" + temp_out)
            return

        out_imp = IJ.openImage(temp_out)
        if out_imp is None:
            IJ.error("SEG-LUT", "Could not open output image.")
            return
        out_imp.setTitle(imp.getTitle() + " (SEG-LUT)")
        out_imp.show()


_imp = IJ.getImage()
if _imp is not None:
    SegLUTPlugin().run(_imp.getProcessor())
else:
    IJ.error("SEG-LUT", "Open a grayscale image first.")
