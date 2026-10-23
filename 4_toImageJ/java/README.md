# SEG-LUT Java plugin (ImageJ 1.x / Fiji)

Pure Java Fiji plugin. The LUT (`.npy`) is **embedded in the JAR** — no Python required at runtime.

## Requirements to build

- JDK 8+
- [Apache Maven](https://maven.apache.org/)

Fiji already includes ImageJ (`ij.jar`); the plugin is compiled against `ij` with `provided` scope.

## One-click package

**Windows (PowerShell):**

```powershell
cd 4_toImageJ\java
.\package.ps1
```

**Linux / macOS:**

```bash
cd 4_toImageJ/java
chmod +x package.sh
./package.sh
```

**Custom LUT** (after `Transfer_Model_Seg.py` exports a new `.npy`):

```powershell
.\package.ps1 -LutPath "..\..\2_Transfer_to_LUT\Seg2_LUT_HV_2_4.npy"
```

Output: [`../dist/Seg-LUT.jar`](../dist/Seg-LUT.jar)

## Install in Fiji

1. Copy `Seg-LUT.jar` to `Fiji.app/plugins/` (Windows: `Fiji.app\plugins\`).
2. Restart Fiji.
3. Open a grayscale (or RGB) image.
4. Run **Plugins → Segmentation → SEG-LUT**.

## Re-bundle after training a new model

1. Export LUT with stage 2 (`Transfer_Model_Seg.py`); use `SAMPLING_INTERVAL=4` to match the plugin constant in `LutResources.java` (change both if you use another interval).
2. Run `package.ps1` / `package.sh` with `-LutPath` pointing to the new `.npy`.
3. Replace the old JAR in `Fiji.app/plugins/` and restart Fiji.
