package seglut;

import java.io.IOException;
import java.io.InputStream;

/** Bundled LUT inside the plugin JAR. */
public final class LutResources {

    public static final int SAMPLING_INTERVAL = 4;
    public static final String LUT_RESOURCE_PATH = "/lut/Seg2_LUT_HV_2_4.npy";

    private static float[] cached;

    private LutResources() {}

    public static float[] getWeight() throws IOException {
        if (cached != null) {
            return cached;
        }
        InputStream in = LutResources.class.getResourceAsStream(LUT_RESOURCE_PATH);
        if (in == null) {
            throw new IOException("Bundled LUT not found in JAR: " + LUT_RESOURCE_PATH);
        }
        try (InputStream stream = in) {
            cached = NpyLoader.loadFloat32(stream);
        }
        return cached;
    }
}
