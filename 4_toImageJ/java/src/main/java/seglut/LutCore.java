package seglut;

/** Full-image LUT segmentation (matches Python lut_core.apply_lut_segmentation). */
public final class LutCore {

    private LutCore() {}

    public static float[] applySegmentation(float[] gray, int width, int height, float[] weight, int samplingInterval) {
        int q = 1 << samplingInterval;
        int L = (1 << (8 - samplingInterval)) + 1;
        int hp = height + 1;
        int wp = width + 1;
        float[] padded = new float[hp * wp];
        for (int y = 0; y < height; y++) {
            System.arraycopy(gray, y * width, padded, y * wp, width);
            padded[y * wp + width] = gray[y * width + width - 1];
        }
        for (int x = 0; x < wp; x++) {
            padded[(height) * wp + x] = padded[(height - 1) * wp + x];
        }

        float[] prob = new float[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int i = y * wp + x;
                float fa = padded[i];
                float fb = padded[i + 1];
                float fc = padded[i + wp];
                float fd = padded[i + wp + 1];

                int a1 = (int) (fa / q);
                int b1 = (int) (fb / q);
                int c1 = (int) (fc / q);
                int d1 = (int) (fd / q);
                int a2 = a1 + 1;
                int b2 = b1 + 1;
                int c2 = c1 + 1;
                int d2 = d1 + 1;

                float p0000 = LutSimplex.lookup(weight, L, a1, b1, c1, d1);
                float p0001 = LutSimplex.lookup(weight, L, a1, b1, c1, d2);
                float p0010 = LutSimplex.lookup(weight, L, a1, b1, c2, d1);
                float p0011 = LutSimplex.lookup(weight, L, a1, b1, c2, d2);
                float p0100 = LutSimplex.lookup(weight, L, a1, b2, c1, d1);
                float p0101 = LutSimplex.lookup(weight, L, a1, b2, c1, d2);
                float p0110 = LutSimplex.lookup(weight, L, a1, b2, c2, d1);
                float p0111 = LutSimplex.lookup(weight, L, a1, b2, c2, d2);
                float p1000 = LutSimplex.lookup(weight, L, a2, b1, c1, d1);
                float p1001 = LutSimplex.lookup(weight, L, a2, b1, c1, d2);
                float p1010 = LutSimplex.lookup(weight, L, a2, b1, c2, d1);
                float p1011 = LutSimplex.lookup(weight, L, a2, b1, c2, d2);
                float p1100 = LutSimplex.lookup(weight, L, a2, b2, c1, d1);
                float p1101 = LutSimplex.lookup(weight, L, a2, b2, c1, d2);
                float p1110 = LutSimplex.lookup(weight, L, a2, b2, c2, d1);
                float p1111 = LutSimplex.lookup(weight, L, a2, b2, c2, d2);

                float raw = LutSimplex.interpolate(
                        fa, fb, fc, fd, q,
                        p0000, p0001, p0010, p0011,
                        p0100, p0101, p0110, p0111,
                        p1000, p1001, p1010, p1011,
                        p1100, p1101, p1110, p1111);
                prob[y * width + x] = raw / 255f;
            }
        }
        return prob;
    }
}
