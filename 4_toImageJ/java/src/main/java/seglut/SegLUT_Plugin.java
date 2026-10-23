package seglut;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;

/**
 * ImageJ 1.x / Fiji plugin: SEG-LUT segmentation with bundled LUT in the JAR.
 */
public class SegLUT_Plugin implements PlugIn {

    private static final double DEFAULT_THRESHOLD = 0.5;

    @Override
    public void run(String arg) {
        ImagePlus imp = IJ.getImage();
        if (imp == null) {
            IJ.error("SEG-LUT", "Open a grayscale image first.");
            return;
        }

        GenericDialog gd = new GenericDialog("SEG-LUT");
        gd.addNumericField("Probability_threshold", DEFAULT_THRESHOLD, 2);
        gd.showDialog();
        if (gd.wasCanceled()) {
            return;
        }
        double threshold = gd.getNextNumber();

        IJ.showStatus("SEG-LUT: loading LUT...");
        float[] weight;
        try {
            weight = LutResources.getWeight();
        } catch (Exception e) {
            IJ.error("SEG-LUT", "Failed to load bundled LUT:\n" + e.getMessage());
            return;
        }

        ImageProcessor ip = imp.getProcessor();
        int w = ip.getWidth();
        int h = ip.getHeight();
        float[] gray = toFloatGray(ip, w, h);

        IJ.showStatus("SEG-LUT: segmenting...");
        long t0 = System.currentTimeMillis();
        float[] prob = LutCore.applySegmentation(gray, w, h, weight, LutResources.SAMPLING_INTERVAL);
        IJ.showStatus("SEG-LUT: done in " + (System.currentTimeMillis() - t0) + " ms");

        ByteProcessor mask = new ByteProcessor(w, h);
        for (int i = 0; i < prob.length; i++) {
            mask.set(i, prob[i] >= threshold ? (byte) 255 : (byte) 0);
        }

        ImagePlus out = new ImagePlus(imp.getTitle() + " (SEG-LUT)", mask);
        out.copyScale(imp);
        out.show();
        IJ.showStatus("");
    }

    private static float[] toFloatGray(ImageProcessor ip, int w, int h) {
        float[] gray = new float[w * h];
        if (ip instanceof ColorProcessor) {
            int[] pixels = (int[]) ip.getPixels();
            for (int i = 0; i < pixels.length; i++) {
                int c = pixels[i];
                int r = (c >> 16) & 0xff;
                int g = (c >> 8) & 0xff;
                int b = c & 0xff;
                gray[i] = (float) (0.299 * r + 0.587 * g + 0.114 * b);
            }
            return gray;
        }
        if (ip instanceof ByteProcessor) {
            byte[] pixels = (byte[]) ip.getPixels();
            for (int i = 0; i < pixels.length; i++) {
                gray[i] = pixels[i] & 0xff;
            }
        } else if (ip instanceof ShortProcessor) {
            short[] pixels = (short[]) ip.getPixels();
            for (int i = 0; i < pixels.length; i++) {
                gray[i] = pixels[i] & 0xffff;
            }
        } else if (ip instanceof FloatProcessor) {
            float[] pixels = (float[]) ip.getPixels();
            System.arraycopy(pixels, 0, gray, 0, pixels.length);
        } else {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    gray[y * w + x] = ip.getPixelValue(x, y);
                }
            }
        }
        return gray;
    }
}
