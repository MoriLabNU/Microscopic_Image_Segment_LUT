package seglut;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Minimal NumPy .npy reader (v1.0, C-order, float32, 1D or 2D).
 */
public final class NpyLoader {

    private static final Pattern SHAPE = Pattern.compile("'shape'\\s*:\\s*\\(([^)]*)\\)");

    private NpyLoader() {}

    public static float[] loadFloat32(InputStream in) throws IOException {
        DataInputStream din = new DataInputStream(in);
        byte[] magic = new byte[6];
        din.readFully(magic);
        if (magic[0] != (byte) 0x93 || !new String(magic, 1, 5, StandardCharsets.US_ASCII).equals("NUMPY")) {
            throw new IOException("Not a .npy file");
        }
        byte major = din.readByte();
        byte minor = din.readByte();
        if (major != 1 || minor != 0) {
            throw new IOException("Only numpy v1.0 .npy is supported");
        }
        int headerLen = Short.reverseBytes(din.readShort()) & 0xffff;
        byte[] headerBytes = new byte[headerLen];
        din.readFully(headerBytes);
        String header = new String(headerBytes, StandardCharsets.US_ASCII).trim();
        if (header.contains("Fortran")) {
            throw new IOException("Fortran-order .npy not supported");
        }
        int count = elementCount(header);
        boolean float32 = header.contains("<f4") || header.contains("float32");
        boolean uint8 = header.contains("|u1") || header.contains("uint8");
        if (!float32 && !uint8) {
            throw new IOException("Expected float32 or uint8 array in .npy");
        }
        byte[] raw = readRemaining(din);
        int elemSize = float32 ? 4 : 1;
        if (raw.length < count * elemSize) {
            throw new IOException("Truncated .npy payload");
        }
        float[] data = new float[count];
        if (float32) {
            ByteBuffer buf = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < count; i++) {
                data[i] = buf.getFloat();
            }
        } else {
            for (int i = 0; i < count; i++) {
                data[i] = raw[i] & 0xff;
            }
        }
        return data;
    }

    private static int elementCount(String header) throws IOException {
        Matcher m = SHAPE.matcher(header);
        if (!m.find()) {
            throw new IOException("Could not parse shape from .npy header");
        }
        String inner = m.group(1).trim();
        if (inner.isEmpty()) {
            return 1;
        }
        int product = 1;
        for (String part : inner.split(",")) {
            part = part.trim();
            if (part.isEmpty()) {
                continue;
            }
            product *= Integer.parseInt(part);
        }
        return product;
    }

    private static byte[] readRemaining(DataInputStream din) throws IOException {
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = din.read(buf)) != -1) {
            baos.write(buf, 0, n);
        }
        return baos.toByteArray();
    }
}
