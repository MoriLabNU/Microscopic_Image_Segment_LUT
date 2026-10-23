package seglut;

/** 4D simplex interpolation (Table 3 in the SEG-LUT paper). */
final class LutSimplex {

    private LutSimplex() {}

    static float interpolate(
            float fa,
            float fb,
            float fc,
            float fd,
            int q,
            float p0000,
            float p0001,
            float p0010,
            float p0011,
            float p0100,
            float p0101,
            float p0110,
            float p0111,
            float p1000,
            float p1001,
            float p1010,
            float p1011,
            float p1100,
            float p1101,
            float p1110,
            float p1111) {
        float qf = q;
        float fa_ = lsb(fa, q);
        float fb_ = lsb(fb, q);
        float fc_ = lsb(fc, q);
        float fd_ = lsb(fd, q);

        boolean cond_a_gt_b = fa_ > fb_;
        boolean cond_b_gt_c = fb_ > fc_;
        boolean cond_c_gt_d = fc_ > fd_;
        boolean cond_a_gt_c = fa_ > fc_;
        boolean cond_b_gt_d = fb_ > fd_;
        boolean cond_a_gt_d = fa_ > fd_;
        boolean cond_c_gt_d_alt = fc_ > fd_;
        boolean cond_b_gt_c_alt = fb_ > fc_;
        boolean cond_a_gt_c_alt = fa_ > fc_;
        boolean cond_b_gt_d_alt = fb_ > fd_;
        boolean cond_c_gt_a = fc_ > fa_;
        boolean cond_b_gt_a = fb_ > fa_;
        boolean cond_d_gt_a = fd_ > fa_;
        boolean cond_d_gt_b = fd_ > fb_;
        boolean cond_d_gt_c = fd_ > fc_;
        boolean cond_b_gt_c2 = fb_ > fc_;
        boolean cond_a_gt_b2 = fa_ > fb_;

        float out = 0f;

        if (cond_a_gt_b && cond_b_gt_c && cond_c_gt_d) {
            out += (qf - fa_) * p0000 + (fa_ - fb_) * p1000 + (fb_ - fc_) * p1100 + (fc_ - fd_) * p1110 + fd_ * p1111;
        }
        if (cond_a_gt_b && cond_b_gt_c && !cond_c_gt_d && cond_b_gt_d) {
            out += (qf - fa_) * p0000 + (fa_ - fb_) * p1000 + (fb_ - fd_) * p1100 + (fd_ - fc_) * p1101 + fc_ * p1111;
        }
        if (cond_a_gt_b && cond_b_gt_c && !cond_c_gt_d && !cond_b_gt_d && cond_a_gt_d) {
            out += (qf - fa_) * p0000 + (fa_ - fd_) * p1000 + (fd_ - fb_) * p1001 + (fb_ - fc_) * p1101 + fc_ * p1111;
        }
        if (cond_a_gt_b && cond_b_gt_c && !cond_c_gt_d && !cond_b_gt_d && !cond_a_gt_d) {
            out += (qf - fd_) * p0000 + (fd_ - fa_) * p0001 + (fa_ - fb_) * p1001 + (fb_ - fc_) * p1101 + fc_ * p1111;
        }
        if (cond_a_gt_b && !cond_b_gt_c && cond_a_gt_c && cond_b_gt_d) {
            out += (qf - fa_) * p0000 + (fa_ - fc_) * p1000 + (fc_ - fb_) * p1010 + (fb_ - fd_) * p1110 + fd_ * p1111;
        }
        if (cond_a_gt_b && !cond_b_gt_c && cond_a_gt_c && !cond_b_gt_d && cond_c_gt_d_alt) {
            out += (qf - fa_) * p0000 + (fa_ - fc_) * p1000 + (fc_ - fd_) * p1010 + (fd_ - fb_) * p1011 + fb_ * p1111;
        }
        if (cond_a_gt_b && !cond_b_gt_c && cond_a_gt_c && !cond_b_gt_d && !cond_c_gt_d_alt && cond_a_gt_d) {
            out += (qf - fa_) * p0000 + (fa_ - fd_) * p1000 + (fd_ - fc_) * p1001 + (fc_ - fb_) * p1011 + fb_ * p1111;
        }
        if (cond_a_gt_b && !cond_b_gt_c && cond_a_gt_c && !cond_b_gt_d && !cond_c_gt_d_alt && !cond_a_gt_d) {
            out += (qf - fd_) * p0000 + (fd_ - fa_) * p0001 + (fa_ - fc_) * p1001 + (fc_ - fb_) * p1011 + fb_ * p1111;
        }
        if (cond_a_gt_b && !cond_b_gt_c && !cond_a_gt_c && cond_b_gt_d) {
            out += (qf - fc_) * p0000 + (fc_ - fa_) * p0010 + (fa_ - fb_) * p1010 + (fb_ - fd_) * p1110 + fd_ * p1111;
        }
        if (cond_a_gt_b && !cond_b_gt_c && !cond_a_gt_c && !cond_b_gt_d && cond_c_gt_d_alt) {
            out += (qf - fc_) * p0000 + (fc_ - fa_) * p0010 + (fa_ - fd_) * p1010 + (fd_ - fb_) * p1011 + fb_ * p1111;
        }
        if (cond_a_gt_b && !cond_b_gt_c && !cond_a_gt_c && !cond_b_gt_d && !cond_c_gt_d_alt && cond_a_gt_d) {
            out += (qf - fc_) * p0000 + (fc_ - fd_) * p0010 + (fd_ - fa_) * p0011 + (fa_ - fb_) * p1011 + fb_ * p1111;
        }
        if (cond_a_gt_b && !cond_b_gt_c && !cond_a_gt_c && !cond_b_gt_d && !cond_c_gt_d_alt && !cond_a_gt_d) {
            out += (qf - fd_) * p0000 + (fd_ - fc_) * p0001 + (fc_ - fa_) * p0011 + (fa_ - fb_) * p1011 + fb_ * p1111;
        }

        if (!cond_a_gt_b && cond_a_gt_c && cond_c_gt_d) {
            out += (qf - fb_) * p0000 + (fb_ - fa_) * p0100 + (fa_ - fc_) * p1100 + (fc_ - fd_) * p1110 + fd_ * p1111;
        }
        if (!cond_a_gt_b && cond_a_gt_c && !cond_c_gt_d && cond_a_gt_d) {
            out += (qf - fb_) * p0000 + (fb_ - fa_) * p0100 + (fa_ - fd_) * p1100 + (fd_ - fc_) * p1101 + fc_ * p1111;
        }
        if (!cond_a_gt_b && cond_a_gt_c && !cond_c_gt_d && !cond_a_gt_d && cond_b_gt_d) {
            out += (qf - fb_) * p0000 + (fb_ - fd_) * p0100 + (fd_ - fa_) * p0101 + (fa_ - fc_) * p1101 + fc_ * p1111;
        }
        if (!cond_a_gt_b && cond_a_gt_c && !cond_c_gt_d && !cond_a_gt_d && !cond_b_gt_d) {
            out += (qf - fd_) * p0000 + (fd_ - fb_) * p0001 + (fb_ - fa_) * p0101 + (fa_ - fc_) * p1101 + fc_ * p1111;
        }
        if (!cond_a_gt_b && !cond_a_gt_c && cond_b_gt_c2 && cond_a_gt_d) {
            out += (qf - fb_) * p0000 + (fb_ - fc_) * p0100 + (fc_ - fa_) * p0110 + (fa_ - fd_) * p1110 + fd_ * p1111;
        }
        if (!cond_a_gt_b && !cond_a_gt_c && cond_b_gt_c2 && !cond_a_gt_d && cond_c_gt_d_alt) {
            out += (qf - fb_) * p0000 + (fb_ - fc_) * p0100 + (fc_ - fd_) * p0110 + (fd_ - fa_) * p0111 + fa_ * p1111;
        }
        if (!cond_a_gt_b && !cond_a_gt_c && cond_b_gt_c2 && !cond_a_gt_d && !cond_c_gt_d_alt && cond_b_gt_d) {
            out += (qf - fb_) * p0000 + (fb_ - fd_) * p0100 + (fd_ - fc_) * p0101 + (fc_ - fa_) * p0111 + fa_ * p1111;
        }
        if (!cond_a_gt_b && !cond_a_gt_c && cond_b_gt_c2 && !cond_a_gt_d && !cond_c_gt_d_alt && !cond_b_gt_d) {
            out += (qf - fd_) * p0000 + (fd_ - fb_) * p0001 + (fb_ - fc_) * p0101 + (fc_ - fa_) * p0111 + fa_ * p1111;
        }
        if (!cond_a_gt_b && !cond_a_gt_c && !cond_b_gt_c2 && cond_a_gt_d) {
            out += (qf - fc_) * p0000 + (fc_ - fb_) * p0010 + (fb_ - fa_) * p0110 + (fa_ - fd_) * p1110 + fd_ * p1111;
        }
        if (!cond_a_gt_b && !cond_a_gt_c && !cond_b_gt_c2 && !cond_a_gt_d && cond_b_gt_d) {
            out += (qf - fc_) * p0000 + (fc_ - fb_) * p0010 + (fb_ - fd_) * p0110 + (fd_ - fa_) * p0111 + fa_ * p1111;
        }
        if (!cond_a_gt_b && !cond_a_gt_c && !cond_b_gt_c2 && !cond_a_gt_d && !cond_b_gt_d && cond_c_gt_d_alt) {
            out += (qf - fc_) * p0000 + (fc_ - fd_) * p0010 + (fd_ - fb_) * p0011 + (fb_ - fa_) * p0111 + fa_ * p1111;
        }
        if (!cond_a_gt_b && !cond_a_gt_c && !cond_b_gt_c2 && !cond_a_gt_d && !cond_b_gt_d && !cond_c_gt_d_alt) {
            out += (qf - fd_) * p0000 + (fd_ - fc_) * p0001 + (fc_ - fb_) * p0011 + (fb_ - fa_) * p0111 + fa_ * p1111;
        }

        return out / qf;
    }

    private static int msb(float v, int q) {
        return (int) (v / q);
    }

    private static float lsb(float v, int q) {
        return v - msb(v, q) * q;
    }

    static int lutIndex(int a, int b, int c, int d, int L) {
        return a * L * L * L + b * L * L + c * L + d;
    }

    static float lookup(float[] weight, int L, int a, int b, int c, int d) {
        return weight[lutIndex(a, b, c, d, L)];
    }
}
