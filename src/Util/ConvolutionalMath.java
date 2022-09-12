package Util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;

public class ConvolutionalMath {
    public static double[][] correlate2DValid(double[][] input, double[][] kernel) {
        double[][] output = new double[input.length-kernel.length+1][input[0].length-kernel.length+1];

        for (int y = 0; y < output.length; y++) {
            for (int x = 0; x < output[0].length; x++) {
                double val = 0;

                for (int j = 0; j < kernel.length; j++) {
                    for (int i = 0; i < kernel[0].length; i++) {
                        val += input[y+j][x+i] * kernel[j][i];
                    }
                }

                output[y][x] = val;


            }
        }

        return output;
    }

    public static double[][] rotateKernel(double[][] kernel) {
        double[][] out = new double[kernel.length][kernel[0].length];

        for (int i = out.length-1; i >= 0; i--) {
            for (int j = out[0].length-1; j >= 0; j--) {
                //System.out.println(i + ", " + (kernel.length - (i+1)));
                //System.out.println(j + ", " + (kernel[0].length - (j+1)));
                out[i][j] = kernel[kernel.length - (i+1)][kernel[0].length - (j+1)];
            }
        }

        return out;
    }

    public static double[][] convolve2DFull(double[][] input, double[][] kernel) {
        double[][] output = new double[input.length+kernel.length-1][input[0].length+kernel.length-1];

        kernel = rotateKernel(kernel);

        for (int y = -kernel.length+1; y < input.length; y++) {
            for (int x = -kernel[0].length+1; x < input[0].length; x++) {
                double val = 0;

                for (int j = 0; j < kernel.length; j++) {
                    for (int i = 0; i < kernel[0].length; i++) {
                        //System.out.println("(" + y + ", " + x + ")  " + "(" + (y+j) + ", " + (x+i) + ") = " + input[y+j][x+i]);
                        if ( (y+j >= 0 && y+j < input.length) && (x+i >= 0 && x+i < input[0].length)) {

                            val += input[y+j][x+i] * kernel[j][i];
                        }

                    }
                }

                //System.out.println(y+kernel.length-1);
                //System.out.println(x+kernel[0].length-1);
                output[y+kernel.length-1][x+kernel[0].length-1] = val;
                //System.out.println();
            }
        }

        return output;
    }

    public static double[][][] reshape(double[][][] input, Size size) {

        double[][][] output = new double[size.z][size.y][size.x];

        int inputCount = input.length * input[0].length * input[0][0].length;


        /*int z = 0;
        int y = 0;
        int x = 0;

        int i = 0;
        while (i < inputCount) {
            x++;

            if (z % size.z == 0 && z != 0) {
                z++;
                y = 0;
                x = 0;
            }

            if (y % size.y == 0 && y != 0) {
                y++;
                x = 0;
            }

            if (x % size.x == 0 && x != 0) {
                x = 0;
            }
            i++;

            output[z][y][x] = i


        }*/

        int i = 0;
        while (i < inputCount) {
            int ox = i % size.x;
            int oy = (i / size.x) % size.y;
            int oz = i / (size.y * size.x);

            int ix = i % input[0][0].length;
            int iy = (i / input[0][0].length) % input[0].length;
            int iz = i / (input[0][0].length * input[0].length);

            output[oz][oy][ox] = input[iz][iy][ix];

            i++;
        }

        return output;
    }
}
