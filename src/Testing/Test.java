package Testing;

import Layer.Convolutional;
import Util.ConvolutionalMath;
import Util.Size;
import sun.nio.ch.Net;

public class Test {
    public static void main(String[] args) {

        /*double[][] input = new double[][] {
                {1, 4, 7},
                {2, 5, 8},
                {3, 6, 9}
        };

        double[][] kernel = new double[][] {
                {1, 2},
                {3, 4}
        };

        //5 - 6 + 7 - 18 + 25 - 12 + 4 - 4 + 1
        //double[][] out = ConvolutionalMath.correlate2DValid(input, kernel);
        double[][] out = ConvolutionalMath.convolve2DFull(input, kernel);

        for (double[] cArr : out) {
            System.out.println(NetworkTest.printArr(cArr));
        }
         */

        double[][][] input = {
                {
                        {1, 2, 3},
                        {4, 5, 6}
                },
                {
                        {7, 8, 9},
                        {10, 11, 12}
                }
        };

        for (double[][] tArr : input) {
            for (double[] oArr : tArr) {
                System.out.println(NetworkTest.printArr(oArr));
            }
            System.out.println();
        }

        input = ConvolutionalMath.reshape(input, new Size(3, 2, 2));

        for (double[][] tArr : input) {
            for (double[] oArr : tArr) {
                System.out.println(NetworkTest.printArr(oArr));
            }
            System.out.println();
        }
    }
}
