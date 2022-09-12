package Testing;

import Layer.Dense;
import Loss.MSE;
import sun.jvm.hotspot.oops.OopUtilities;

public class DenseTest {
    public static void main(String[] args) {
        Dense dense = new Dense(2, 1);
        MSE loss = new MSE();


        /*for (int j = 0; j < dense.weights.length; j++) {
            for (int i = 0; i < dense.weights[j].length; i++) {
                System.out.print(dense.weights[j][i] + ", ");
            }
        }*/

        double[][] xTrain = new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] yTrain = new double[][] {{0}, {1}, {1}, {0}};

        for (int e = 0; e < 1000; e++) {
            double epochError = 0;
            for (int sample = 0; sample < xTrain.length; sample++) {
                double[] ff = xTrain[sample];
                ff = dense.feedForward(ff);

                double error = loss.forward(yTrain[sample], ff);
                epochError += error;
                double[] mse_prime = loss.back(yTrain[sample], ff);

                dense.backPropagate(mse_prime, 0.01);
            }

            System.out.println(epochError/ xTrain.length + ", epoch: " + e );
        }

        for (int i = 0; i < xTrain.length; i++) {
            double[] actual = dense.feedForward(xTrain[i]);

            System.out.println(" The network was given { " + printArr(xTrain[i]) + "}. The expected output was " + yTrain[i][0] + " and the actual output was " + printArr(actual));

        }
    }

    public static String printArr(double[] arr) {
        String output = "";
        for (int i = 0; i < arr.length; i++) {
            output += arr[i] + " ";
        }

        return output;
    }
}
