package Testing;

import Layer.Activation.ReLU;
import Layer.Activation.Sigmoid;
import Layer.Activation.Tanh;
import Layer.Dense;
import Layer.Layer;
import Layer.LayerInterface;
import Loss.MSE;
import Network.Network;

public class NetworkTest {
    public static void main(String[] args) {

        double[][] xTrain = new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] yTrain = new double[][] {{0}, {1}, {1}, {0}};

        //double[][] xTrain = new double[][] {{0}, {1}, {2}, {3}};
        //double[][] yTrain = new double[][] {{0}, {5}, {10}, {15}};

        Network network = new Network(
            new Layer[] {
                new Dense(2, 3),
                new Tanh(),
                new Dense(3, 1),
                new Tanh()
            },
            xTrain,
            yTrain,
            0.1,
            new MSE()
        );

        network.Train(10000);

        for (int i = 0; i < xTrain.length; i++) {
            double[] actual = network.predict(xTrain[i]);

            System.out.println("The network was given { " + printArr(xTrain[i]) + "}. The expected output was " + yTrain[i][0] + " and the actual output was " + printArr(actual));

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
