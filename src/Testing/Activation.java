package Testing;

import Layer.Activation.ReLU;
import Layer.Layer;
import Layer.Dense;
import Layer.Activation.Tanh;
import Loss.MSE;
import Network.Network;

public class Activation {
    public static void main(String[] args) {
        double[][] xTrain = new double[][] {{-1}, {-2}, {-3}, {0}, {1}, {2}, {3}};
        //double[][] yTrain = new double[][] {{2*Math.tanh(-1)}, {2*Math.tanh(-2)}, {2*Math.tanh(-3)}, {2*Math.tanh(0)}, {2*Math.tanh(1)}, {2*Math.tanh(2)}, {2*Math.tanh(3)}};
        double[][] yTrain = Network.generateDataSet(xTrain, (a) -> a*a);

        //double[][] yTrain = new double[][] {{5}, {10}, {15}};
        Network network = new Network(new Layer[] {
                new Dense(1, 5),
                //new Dense(3, 1)
                new Tanh(),
                new Dense(5, 10),
                new Tanh(),
                new Dense(10, 1),

        },
                xTrain,
                yTrain,
                0.01,
                new MSE());

        network.Train(1000);

        for (int i = 0; i < xTrain.length; i++) {
            double[] actual = network.predict(xTrain[i]);

            System.out.println("The network was given { " + printArr(xTrain[i]) + "}. The expected output was " + yTrain[i][0] + " and the actual output was " + printArr(actual));

        }

        network.print2DDataSet();
    }

    public static String printArr(double[] arr) {
        String output = "";
        for (int i = 0; i < arr.length; i++) {
            output += arr[i] + " ";
        }

        return output;
    }
}
