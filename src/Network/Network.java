package Network;

import Layer.Layer;
import Layer.LayerInterface;
import Loss.Loss;
import Testing.Activation;
import Testing.NetworkTest;

import java.util.function.Function;

public class Network {
    public Layer[] network;

    public double l_rate;

    public double[][] xTrain;
    public double[][] yTrain;

    public Loss loss;

    public Network(Layer[] network, double[][] xTrain, double[][] yTrain, double l_rate, Loss loss) {
        this.network = network;

        this.xTrain = xTrain;
        this.yTrain = yTrain;

        this.l_rate = l_rate;

        this.loss = loss;
    }

    public void Train(int epochs) {
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double epochError = 0;
            for (int sample = 0; sample < xTrain.length; sample++) {
                double[] ff = xTrain[sample];

                ff = predict(ff);

                epochError += loss.forward(yTrain[sample], ff);

                double[] gradient = loss.back(yTrain[sample], ff);

                for (int layerIdx = network.length - 1; layerIdx >= 0; layerIdx--) {
                    gradient = network[layerIdx].backPropagate(gradient, l_rate);
                    //System.out.println(network[layerIdx].getClass().getName() + "'s gradient is: " + Activation.printArr(gradient));
                }

            }
            System.out.println(epochError/xTrain.length + ", epoch: " + epoch);
        }
    }

    public double[] predict(double[] ff) {
        for (LayerInterface layer : network) {
            ff = layer.feedForward(ff);
        }

        return ff;
    }

    public static double[][] generateDataSet(double[][] xTrain, Function<Double, Double> func) {
        double[][] out = new double[xTrain.length][xTrain[0].length];

        for (int j = 0; j < xTrain.length; j++) {
            for (int i = 0; i < xTrain[j].length; i++) {
                out[j][i] = func.apply(xTrain[j][i]);
            }
        }

        return out;
    };

    public void print2DDataSet() {
        for (double[] arr : xTrain) {
            System.out.println("(" + arr[0] + ", " + predict(arr)[0] + ")");
        }
    }


}
