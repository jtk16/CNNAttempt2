package Layer;

import java.lang.reflect.Array;

public class Layer implements LayerInterface {

    public double[] inputs;
    public double[] outputs;

    @Override
    public double[] feedForward(double[] fInput) {
        return new double[0];
    }

    public double[][][] feedForward(double[][][] fInput) {
        return new double[0][0][0];
    }

    @Override
    public double[] backPropagate(double[] incomingGradient, double l_rate) {
        return new double[0];
    }

    public double[][][] backPropagate(double[][][] incomingGradient, double l_rate) {return new double[0][0][0];}


}
