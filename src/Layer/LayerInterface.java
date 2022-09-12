package Layer;

public interface LayerInterface {

    double[] feedForward(double[] fInput);

    double[] backPropagate(double[] incomingGradient, double l_rate);

}
